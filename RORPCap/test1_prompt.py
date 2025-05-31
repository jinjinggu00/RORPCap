import pickle

import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image
from IPython.display import Image
from enum import Enum

#import model1_mapping

from mamba_ssm.models.mixer_seq_simple import MixerModel

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]
D = torch.device
CPU = torch.device('cpu')

def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')
gpt2_type = 'gpt2'
CUDA = get_device
current_directory = os.getcwd()
#coco_prefix-000
model_path = os.path.join('checkpoints/', 'coco_prefix-005.pt')
#model_weights.pt
# @title Model

class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix)#.view(-1, self.prefix_length, self.gpt_embedding_size)

        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, ):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_type)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.prefix_length = prefix_length

        self.clip_project = MixerModel(d_model=768, n_layer=10, d_intermediate=0, seq_len=self.prefix_length)

        self.freeze_last_n_layers(1, 8)

    def freeze_last_n_layers(self, n, n2):

        total_layers = len(self.gpt.transformer.h)  # total 36
        assert n <= total_layers, f"Number of layers to freeze exceeds total layers in GPT-2 ({total_layers})"

        for i in range(n, n2):
            for param in self.gpt.transformer.h[i].parameters():
                param.requires_grad = False

class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed#(1,40,1280)
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits#(1,40,50257)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)#(1,50257)
            logits = logits.softmax(-1).log()#(1,50257)
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)#(1,5) (1,5)
                generated = generated.expand(beam_size, *generated.shape[1:])#(5,40,1280)
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits#(5,50257)
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]#(5)
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]#(5,)
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]#(5,1)
                tokens = torch.cat((tokens, next_tokens), dim=1)#(5,2)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths#(5,)
                is_stopped = is_stopped[next_tokens_source]#(5,)
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)#(5,1,1280)
            generated = torch.cat((generated, next_token_embed), dim=1)#(5,40- ,1280)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()#(5,)
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

is_gpu = True


device = CUDA(0) if is_gpu else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)

prefix_length = 10

#model = ClipCaptionPrefix(prefix_length, clip_length=10, prefix_size=512,)
model = ClipCaptionModel(prefix_length, clip_length=5, prefix_size=512,)
model.load_state_dict(torch.load(model_path, map_location=CPU))
use_beam_search = True
model = model.eval()
device = CUDA(0) if is_gpu else "cpu"
model = model.to(device)

######################################

data_path ='/home/l3408/Q/smallcap/coco-caption/annotations/captions_testKarpathy.json'

with open(data_path, 'r') as file:
    data = json.load(file)

data = data['annotations']
seen = set()  # Set to keep track of unique image IDs
unique_image_ids = []  # List to store unique image IDs

# Iterate through the annotations to collect unique image IDs
for item in data:
    image_id = item["image_id"]
    if image_id not in seen:
        seen.add(image_id)  # Add the image ID to the set
        unique_image_ids.append(image_id)  # Append to the list of unique IDs

out = []
prefix_out = []
k = 0
gpt_tensor = GPT2LMHeadModel.from_pretrained(gpt2_type)
with open('objects_relations.json', 'r') as file:
    data = json.load(file)

for j in range(len(unique_image_ids)):

    first_prompt = (
                    'a photo contains objects: ' +
                      str(data[str(unique_image_ids[j])][0][0]) + ', '
                    + str(data[str(unique_image_ids[j])][0][1]) + ', '
                    + str(data[str(unique_image_ids[j])][0][2]) + ', '
                    + str(data[str(unique_image_ids[j])][0][3]) + ', '
                    + str(data[str(unique_image_ids[j])][0][4]) + ', '
                    + str(data[str(unique_image_ids[j])][0][5]) + '. '
                    'and the relations are ' +
                    str(data[str(unique_image_ids[j])][1][0]) + ', '
                    + str(data[str(unique_image_ids[j])][1][1]) + ', '
                    + str(data[str(unique_image_ids[j])][1][2]) +
                    '. its caption is '
                    )
    first_prompt_token = torch.tensor(tokenizer.encode(first_prompt), dtype=torch.int64).to(device)

    aa = str(unique_image_ids[j]).zfill(12)
    k = k+1

    UPLOADED_FILE = 'data/coco/' + aa + '.jpg'
    print(UPLOADED_FILE, k)

    image = io.imread(UPLOADED_FILE)
    pil_image = PIL.Image.fromarray(image)  # Convert to PIL Image for processing

    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)

        prefix_embed1 = model.clip_project(prefix)

    prefix_embed2 = model.gpt.transformer.wte(first_prompt_token).unsqueeze(0).to(device)

    prefix_embed = torch.cat((prefix_embed2, prefix_embed1), dim=1)

    if use_beam_search:
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]

    print(int(unique_image_ids[j]), generated_text_prefix)

    out.append({"image_id": int(unique_image_ids[j]), "caption": generated_text_prefix})

# Save the generated captions to a JSON file
with open('checkpoints/result.json', 'w') as file:
    json.dump(out, file, indent=4)



