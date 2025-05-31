import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from mamba_ssm.models.mixer_seq_simple import MixerModel

device = torch.device('cuda:0')
gpt2_type = 'gpt2'
class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        # mask = torch.cat((torch.zeros(40), torch.ones(self.prefix_length), mask), dim=0)  #  mask image prompt, but do not mask words prompt, bert
        mask = torch.cat((torch.ones(50), torch.ones(self.prefix_length), mask), dim=0)  # mask all prompt,gpt2
        return tokens, mask

    def pad_tokens_first_prompt(self, item: int):
        first_token = self.first_prompts[item]
        padding = 50 - first_token.shape[0]
        first_token = torch.cat((first_token, torch.zeros(padding, dtype=torch.int64) - 1))
        self.first_prompts[item] = first_token
        mask = first_token.ge(0)  # mask is zero where we out of sequence
        first_token[~mask] = 0
        return first_token

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]

        first_prompt = self.pad_tokens_first_prompt(item)

        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)

        return tokens, mask, prefix, first_prompt

    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = gpt2_type, normalize_prefix=False):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)


        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix


        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)


        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()


        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]


        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]


        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl") and os.path.isfile(f"{data_path[:-4]}_first_prompt_tokens.pkl"):

            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)

            with open(f"{data_path[:-4]}_first_prompt_tokens.pkl", 'rb') as f:
                self.first_prompts, self.caption2embedding, self.max_seq_len = pickle.load(f)

        else:

            self.captions_tokens = []
            self.caption2embedding = []
            self.first_prompts = []

            max_seq_len = 0
            with open('objects_relations.json', 'r') as file:
                data = json.load(file)

            for caption in captions_raw:

                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))

                self.caption2embedding.append(caption["clip_embedding"])

                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])

                first_prompt = (
                                'a photo contains objects: ' +
                                str(data[str(caption['image_id'])][0][0]) + ', ' +
                                str(data[str(caption['image_id'])][0][1]) + ', ' +
                                str(data[str(caption['image_id'])][0][2]) + ', ' +
                                str(data[str(caption['image_id'])][0][3]) + ', ' +
                                str(data[str(caption['image_id'])][0][4]) + ', ' +
                                str(data[str(caption['image_id'])][0][5]) + ', ' +
                                'and the relations are ' +
                                str(data[str(caption['image_id'])][1][0]) + ', ' +
                                str(data[str(caption['image_id'])][1][1]) + ', ' +
                                str(data[str(caption['image_id'])][1][2]) +
                                '. its caption is'
                                )
                self.first_prompts.append(torch.tensor(self.tokenizer.encode(first_prompt), dtype=torch.int64))

            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)

            with open(f"{data_path[:-4]}_first_prompt_tokens.pkl", 'wb') as f:
                pickle.dump([self.first_prompts, self.caption2embedding, max_seq_len], f)

        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()

        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, first_prompt: torch.Tensor,
                mask: Optional[torch.Tensor] = None,labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        first_prompt_embed = self.gpt.transformer.wte(first_prompt)
        prefix_projections = self.clip_project(prefix)
        embedding_cat = torch.cat((first_prompt_embed, prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, args):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_type)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.batch_size = args.bs

        self.clip_project = MixerModel(d_model=768, n_layer=10, d_intermediate=0, seq_len=self.prefix_length)


        self.freeze_last_n_layers(1, 8)

    def freeze_last_n_layers(self, n, n2):

        total_layers = len(self.gpt.transformer.h)  # total 3.6
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

def train(dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 9e-6, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader))

    for epoch in range(epochs):
        print(epoch)
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix, first_prompt) in enumerate(train_dataloader):

            optimizer.zero_grad()
            tokens, mask, prefix, first_prompt = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32), first_prompt.to(device)
            outputs = model(tokens, prefix, first_prompt, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1 + 50: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='oscar_split_ViT-B_32_train.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true',default=False)
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, args,clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, args,clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                 num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)

if __name__ == '__main__':
    main()
