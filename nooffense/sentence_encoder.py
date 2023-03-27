import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from .utils import quick_clean, PredictDataset

import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


class MeanPooling(torch.nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class SentenceEncoder:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model = model_path
        self.device = device
        self.pool = MeanPooling()
    def encode(self, texts: list, progress_bar=False):

        data = [quick_clean(text) for text in texts]
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        dataset = PredictDataset(data, tokenizer, max_len=64)
        data_collator = DataCollatorWithPadding(
            tokenizer, padding="longest")
        dataloader = DataLoader(
            dataset, batch_size=8, shuffle=False, collate_fn=data_collator)


        predictions = []
        model = AutoModel.from_pretrained(self.model,

                                                                   ).to(self.device)
        model.eval()
        for encoding in tqdm(dataloader, disable=not progress_bar):
            output = model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(
                self.device), encoding['token_type_ids'].to(self.device))


            mean_pool = self.pool(output.last_hidden_state, encoding['attention_mask'].to(
                self.device))


            predictions.append(mean_pool.detach().cpu())
        encodings = np.concatenate(predictions, axis=0)
        return encodings