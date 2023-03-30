import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from .utils import quick_clean, PredictDataset

import numpy as np
from numpy.linalg import norm

import os

from transformers import logging

logging.set_verbosity_error()

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


class MeanPooling(torch.nn.Module):
    """
    This class defines a PyTorch module that implements mean pooling over the input embeddings.
    The `forward` method takes in `last_hidden_state` and `attention_mask` as inputs and returns the mean-pooled embeddings.
    """
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
    """
    This class provides a sentence encoder that utilizes a pre-trained Transformer model to encode
    input text into dense vector representations.
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = model_path
        self.device = device
        self.pool = MeanPooling()
    def encode(self, texts: list, progress_bar=False):

        """
        Encodes a list of input texts into dense vector representations using the pre-trained Transformer model.
        """

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
            try:
                output = model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(
                    self.device), encoding['token_type_ids'].to(self.device))
            except:
                output = model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(
                    self.device))

            mean_pool = self.pool(output.last_hidden_state, encoding['attention_mask'].to(
                self.device))


            predictions.append(mean_pool.detach().cpu())
        encodings = np.concatenate(predictions, axis=0)
        return encodings

    def pair_similarity(self, text_1, text_2):
        """
        Calculates the cosine similarity between each pair of texts in text_1 and text_2.
        """

        cossims = []
        pairs = []
        for t1, t2 in zip(text_1, text_2):
            text_1_emb = self.encode([t1])[0]
            text_2_emb = self.encode([t2])[0]
            cos = self._cos_sim(text_1_emb, text_2_emb)
            cossims.append(cos)
            pairs.append((t1,t2))
        data_dict= dict(zip(pairs, cossims))
        return data_dict

    def _cos_sim(self, A, B):
        """
        Calculates the cosine similarity between two vectors A and B.
        """
        cosine = np.dot(A, B) / (norm(A) * norm(B))
        return cosine

    def find_most_similar(self, input_text, text_to_look):
        """
        Finds the text in text_to_look that is most similar to input_text based on cosine similarity.
        """
        input_embed = self.encode([input_text])[0]

        texts = []
        cossims = []

        for text in text_to_look:
            text_embed = self.encode([text])[0]
            cos = self._cos_sim(input_embed, text_embed)

            texts.append(text)
            cossims.append(cos)

        data_dict = dict(zip(texts, cossims))
        sorted_dict = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_dict

