import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from .preprocess import quick_df_clean
import glob

import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


class DFPredictor:
    def __init__(self, df_path: str, model_path: str, device: str = 'cuda', ensemble=True):
        self.data = pd.read_csv(df_path, sep='|')
        self.data['text'] = self.data['text'].apply(lambda x: quick_df_clean(x))
        self.id2label = {0: 'INSULT', 1: 'OTHER',
                         2: 'PROFANITY', 3: 'RACIST', 4: 'SEXIST'}
        self.label2id = {v: k for k, v in self.id2label.items()}
        if ensemble:
            self.models = glob.glob(model_path + "/*", recursive=False)
        else:
            self.models = [model_path]
        self.device = device

    def predict_df(self, save_csv=False):

        print(f"Predicting for given model weights:\n\tTotal Number of Models: {len(self.models)}")
        final_preds = []
        for model_name in self.models:
            print(f"\n\t Predicting for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            dataset = PredictDataset(self.data, tokenizer, max_len=64)
            data_collator = DataCollatorWithPadding(
                tokenizer, padding="longest")
            dataloader = DataLoader(
                dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
            predictions = []
            model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                       problem_type="single_label_classification",
                                                                       id2label=self.id2label,
                                                                       label2id=self.label2id,
                                                                       num_labels=5,
                                                                       output_hidden_states=False,
                                                                       ignore_mismatched_sizes=True
                                                                       ).to(self.device)
            model.eval()
            for encoding in tqdm(dataloader):
                output = model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(
                    self.device), encoding['token_type_ids'].to(self.device))
                predictions.append(output.logits.detach().cpu())
            final_preds.append(torch.softmax(torch.cat(predictions), dim=-1))
        final_preds = np.mean((torch.stack(final_preds)).numpy(), axis=0)
        new_df = self.data.copy()
        new_df['is_offensive'] = 0
        new_df.loc[:, "target"] = np.argmax(final_preds, axis=-1)
        new_df.loc[:, 'is_offensive'] = np.where(new_df.target == 1, 0, 1)
        new_df.loc[:, "target"] = new_df['target'].map(self.id2label)
        if save_csv:
            new_df.to_csv('predictions.csv', index=False, sep='|')
        return new_df


class Predictor:
    def __init__(self, texts: list, model_path: str, device: str = 'cuda', ensemble=True):

        self.data = texts
        self.data = [quick_df_clean(text) for text in texts]
        self.id2label = {0: 'INSULT', 1: 'OTHER',
                         2: 'PROFANITY', 3: 'RACIST', 4: 'SEXIST'}
        self.label2id = {v: k for k, v in self.id2label.items()}
        if ensemble:
            self.models = glob.glob(model_path + "/*", recursive=False)
        else:
            self.models = [model_path]
        self.device = device

    def predict(self):
        print(f"Predicting for given model weights:\n\tTotal Number of Models: {len(self.models)}")
        final_preds = []
        for model_name in self.models:
            print(f"\n\t Predicting for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            dataset = PredictDataset(self.data, tokenizer, max_len=64)
            data_collator = DataCollatorWithPadding(
                tokenizer, padding="longest")
            dataloader = DataLoader(
                dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
            predictions = []
            model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                       problem_type="single_label_classification",
                                                                       id2label=self.id2label,
                                                                       label2id=self.label2id,
                                                                       num_labels=5,
                                                                       output_hidden_states=False,
                                                                       ignore_mismatched_sizes=True
                                                                       ).to(self.device)
            model.eval()
            for encoding in tqdm(dataloader):
                output = model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(
                    self.device), encoding['token_type_ids'].to(self.device))
                predictions.append(output.logits.detach().cpu())
            final_preds.append(torch.softmax(torch.cat(predictions), dim=-1))
        probas = np.mean((torch.stack(final_preds)).numpy(), axis=0)
        predictions = np.argmax(probas, axis=-1)
        predicted_labels = [self.id2label[i] for i in predictions]
        return {"probas": probas, "predictions": predictions, "predicted_labels": predicted_labels}


class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.data, pd.DataFrame):
            row = self.data.iloc[idx]
            text = row.text
        else:
            text = self.data[idx]

        encoding = self.tokenizer(
            text, max_length=self.max_len, truncation=True)
        encoding = {key: torch.tensor(val, dtype=torch.int64)
                    for key, val in encoding.items()}
        return dict(encoding)
