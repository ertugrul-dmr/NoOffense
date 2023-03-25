import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from .utils.preprocess import quick_df_clean
import glob

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


class DFPredictor:
    def __init__(self, df_path: str, model_path: str, device:str='cuda'):
        self.dirname = os.path.dirname(__file__)
        self.df = pd.read_csv(os.path.join(self.dirname, df_path), sep='|')
        self.df['text'] = self.df['text'].apply(lambda x: quick_df_clean(x))
        self.id2label = {0: 'INSULT', 1: 'OTHER',2: 'PROFANITY', 3: 'RACIST',4: 'SEXIST'}
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.models = glob.glob(os.path.join(self.dirname, model_path)+"/*", recursive=False)
        self.device = device
    @property
    def predict_df(self):
        final_preds = []
        for model_name in self.models:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            dataset = PredictDataset(self.df, tokenizer, max_len=64)
            data_collator = DataCollatorWithPadding(tokenizer, padding="longest")
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
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
                output = model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device), encoding['token_type_ids'].to(self.device))
                predictions.append(output.logits.detach().cpu())
            final_preds.append(torch.softmax(torch.cat(predictions), dim=-1))
        final_preds = np.mean((torch.stack(final_preds)).numpy(), axis=0)
        new_df = self.df.copy()
        new_df['is_offensive']  = 0
        new_df.loc[:, "target"] = np.argmax(final_preds, axis=-1)
        new_df.loc[:, 'is_offensive'] = np.where(new_df.target ==1, 0, 1)
        new_df.loc[:, "target"] =  new_df['target'].map(self.id2label)
        return new_df



class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row.text
        encoding = self.tokenizer(text, max_length=self.max_len, truncation=True)
        encoding = {key: torch.tensor(val, dtype=torch.int64) for key, val in encoding.items()}
        return dict(encoding)
