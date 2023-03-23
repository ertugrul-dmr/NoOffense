import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from .utils.preprocess import quick_df_clean


class DFPredictor:
    def __init__(self, df_path: str, model_path: str):
        self.df = pd.read_csv(df_path)
        self.df['text'] = self.df['text'].apply(lambda x: quick_df_clean(x))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        problem_type="single_label_classification",
                                                                        id2label={0: 'INSULT', 1: 'OTHER',
                                                                                  2: 'PROFANITY', 3: 'RACIST',
                                                                                  4: 'SEXIST'},
                                                                        label2id={'INSULT': 0, 'OTHER': 1,
                                                                                  'PROFANITY': 2, 'RACIST': 3,
                                                                                  'SEXIST': 4},
                                                                        num_labels=5,
                                                                        output_hidden_states=False,
                                                                        ignore_mismatched_sizes=True

                                                                        )
        self.dataset = PredictDataset(self.df, self.tokenizer, max_len=64)
        self.data_collator = DataCollatorWithPadding(self.tokenizer, padding="longest")
        self.dataloader = DataLoader(pd, batch_size=8, shuffle=False, collate_fn=self.data_collator)
    @property
    def predict_df(self):
        predictions = []
        for encoding in tqdm(self.dataloader):
            logits = self.model(**encoding).logits
            preds = torch.argmax(logits, axis=1)
            predictions.extend(preds.tolist())
        return predictions



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
