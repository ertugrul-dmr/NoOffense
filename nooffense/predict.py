import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, FillMaskPipeline, \
    AutoModelForMaskedLM
from .utils import quick_clean, PredictDataset
import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


class DFPredictor:
    def __init__(self, model_path, device: str = 'cpu', weights=None):
        """A class to make predictions on a dataframe of texts using one or more Hugging Face Transformers models."""
        # Mapping of label indices to label names
        self.id2label = {0: 'INSULT', 1: 'OTHER',
                         2: 'PROFANITY', 3: 'RACIST', 4: 'SEXIST'}
        # Mapping of label names to label indices
        self.label2id = {v: k for k, v in self.id2label.items()}
        # Store model paths
        if isinstance(model_path, list):
            self.models = model_path
        else:
            self.models = [model_path]
        # Store device to use
        self.device = device
        # Store weights to apply to predictions
        self.weights = weights

    def predict_df(self, df_path: str, max_len=64, batch_size=8, save_csv=False, progress_bar=True):
        """Predict the offensive language labels for a dataframe of texts using one or more Hugging Face Transformers models."""

        print(
            f"Predicting for given model weights:\n\tTotal Number of Models: {len(self.models)}")
        # Read input dataframe
        data = pd.read_csv(df_path, sep='|')
        # Clean input texts using a helper function
        data['text'] = data['text'].apply(
            lambda x: quick_clean(x))
        # List to store predictions from all models
        final_preds = []
        for model_name in self.models:
            print(f"\n\t Predicting for {model_name}")
            # Get tokenizer for the current model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Create a PredictDataset for the input dataframe
            dataset = PredictDataset(data, tokenizer, max_len=max_len)
            # Create a DataCollatorWithPadding for batching
            data_collator = DataCollatorWithPadding(
                tokenizer, padding="longest")
            # Create a DataLoader for the dataset
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
            # List to store predictions from the current model
            predictions = []
            # Load the current model and set it to evaluation mode
            model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                       problem_type="single_label_classification",
                                                                       id2label=self.id2label,
                                                                       label2id=self.label2id,
                                                                       num_labels=5,
                                                                       output_hidden_states=False,
                                                                       ignore_mismatched_sizes=True
                                                                       ).to(self.device)
            model.eval()
            # Iterate over batches in the dataloader and make predictions
            for encoding in tqdm(dataloader, disable=not progress_bar):
                try:
                    output = model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(
                        self.device), encoding['token_type_ids'].to(self.device))
                except:
                    output = model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(
                        self.device))
                predictions.append(output.logits.detach().cpu())
            # Storing softmax probabilities for current model
            final_preds.append(torch.softmax(torch.cat(predictions), dim=-1))
        # Averaging softmax probabilities for all models using given weights
        final_preds = np.average((torch.stack(final_preds)).numpy(), axis=0, weights=self.weights)
        # Creating a new dataframe for predictions
        new_df = data.copy()
        new_df['is_offensive'] = 0
        # Assigning predicted target label and offensive status to each row of dataframe
        new_df.loc[:, "target"] = np.argmax(final_preds, axis=-1)
        new_df.loc[:, 'is_offensive'] = np.where(new_df.target == 1, 0, 1)
        # Mapping target labels
        new_df.loc[:, "target"] = new_df['target'].map(self.id2label)
        # Saving dataframe as csv file if required
        if save_csv:
            new_df.to_csv('predictions.csv', index=False, sep='|')
        # Returning the final dataframe with predicted labels and offensive status
        return new_df


class Predictor:
    def __init__(self, model_path, device: str = 'cpu', weights=None):
        """
         A class to make predictions on a given list of texts using one or more Hugging Face Transformers models.
        """
        # Mapping of label indices to label names
        self.id2label = {0: 'INSULT', 1: 'OTHER',
                         2: 'PROFANITY', 3: 'RACIST', 4: 'SEXIST'}
        # Mapping of label names to label indices
        self.label2id = {v: k for k, v in self.id2label.items()}
        # Store model paths
        if isinstance(model_path, list):
            self.models = model_path
        else:
            self.models = [model_path]
        self.device = device
        # Store weights to apply to predictions
        self.weights = weights

    def predict(self, texts: list, max_len=64, batch_size=8, progress_bar=False):
        """
        Generates predictions for the provided text.
        """
        print(
            f"Predicting for given model weights:\n\tTotal Number of Models: {len(self.models)}")
        # Clean input texts using a helper function
        data = [quick_clean(text) for text in texts]
        # List to store predictions from all models
        final_preds = []
        for model_name in self.models:
            print(f"\n\t Predicting for {model_name}")
            # Get tokenizer for the current model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Create a PredictDataset for the input dataframe
            dataset = PredictDataset(data, tokenizer, max_len=max_len)
            # Create a DataLoader for the dataset
            data_collator = DataCollatorWithPadding(
                tokenizer, padding="longest")
            # Create a DataLoader for the dataset
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
            predictions = []
            # Load the current model and set it to evaluation mode
            model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                       problem_type="single_label_classification",
                                                                       id2label=self.id2label,
                                                                       label2id=self.label2id,
                                                                       num_labels=5,
                                                                       output_hidden_states=False,
                                                                       ignore_mismatched_sizes=True
                                                                       ).to(self.device)
            model.eval()
            # Iterate over batches in the dataloader and make predictions
            for encoding in tqdm(dataloader, disable=not progress_bar):
                try:
                    output = model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(
                        self.device), encoding['token_type_ids'].to(self.device))
                except:
                    output = model(encoding['input_ids'].to(self.device), encoding['attention_mask'].to(
                        self.device))
                predictions.append(output.logits.detach().cpu())
            # Storing softmax probabilities for current model
            final_preds.append(torch.softmax(torch.cat(predictions), dim=-1))
        # Averaging softmax probabilities for all models using given weights
        probas = np.average((torch.stack(final_preds)).numpy(), axis=0, weights=self.weights)
        predictions = np.argmax(probas, axis=-1)
        predicted_labels = [self.id2label[i] for i in predictions]
        return {"probas": probas, "predictions": predictions, "predicted_labels": predicted_labels}


class MaskPredictor:
    def __init__(self, model_path):
        """
        Constructor method to initialize the MaskPredictor class with the given pre-trained model.
        """
        # Load pre-trained language model
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        # Load pre-trained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Create FillMaskPipeline object
        self.pipe = FillMaskPipeline(model=self.model, tokenizer=self.tokenizer)

    def mask_filler(self, text):
        """
        Method to fill the masked words in the given text.
        """
        # Fill masked words in the given text
        filled = self.pipe(text)
        return filled
