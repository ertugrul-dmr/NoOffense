# NoOffense
A smart and fast offensive language detector for Turkish using transformers.

This package is built by **Overfit-GM** team as a part of the **Türkiye Açık Kaynak Platformu** NLP challenge.


## How to Install

```python
pip install git+https://github.com/ertugrul-dmr/NoOffense.git
```

# Usage

NoOffense supports multiple tasks in your standard NLP tasks, especially focused on Offensive Language in Turkish texts.

# Get Predictions
You can feed your list of texts and get predictions easily with NoOffense:
```python
from noofense import Predictor

# Predictor applies some text cleaning&preps internally
# In case of multiple trained models, use ensemble=True,
# and give path to weights folder
# you can use gpu support by setting device for faster inference

clf = Predictor(model_path='path_to_your_model_weight_folder', ensemble=True, device='cuda')
predictions= clf.predict(['sen ne gerizekalısın ya'], progress_bar=False)

# Code above will return dict contains these keys:
# ['probas']: probabilities of each class
# ['predictions']: label encoding for each predicted class
# ['predicted_labels']: str label mappings for each class, INSULT, Racist etc.
```
NoOffense supports pandas dataframe predictions too:

It expects column named "text" to make predictions.
```python
from noofense import DFPredictor

clf = DFPredictor(model_path='path_to_your_model_weights_folder', ensemble=True, device='cpu')
predicted_df = clf.predict_df(df_path="path_to_your_csv_file", save_csv=False, progress_bar=True)
# Code above will return a copy of original dataframe with predictions.
```

## Fill Masks

NoOffense models trained in self supervised fashion with large Turkish offensive text corpus. These models then finetuned for text classification task, but byproduct of our Masked Language Model training you can use it as fill mask pipe too!
```python
from nooffense import MaskPredictor
model = MaskPredictor("path_to_your_model_weight_folder")
model.mask_filler('sen tam bir [MASK]')
# output:
# [{'score': 0.26599952578544617,
#   'token': 127526,
#   'token_str': 'aptalsın',
#   'sequence': 'sen tam bir aptalsın'},
#  {'score': 0.10496608912944794,
#   'token': 69461,
#   'token_str': 'ibne',
#   'sequence': 'sen tam bir ibne'},
#  {'score': 0.10287725925445557,
#   'token': 19111,
#   'token_str': 'pislik',
#   'sequence': 'sen tam bir pislik'},
#  {'score': 0.05552872270345688,
#   'token': 53442,
#   'token_str': 'kaltak',
#   'sequence': 'sen tam bir kaltak'}]
```

As you can see it fills the [MASK] token by pretty offensive content, to compare let's use it with casual NLP model, and by the way NoOffense supports hugginface model hub too!

```python
from nooffense import MaskPredictor
model = MaskPredictor("dbmdz/bert-base-turkish-128k-uncased")
model.mask_filler('sen tam bir [MASK]')
# output:
# [{'score': 0.1943783015012741,
#   'token': 18,
#   'token_str': '.',
#   'sequence': 'sen tam bir.'},
#  {'score': 0.09647665172815323,
#   'token': 5,
#   'token_str': '!',
#   'sequence': 'sen tam bir!'},
#  {'score': 0.030896930024027824,
#   'token': 2289,
#   'token_str': 'sen',
#   'sequence': 'sen tam bir sen'},
#  {'score': 0.024956168606877327,
#   'token': 3191,
#   'token_str': 'sey',
#   'sequence': 'sen tam bir sey'}]
```
You can notice it produces much less offensive content.

# Get Sentence Embeddings

Sometimes you need to encode your sentences to vector form to use them in different tasks than classification. With NoOffense you can get sentence embeddings for offensive language inputs and use them for downstream tasks like sentence similarity, retrieve&rank etc.

```python
from nooffense import SentenceEncoder
model = SentenceEncoder("path_to_your_model_weight_folder")
model.encode(['input_text_here'])

# The code above will resuls mxn matrix where m is number of sentences given and n is dimension of encoder model.
```