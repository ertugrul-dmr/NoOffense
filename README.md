# NoOffense
This package provides a smart and fast offensive language detection tool for Turkish, which is based on the transformers technology.

The **Overfit-GM** team developed this package as a part of the **Türkiye Açık Kaynak Platformu** NLP challenge.


## Installation Instructions
To install this package, simply run the following command in your Python environment:

```python
pip install git+https://github.com/ertugrul-dmr/NoOffense.git
```
This command will install the package directly from the GitHub repository.
# Usage

NoOffense is a versatile tool that supports multiple standard NLP tasks, with a particular focus on detecting offensive language in Turkish texts.

# Getting Predictions
You can easily obtain predictions for a list of texts by using NoOffense as follows:
```python
from noofense import Predictor

# Predictor applies some text cleaning&preps internally
# In case of multiple trained models use list of model paths,
# and NoOffense will yield ensemble results internally
# you can use gpu support by setting device for faster inference

clf = Predictor(model_path='path_to_your_model_weight_folder', device='cuda')
predictions= clf.predict(['sen ne gerizekalısın ya'], progress_bar=False)

# Code above will return dict contains these keys:
# ['probas']: probabilities of each class
# ['predictions']: label encoding for each predicted class
# ['predicted_labels']: str label mappings for each class, INSULT, Racist etc.
```
NoOffense also supports making predictions on Pandas dataframes:

It expects a column named "text" to make predictions.
```python
from noofense import DFPredictor

clf = DFPredictor(model_path='path_to_your_model_weights_folder', device='cpu')
predicted_df = clf.predict_df(df_path="path_to_your_csv_file", save_csv=False, progress_bar=True)
# Code above will return a copy of original dataframe with predictions.
```

## Fill Masks

NoOffense models are trained in a self-supervised fashion using a large Turkish offensive text corpus. These models are then fine-tuned for text classification tasks. However, as a byproduct of our Masked Language Model training, you can also use it as a fill-mask pipe!
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

As you can see, it fills the [MASK] token with pretty offensive content. To compare, let's use it with a casual NLP model. By the way, NoOffense also supports Hugging Face's model hub!

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
As you can see, the casual NLP model produces much less offensive content than NoOffense.

# Get Sentence Embeddings

Sometimes, you need to encode your sentences in vector form to use them in different tasks than classification. With NoOffense, you can get sentence embeddings for offensive language inputs and use them for downstream tasks like sentence similarity, retrieval, and ranking, etc.
```python
from nooffense import SentenceEncoder
model = SentenceEncoder("path_to_your_model_weight_folder")
model.encode(['input_text_here'])

# The code above will resuls mxn matrix where m is number of sentences given and n is dimension of encoder model.
```


Speaking of sentence similarity, NoOffense library has method that helps you find most similar sentences using cosine similarity based on SentenceEncodings:
```python
from nooffense import SentenceEncoder
model = SentenceEncoder("path_to_your_model_weight_folder")
model.find_most_similar("bir küfürlü içerik", ["text1", "text2", ..., "text_n"])
```
The code above will result sorted dictionary starting with most similar text given from list of strings.
# Team

![](https://github.com/ertugrul-dmr/NoOffense/blob/master/docs/img/team_logov1.png?raw=true)