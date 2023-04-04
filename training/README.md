# Training Notebooks

Welcome to the training notebooks section! Here you can find a step-by-step guide on how to train your own model using our multi-stage training scripts. 

## Overview of Training Stages
Our training process consists of several stages that utilize transfer learning and pseudo-labeling to improve the model's performance. Here's a brief overview of the stages:

1. **Whole Word Masked Language Model Training**: This stage involves training a language model on an unlabeled corpus using the whole-word masking technique. This pre-training step helps the model to learn the language patterns and improve its understanding of the text.

2. **5-Fold Classification Training**: In this stage, we use the labeled data to train a 5-fold classification model while also using the pseudo-labeled data generated from the pre-trained language model. The pseudo-labeled data is labeled based on the predictions made by the pre-trained model on the unlabeled data.

3. **Single Classification Model Training**: We train a single classification model using the pseudo-labeled data from the previous step and transfer the weights to the labeled data training.

4. **Repeat Pseudo-Labeling**: Using the steps above, we predict the unlabeled data again and repeat the process of generating pseudo-labeled data multiple times to further improve the model's performance.

5. **Test Data Prediction**: We use the final weights and models to predict the test data.
