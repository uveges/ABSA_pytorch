# ABSA-PyTorch

This project implements aspect-based sentiment analysis using BERT models in a PyTorch environment. The `config.py` file contains key configurations designed for training and predicting with models in English and Hungarian.

## Prerequisites

Before running the project, install the required dependencies listed in `requirements_rtx30.txt`:
```bash
pip install -r requirements_rtx30.txt
```

## Currently Used Python Files

### Prediction
- Main script: `./examples/examples_predict.py`
- Dependencies:
    - **Data Preparation**: NER-based data preprocessing is handled by `./preprocessors/prepeare_data_for_prediction.py` (`DataPreparator` class), which transforms raw xlsx data into prediction-ready format.
    - **Prediction**: Predictions are made using `./src/prediction.py` (`Predictor` class).
    - **Configurations**: All settings are specified in `./config.py`.

### Training
- Main script: `./examples/examples_train.py`
- Dependencies:
    - **Training**: The training process is controlled by `./src/training.py` (`Trainer` class).
    - **Configurations**: All settings are specified in `./config.py`.

## Config.py Parameters

### Train - Test Set Creation Parameters
- **dataset_name**: Name of the dataset, in this case: `Validated`.
- **test_size**: Proportion of the dataset to use as the test set, e.g., `0.2` (20%).
- **text_column**: Column name for text data.
- **NE_column**: Column name for Named Entity (NER) labeling.
- **NE_type_column**: Column for the type of Named Entity.
- **predictions_column**: Column for storing prediction results.

### Model-Specific Parameters
- **checkpoint**: Path to the BERT model checkpoint containing the latest training state.
- **train_dataset** and **test_dataset**: Paths to the English and Hungarian training and test datasets.
- **bert_model**: The BERT model to use. For English: `bert-base-cased`, and for Hungarian: `SZTAKI-HLT/hubert-base-cc`.
- **spacy_model_name**: SpaCy model name for NER, e.g., `en_core_web_lg` for English or `hu_core_news_lg` for Hungarian.

### Model Parameters
- **dropout**: Dropout rate (0.01).
- **bert_dim**: Hidden layer dimension of the BERT model (768).
- **polarities_dim**: Number of sentiment polarities (3).
- **max_seq_len**: Maximum input sequence length for BERT (85).
- **bert_model_name**: The name of the BERT model used.
- **optimizer**: Optimization algorithm, e.g., `adam`.
- **initializer**: Weight initialization method, e.g., `xavier_uniform_`.
- **lr**: Learning rate, set to `2e-5`.
- **l2reg**: L2 regularization factor (0.01).
- **num_epoch**: Number of epochs during training (20).
- **batch_size**: Batch size (16).
- **log_step**: Step interval for logging (10).
- **embed_dim** and **hidden_dim**: Dimensions of the embedding and hidden layers (300).
- **hops**: Steps for the attention mechanism (3).
- **patience**: Number of epochs to wait for improvement before stopping (5).
- **device**: Device for computation (CPU or GPU).
- **seed**: Seed for randomness (1234).
- **valset_ratio**: Size of the validation set (0, so no separate validation set is used within the test set).

## Licence

MIT
