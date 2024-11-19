# Transformer-based Translation Model

This project implements a Transformer-based model for machine translation using PyTorch.

## Prerequisites

- Python 3.7+
- PyTorch 1.7+
- tokenizers
- tqdm
- nltk

You can install the required packages using:

```
pip install torch tokenizers tqdm nltk
```

## Project Structure

- `encoder.py`: Contains the implementation of the Transformer encoder.
- `decoder.py`: Contains the implementation of the Transformer decoder.
- `utils.py`: Includes utility functions and classes such as MultiHeadAttention, PositionWiseFeedForward, and the main Transformer model.
- `train.py`: Script for training the Transformer model.
- `test.py`: Script for evaluating the trained model and generating translations.

## Data Preparation

Ensure you have your parallel corpus files in the `ted-talks-corpus` directory:
- `train.en` and `train.fr` for training data
- `dev.en` and `dev.fr` for validation data
- `test.en` and `test.fr` for test data

## Training the Model

To train the model, run:

```
python train.py
```

This script will:
1. Train a tokenizer on the training data.
2. Create and train the Transformer model.
3. Save the best model based on validation loss.

You can modify hyperparameters in the `train.py` file.

## Testing the Model

To evaluate the model and generate translations, run:

```
python test.py
```

This script will:
1. Load the trained model.
2. Generate translations for the test set.
3. Calculate BLEU scores.

## Customization

You can customize various aspects of the model:
- Modify the model architecture in `utils.py`
- Adjust hyperparameters in `train.py`
- Change evaluation metrics or generation process in `test.py`

## Note

Training a Transformer model can be computationally intensive. Ensure you have adequate computational resources, preferably a GPU, for efficient training.

## Troubleshooting

If you encounter any issues:
1. Ensure all required packages are installed.
2. Check that your data files are in the correct location and format.
3. Verify that you have sufficient computational resources for training.

For any additional questions or issues, please open an issue in the project repository.

Link to pretrained model : https://drive.google.com/file/d/1GNTafIEyjZIKWfK9ryV0ruF7ZzkzrMHf/view?usp=drive_link