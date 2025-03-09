# English-to-Russian Machine Translation

This project implements a machine translation system using a sequence-to-sequence (Seq2Seq) model with LSTM encoder-decoder architecture in PyTorch. The model is designed to translate English sentences into Russian and is trained on a dataset of English-Russian sentence pairs.

## Overview

- **Objective:**  
  Build and train a Seq2Seq neural network to translate English sentences to Russian.

- **Key Features:**  
  - Data preprocessing (cleaning, tokenization, vocabulary creation).
  - Custom PyTorch Dataset and DataLoader with dynamic padding.
  - Encoder-Decoder architecture using LSTMs.
  - Implementation of teacher forcing during training.
  - Model checkpointing based on validation loss.
  - Configuration management using a YAML file.

## Directory Structure

```plaintext
project-root/
├── config/
│   └── config.yaml           # Hyperparameters and file paths
├── data/
│   └── rus.txt               # Raw dataset file (tab-separated sentence pairs)
├── model/
│   ├── encoder.py            # Encoder module
│   ├── decoder.py            # Decoder module
│   └── seq2seq.py            # Seq2Seq model combining encoder and decoder
├── src/
│   └── train.py              # Training script
├── dataset.py                # Custom Dataset and DataLoader utilities
├── preprocess.py             # Data preprocessing, tokenization, and vocabulary building
├── utils.py                  # Utility functions (e.g., config loader)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/QnarikP/seq2seq
   cd your-repo
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** Make sure you have PyTorch installed with CUDA support if you plan to use a GPU.

## Configuration

The project uses a YAML configuration file located at `config/config.yaml` to manage hyperparameters and file paths. An example configuration is:

```yaml
data:
  file_path: "../data/rus.txt"
  train_split: 0.8
  validation_split: 0.2
  columns:
    source: 0
    target: 1
  source_vocab: "../data/src_vocab.json"
  target_vocab: "../data/tgt_vocab.json"

model:
  embedding_dim: 256
  hidden_dim: 1024
  num_layers: 1
  dropout: 0.5
  special_tokens:
    sos: "<sos>"
    eos: "<eos>"
    pad: "<pad>"

training:
  epochs: 20
  batch_size: 16
  learning_rate: 0.001
  clip: 1.0
  teacher_forcing_ratio: 0.5
  loss_function: "CrossEntropyLoss"

logging:
  checkpoint_path: "checkpoints/best_model.pt"
  print_every: 100
```

## Data Preparation

- **Dataset Format:**  
  The raw dataset (`data/rus.txt`) is a tab-separated file where:
  - The first column contains English sentences.
  - The second column contains corresponding Russian sentences.
  - Additional columns (e.g., attribution) are ignored.

- **Preprocessing:**  
  The `preprocess.py` script cleans text by converting to lowercase, removing numbers and punctuation, tokenizing sentences, and adding special tokens (`<sos>` and `<eos>`) to the Russian sentences.

- **Vocabulary:**  
  The vocabulary is built from the cleaned and tokenized sentences. The vocabularies are saved as JSON files and loaded for training.

## Training the Model

To start training, run:

```bash
python src/train.py
```

The training script will:
- Load the configuration and preprocess the dataset.
- Create PyTorch DataLoaders with dynamic padding.
- Initialize the Encoder, Decoder, and Seq2Seq model.
- Train the model with teacher forcing.
- Evaluate on a validation set and save the best model checkpoint.

During training, batch-level and epoch-level progress is printed to the console.

## Inference & Translation

Once the model is trained, you can create a translation script (e.g., `translate.py`) that:
- Loads the trained model checkpoint.
- Accepts an English sentence as input.
- Processes the input and generates a Russian translation token by token until the `<eos>` token is generated or a maximum length is reached.