import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import random
import numpy as np

from dataset import TranslationDataset, collate_fn
from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from preprocess import load_vocab, load_data
from utils import load_config  # Load config from utils.py

print("Loading configuration from YAML...")
# Load config from YAML file
config = load_config("../config/config.yaml")
print("Configuration loaded.")

print("Setting random seeds for reproducibility...")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
print("Random seeds set.")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading vocabularies...")
# Load vocabularies
src_vocab = load_vocab(config["data"]["source_vocab"])
trg_vocab = load_vocab(config["data"]["target_vocab"])
src_vocab_size = len(src_vocab)
trg_vocab_size = len(trg_vocab)
print(f"Source vocabulary size: {src_vocab_size}")
print(f"Target vocabulary size: {trg_vocab_size}")

# Special tokens
PAD_IDX = src_vocab["<pad>"]
SOS_IDX = trg_vocab["<sos>"]
EOS_IDX = trg_vocab["<eos>"]

print("Loading and preprocessing dataset...")
sources, targets = load_data(config["data"]["file_path"])
print(f"Dataset loaded with {len(sources)} samples.")
# Take half of the dataset
half_size = len(sources) // 2
sources, targets = sources[:half_size], targets[:half_size]

print("Creating TranslationDataset objects...")
# Create dataset
train_dataset = TranslationDataset(sources, targets, src_vocab, trg_vocab)
valid_dataset = TranslationDataset(sources, targets, src_vocab, trg_vocab)
print("Datasets created.")

print("Creating DataLoaders...")
train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, PAD_IDX)
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    collate_fn=lambda batch: collate_fn(batch, PAD_IDX)
)
print("DataLoaders created.")

print("Initializing model components...")
# Initialize model
encoder = Encoder(
    src_vocab_size, config["model"]["embedding_dim"], config["model"]["hidden_dim"], config["model"]["num_layers"]
).to(device)
decoder = Decoder(
    trg_vocab_size, config["model"]["embedding_dim"], config["model"]["hidden_dim"], config["model"]["num_layers"]
).to(device)
model = Seq2Seq(encoder, decoder, device, config["training"]["teacher_forcing_ratio"]).to(device)
print("Model initialized.")

print("Setting up loss function and optimizer...")
# Loss function & optimizer
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
# Gradient clipping threshold
CLIP = config["training"]["clip"]
print("Loss function and optimizer set.")


# Training function
def train_epoch(model, iterator, optimizer, criterion, clip):
    """
    Trains a single epoch for the given model on the provided dataset iterator.
    This function handles the forward pass of the data through the model, computes
    loss, performs backpropagation, and updates the model's weights. It utilizes
    gradient clipping to prevent exploding gradients during training and calculates
    the average loss over all batches in the epoch.

    :param model: The neural network model to be trained.
    :type model: torch.nn.Module
    :param iterator: An iterator providing batches of source and target sequences. Each batch
        is expected to contain source and target data ready for training.
    :type iterator: DataLoader or Iterator
    :param optimizer: The optimizer used for updating model parameters.
    :type optimizer: torch.optim.Optimizer
    :param criterion: The loss function used for computing the difference between the
        predicted and target output sequences.
    :type criterion: torch.nn.Module
    :param clip: The maximum gradient norm for gradient clipping.
    :type clip: float
    :return: The average loss over all batches for the epoch.
    :rtype: float
    """
    model.train()
    epoch_loss = 0
    total_batches = len(iterator)
    print("Starting training epoch...")

    for i, (src, trg) in enumerate(iterator, 1):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, trg_vocab_size)  # Output: [batch_size, trg_len, trg_vocab_size]

        # Reshape output and target for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # Remove <sos> token
        trg = trg[:, 1:].reshape(-1)  # Remove <sos> token

        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        if i % config["logging"]["print_every"] == 0 or i == total_batches:
            print(f"Batch {i}/{total_batches} - Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / total_batches
    print("Training epoch completed.")
    return avg_loss


# Validation function
def evaluate(model, iterator, criterion):
    """
    Evaluates the performance of the given model over the provided data iterator. The function
    executes the evaluation process in no_grad mode to prevent gradient calculation. It computes
    the evaluation loss for each batch in the iterator using the designated criterion, then returns
    the average loss across all batches. Additionally, intermittent progress messages are printed
    to indicate the current evaluation stage.

    :param model: The model to evaluate.
    :type model: torch.nn.Module
    :param iterator: The data iterator for evaluation. Must yield (src, trg) tuples where src is the
        source sequence tensor and trg is the target sequence tensor.
    :type iterator: torch.utils.data.DataLoader
    :param criterion: The loss function used for loss computation during evaluation.
    :type criterion: torch.nn.Module
    :return: The average evaluation loss computed over all batches in the iterator.
    :rtype: float
    """
    model.eval()
    epoch_loss = 0
    total_batches = len(iterator)
    print("Starting evaluation...")

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator, 1):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, trg_vocab_size)  # No teacher forcing during evaluation

            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

            if i % config["logging"]["print_every"] == 0 or i == total_batches:
                print(f"Validation Batch {i}/{total_batches} - Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / total_batches
    print("Evaluation completed.")
    return avg_loss


# Training loop
def train_model(model, train_loader, valid_loader, optimizer, criterion, clip, epochs, checkpoint_path):
    """
    Trains a given model using specified training and validation data loaders, optimizer,
    loss function, gradient clipping, and saves the best performing model based on
    validation loss.

    This function runs for an indicated number of epochs and monitors training and validation
    losses for each epoch. If the validation loss improves, the model checkpoint is saved
    to the specified filepath. If the directory for the checkpoint does not exist, it is
    automatically created.

    :param model: Model to be trained.
    :type model: torch.nn.Module
    :param train_loader: DataLoader for the training dataset.
    :type train_loader: torch.utils.data.DataLoader
    :param valid_loader: DataLoader for the validation dataset.
    :type valid_loader: torch.utils.data.DataLoader
    :param optimizer: Optimizer for updating model weights.
    :type optimizer: torch.optim.Optimizer
    :param criterion: Loss function used during training and evaluation.
    :type criterion: torch.nn.modules.loss._Loss
    :param clip: Gradient clipping value to prevent exploding gradients.
    :type clip: float
    :param epochs: Number of training epochs.
    :type epochs: int
    :param checkpoint_path: Path to save the model checkpoint for the best validation loss.
    :type checkpoint_path: str
    :return: None
    :rtype: NoneType
    """
    best_valid_loss = float("inf")
    print("Starting training loop...\n")

    for epoch in range(epochs):
        print(f"=== Epoch {epoch + 1}/{epochs} ===")
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_loader, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(int(end_time - start_time), 60)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}")

        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            # Ensure the directory for checkpoint exists
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if not os.path.exists(checkpoint_dir):
                print(f"Creating directory: {checkpoint_dir}")
                os.makedirs(checkpoint_dir)

            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at epoch {epoch + 1}\n")
        else:
            print("No improvement, model not saved.\n")


# Run training
if __name__ == "__main__":
    print("Beginning training process...")
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        clip=CLIP,
        epochs=config["training"]["epochs"],
        checkpoint_path=config["logging"]["checkpoint_path"]
    )
    print("Training process finished.")
