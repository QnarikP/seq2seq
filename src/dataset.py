import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    def __init__(self, sources, targets, src_word2idx, tgt_word2idx):
        """
        Args:
            sources (list of list of str): Tokenized English sentences.
            targets (list of list of str): Tokenized Russian sentences (with <sos> and <eos>).
            src_word2idx (dict): Vocabulary mapping for source language.
            tgt_word2idx (dict): Vocabulary mapping for target language.
        """
        self.sources = sources
        self.targets = targets
        self.src_word2idx = src_word2idx
        self.tgt_word2idx = tgt_word2idx

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        # Convert tokens to indices, using <unk> if a word is not found.
        src_sentence = self.sources[idx]
        tgt_sentence = self.targets[idx]
        src_unk = self.src_word2idx.get("<unk>")
        tgt_unk = self.tgt_word2idx.get("<unk>")
        src_indices = [self.src_word2idx.get(token, src_unk) for token in src_sentence]
        tgt_indices = [self.tgt_word2idx.get(token, tgt_unk) for token in tgt_sentence]

        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
        return src_tensor, tgt_tensor


def collate_fn(batch, pad_token=0):
    """
    Pads source and target sequences in a batch to the same length.

    Args:
        batch: List of tuples (src_tensor, tgt_tensor).
        pad_token (int): Index of the padding token.

    Returns:
        src_batch: Padded source tensor batch.
        tgt_batch: Padded target tensor batch.
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_token)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_token)
    return src_batch, tgt_batch


def get_data_loaders(dataset, batch_size, train_split=0.8, pad_token=0):
    """
    Splits the dataset into training and validation sets and returns their DataLoaders.

    Args:
        dataset (TranslationDataset): The full translation dataset.
        batch_size (int): Batch size.
        train_split (float): Fraction of data to be used for training.
        pad_token (int): Padding token index.

    Returns:
        train_loader, val_loader (DataLoader): DataLoader objects for training and validation.
    """
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_token)
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage (requires preprocess.py to be implemented)
    from preprocess import load_data, build_vocab

    file_path = "../data/rus.txt"
    sources, targets = load_data(file_path)

    # Build vocabularies for source and target
    src_word2idx, src_idx2word = build_vocab(sources)
    tgt_word2idx, tgt_idx2word = build_vocab(targets)

    # Create the dataset
    dataset = TranslationDataset(sources, targets, src_word2idx, tgt_word2idx)
    train_loader, val_loader = get_data_loaders(dataset, batch_size=64, pad_token=src_word2idx["<pad>"])

    # Test the dataloader by fetching one batch
    for src_batch, tgt_batch in train_loader:
        print("Source batch shape:", src_batch.shape)
        print("Target batch shape:", tgt_batch.shape)
        break