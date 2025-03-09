import re
import string
from collections import Counter
import json


def clean_text(text):
    """
    Cleans the input text by:
      - Converting to lowercase
      - Removing numbers
      - Removing punctuation
      - Stripping extra whitespace
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def tokenize(text):
    """
    Splits the input text into tokens based on whitespace.
    """
    return text.split()


def load_data(file_path, source_index=0, target_index=1):
    """
    Loads and preprocesses data from a tab-separated file.
    Only the first two columns (source and target sentences) are used.

    Args:
        file_path (str): Path to the dataset file.
        source_index (int): Column index for the source sentence.
        target_index (int): Column index for the target sentence.

    Returns:
        sources (list of list of str): Tokenized source sentences.
        targets (list of list of str): Tokenized target sentences with special tokens added.
    """
    sources = []
    targets = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue  # Skip lines that don't have enough columns
            src_sentence = clean_text(parts[source_index])
            tgt_sentence = clean_text(parts[target_index])
            # Add special tokens for the target sentence
            tgt_sentence = "<sos> " + tgt_sentence + " <eos>"
            sources.append(tokenize(src_sentence))
            targets.append(tokenize(tgt_sentence))
    return sources, targets


def build_vocab(sentences, min_freq=1):
    """
    Builds a vocabulary mapping from words to indices for a list of tokenized sentences.

    Args:
        sentences (list of list of str): List of tokenized sentences.
        min_freq (int): Minimum frequency for a word to be included.

    Returns:
        word2idx (dict): Mapping from words to indices.
        idx2word (dict): Mapping from indices to words.
    """
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence)

    # Only include words that appear at least min_freq times
    vocab = {word for word, freq in counter.items() if freq >= min_freq}

    # Initialize vocabulary with special tokens for padding and unknown words.
    word2idx = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    for word in sorted(vocab):
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def save_vocab(vocab, file_path):
    """
    Saves a vocabulary dictionary to a JSON file.

    Args:
        vocab (dict): Word-to-index dictionary.
        file_path (str): Path to save the vocabulary file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)


def load_vocab(file_path):
    """
    Loads a vocabulary dictionary from a JSON file.

    Args:
        file_path (str): Path to the vocabulary file.

    Returns:
        vocab (dict): Loaded vocabulary mapping from words to indices.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab


if __name__ == "__main__":
    # Example usage for testing the preprocessing functions.
    file_path = "../data/rus.txt"
    sources, targets = load_data(file_path)
    print("First source sentence tokens:", sources[0])
    print("First target sentence tokens:", targets[0])

    src_word2idx, src_idx2word = build_vocab(sources)
    tgt_word2idx, tgt_idx2word = build_vocab(targets)

    print("Source vocabulary size:", len(src_word2idx))
    print("Target vocabulary size:", len(tgt_word2idx))

    # Save vocabularies
    save_vocab(src_word2idx, "../data/src_vocab.json")
    save_vocab(tgt_word2idx, "../data/tgt_vocab.json")

    # Load vocabularies
    loaded_src_vocab = load_vocab("../data/src_vocab.json")
    loaded_tgt_vocab = load_vocab("../data/tgt_vocab.json")

    print("Loaded Source vocabulary size:", len(loaded_src_vocab))
    print("Loaded Target vocabulary size:", len(loaded_tgt_vocab))
