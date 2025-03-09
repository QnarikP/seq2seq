import torch
from preprocess import load_vocab


# Function to translate input text
def translate(model, input_text, src_vocab, trg_vocab, device, max_len=50):
    model.eval()  # Set the model to evaluation mode
    tokens = input_text.split()  # Tokenize the input sentence

    # Convert the tokens to indices for the source vocabulary
    src_indices = [src_vocab.get(token, src_vocab["<unk>"]) for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # Add batch dimension

    # Indices for <sos> and <eos>
    sos_idx = trg_vocab["<sos>"]
    eos_idx = trg_vocab["<eos>"]

    # Use the model's translate method
    translated_indices = model.translate(src_tensor, sos_idx, eos_idx, max_len)

    # Convert indices back to tokens using the target vocabulary
    translated_sentence = [list(trg_vocab.keys())[list(trg_vocab.values()).index(idx)] for idx in translated_indices[0]]

    # Join the translated tokens to form the sentence
    return ' '.join(translated_sentence)


# Main function to run translation
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load vocabularies
    src_vocab = load_vocab("../data/src_vocab.json")  # Adjust paths
    trg_vocab = load_vocab("../data/tgt_vocab.json")

    # Hyperparameters (adjust these based on your model's architecture)
    embedding_dim = 256
    hidden_dim = 1024
    num_layers = 1
    teacher_forcing_ratio = 0

    # Load the encoder and decoder models
    from model.encoder import Encoder
    from model.decoder import Decoder
    from model.seq2seq import Seq2Seq


    encoder = Encoder(len(src_vocab), embedding_dim, hidden_dim, num_layers).to(device)
    decoder = Decoder(len(trg_vocab), embedding_dim, hidden_dim, num_layers).to(device)

    # Create the Seq2Seq model
    model = Seq2Seq(encoder, decoder, device, teacher_forcing_ratio).to(device)

    # Load model weights
    checkpoint_path = "checkpoints/best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # Set to evaluation mode

    # Example input text
    input_text = "That's great!"
    translated_text = translate(model, input_text, src_vocab, trg_vocab, device)
    print(f"Translated: {translated_text}")


if __name__ == "__main__":
    main()
