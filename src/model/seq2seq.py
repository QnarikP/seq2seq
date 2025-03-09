import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        """
        Initializes the Seq2Seq model.

        Args:
            encoder (Encoder): The encoder module.
            decoder (Decoder): The decoder module.
            device (torch.device): The device to run the model on (CPU/GPU).
            teacher_forcing_ratio (float): Probability of using teacher forcing during training.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, trg, trg_vocab_size, max_len=50):
        """
        Performs a forward pass of the Seq2Seq model.

        Args:
            src (torch.Tensor): Input sequence (source language), shape [batch_size, src_len].
            trg (torch.Tensor): Target sequence (target language), shape [batch_size, trg_len].
            trg_vocab_size (int): Vocabulary size of the target language.
            max_len (int): Maximum length for output sequence.

        Returns:
            outputs (torch.Tensor): Predicted token scores, shape [batch_size, trg_len, trg_vocab_size].
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode input sequence
        hidden, cell = self.encoder(src)

        # First token to be passed to decoder is the <sos> token
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = output  # Store the output predictions

            # Determine whether to use teacher forcing
            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            top1 = output.argmax(1)  # Get the predicted token

            # Use ground truth token if teacher forcing, otherwise use model's prediction
            input_token = trg[:, t] if teacher_force else top1

        return outputs

    def translate(self, src, sos_idx, eos_idx, max_len=50):
        """
        Translates a given input sequence using the trained Seq2Seq model.

        Args:
            src (torch.Tensor): Input sequence, shape [batch_size, src_len].
            sos_idx (int): Index of the <sos> token.
            eos_idx (int): Index of the <eos> token.
            max_len (int): Maximum output sequence length.

        Returns:
            translations (torch.Tensor): Predicted sequence indices, shape [batch_size, max_len].
        """
        batch_size = src.shape[0]
        translations = torch.zeros(batch_size, max_len).long().to(self.device)

        # Encode input sequence
        hidden, cell = self.encoder(src)

        # First token to be passed to decoder is the <sos> token
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long).to(self.device)

        for t in range(max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            top1 = output.argmax(1)  # Get predicted token

            translations[:, t] = top1  # Store the prediction

            if (top1 == eos_idx).all():  # Stop if all sentences reach <eos>
                break

            input_token = top1  # Next input is the predicted token

        return translations


if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Example hyperparameters
    src_vocab_size = 5000
    trg_vocab_size = 6000
    embedding_dim = 256
    hidden_dim = 1024
    num_layers = 1
    teacher_forcing_ratio = 0.5

    # Import Encoder and Decoder
    from encoder import Encoder
    from decoder import Decoder

    # Instantiate the encoder and decoder
    encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    decoder = Decoder(trg_vocab_size, embedding_dim, hidden_dim, num_layers).to(device)

    # Create the Seq2Seq model
    model = Seq2Seq(encoder, decoder, device, teacher_forcing_ratio).to(device)

    # Create dummy input tensors
    batch_size = 16
    src_len = 10
    trg_len = 12
    src = torch.randint(0, src_vocab_size, (batch_size, src_len)).to(device)
    trg = torch.randint(0, trg_vocab_size, (batch_size, trg_len)).to(device)

    # Forward pass
    output = model(src, trg, trg_vocab_size)
    print("Output shape:", output.shape)  # Expected: [batch_size, trg_len, trg_vocab_size]