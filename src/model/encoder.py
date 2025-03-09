import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=1, dropout=0.5):
        """
        Initializes the Encoder.

        Args:
            input_dim (int): Size of the source vocabulary.
            embedding_dim (int): Dimension of the word embeddings.
            hidden_dim (int): Dimension of the LSTM hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate (applied if num_layers > 1).
        """
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

    def forward(self, src):
        """
        Performs a forward pass of the encoder.

        Args:
            src (torch.Tensor): Tensor of shape [batch_size, src_len] containing source word indices.

        Returns:
            hidden (torch.Tensor): Final hidden state of the LSTM, shape [num_layers, batch_size, hidden_dim].
            cell (torch.Tensor): Final cell state of the LSTM, shape [num_layers, batch_size, hidden_dim].
        """
        embedded = self.embedding(src)  # [batch_size, src_len, embedding_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    batch_size = 16
    src_len = 10
    input_dim = 5000  # Example vocabulary size
    embedding_dim = 256
    hidden_dim = 1024
    num_layers = 1

    # Instantiate the encoder and move it to the appropriate device
    encoder = Encoder(input_dim, embedding_dim, hidden_dim, num_layers).to(device)

    # Create a dummy input tensor with random indices and move it to the device
    src = torch.randint(0, input_dim, (batch_size, src_len)).to(device)

    hidden, cell = encoder(src)

    print("Hidden state shape:", hidden.shape)  # Expected: [num_layers, batch_size, hidden_dim]
    print("Cell state shape:", cell.shape)  # Expected: [num_layers, batch_size, hidden_dim]
