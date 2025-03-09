import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers=1, dropout=0.5):
        """
        Initializes the Decoder.

        Args:
            output_dim (int): Size of the target vocabulary.
            embedding_dim (int): Dimension of the word embeddings.
            hidden_dim (int): Dimension of the LSTM hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate (applied if num_layers > 1).
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        """
        Performs a forward pass for a single time step of the decoder.

        Args:
            input (torch.Tensor): Tensor of shape [batch_size] representing the current token indices.
            hidden (torch.Tensor): Hidden state from previous time step; shape [num_layers, batch_size, hidden_dim].
            cell (torch.Tensor): Cell state from previous time step; shape [num_layers, batch_size, hidden_dim].

        Returns:
            prediction (torch.Tensor): Unnormalized logits for the next token, shape [batch_size, output_dim].
            hidden (torch.Tensor): Updated hidden state.
            cell (torch.Tensor): Updated cell state.
        """
        # Convert input shape from [batch_size] to [batch_size, 1] for embedding
        input = input.unsqueeze(1)
        embedded = self.embedding(input)  # Shape: [batch_size, 1, embedding_dim]

        # Pass the embedded token through the LSTM
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # Pass the LSTM output through the fully connected layer to get predictions
        # Squeeze the output to remove the time-step dimension
        prediction = self.fc_out(output.squeeze(1))  # Shape: [batch_size, output_dim]
        return prediction, hidden, cell


if __name__ == "__main__":
    # Check for GPU availability and set the device accordingly.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Example hyperparameters
    batch_size = 16
    output_dim = 6000  # Example target vocabulary size
    embedding_dim = 256
    hidden_dim = 1024
    num_layers = 1

    # Instantiate the decoder and move it to the appropriate device
    decoder = Decoder(output_dim, embedding_dim, hidden_dim, num_layers).to(device)

    # Create a dummy input token for the decoder (random indices, shape: [batch_size])
    input_token = torch.randint(0, output_dim, (batch_size,)).to(device)

    # Create dummy hidden and cell states (typically output from the encoder)
    hidden = torch.randn(num_layers, batch_size, hidden_dim).to(device)
    cell = torch.randn(num_layers, batch_size, hidden_dim).to(device)

    # Forward pass: get the prediction and updated states
    prediction, hidden, cell = decoder(input_token, hidden, cell)

    print("Prediction shape:", prediction.shape)  # Expected: [batch_size, output_dim]
