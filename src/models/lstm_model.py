import torch
import torch.nn as nn


class LSTM_MODEL(nn.Module):
    """
    A flexible LSTM model designed for both time-series forecasting and
    future extension into stateful or reinforcement learning-based control systems.

    This model supports both **stateless** and **stateful** operation modes.
    - Stateless: Each forward pass is independent (typical for batch training).
    - Stateful: Hidden and cell states are retained across time steps or episodes
      (useful in online learning, simulation, or control tasks).

    Parameters
    ----------
    input_dim : int
        Number of input features at each time step.
    hidden_dim : int, optional (default=64)
        Number of hidden units in each LSTM layer.
    num_layers : int, optional (default=2)
        Number of stacked LSTM layers.
    output_dim : int, optional (default=1)
        Number of output features (e.g., 1 for regression).
    dropout : float, optional (default=0.1)
        Dropout probability applied between LSTM layers.
    stateful : bool, optional (default=False)
        If True, retains hidden state between calls to `forward()`.

    Example
    -------
    >>> import torch
    >>> model = LSTM_MODEL(input_dim=2, hidden_dim=32, output_dim=1, stateful=True)
    >>> x = torch.randn(4, 10, 2)  # (batch_size, seq_len, input_dim)
    >>> model.reset_state(batch_size=4)
    >>> y = model(x)
    >>> print(y.shape)
    torch.Size([4, 1])
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1, stateful=False):
        super(LSTM_MODEL, self).__init__()

        # Core architecture parameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.stateful = stateful

        # Define LSTM network
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected layer maps hidden features to desired output size
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Internal hidden and cell states (used only if stateful=True)
        self.hidden_state = None

    def reset_state(self, batch_size=1, device=None):
        """
        Reset the hidden and cell states of the LSTM.

        This should be called at the beginning of each new simulation,
        episode, or sequence when using stateful mode.

        Parameters
        ----------
        batch_size : int, optional (default=1)
            Batch size for initializing hidden states.
        device : torch.device, optional
            Device where the tensors should be allocated. If None, uses model's device.
        """
        if device is None:
            device = next(self.parameters()).device

        # Initialize hidden and cell states with zeros
        self.hidden_state = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Model output of shape (batch_size, output_dim),
            corresponding to the last time step in the sequence.
        """
        if self.stateful and self.hidden_state is not None:
            # Use retained hidden state
            out, self.hidden_state = self.lstm(x, self.hidden_state)
        else:
            # Stateless forward
            out, _ = self.lstm(x)

        # Take output at the final time step and map to output dimension
        out = self.fc(out[:, -1, :])
        return out

    def detach_state(self):
        """
        Detach hidden and cell states from the computational graph.

        Use this during training to prevent gradients from
        propagating through time indefinitely (avoiding gradient explosion).
        """
        if self.hidden_state is not None:
            self.hidden_state = (
                self.hidden_state[0].detach(),
                self.hidden_state[1].detach()
            )


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # 🔬 Example: Testing LSTM_MODEL for time-series forecasting
    # ----------------------------------------------------------------------
    batch_size, seq_len, input_dim = 4, 10, 2
    model = LSTM_MODEL(input_dim=input_dim, hidden_dim=32, output_dim=1, stateful=True)

    # Generate synthetic input data
    x = torch.randn(batch_size, seq_len, input_dim)

    # Reset hidden states before sequence start
    model.reset_state(batch_size=batch_size)

    # Perform forward pass
    y = model(x)

    print("✅ Output shape:", y.shape)
    print("Sample output:", y.detach().cpu().numpy().flatten()[:5])
