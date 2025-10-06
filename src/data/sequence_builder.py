# src/utils/sequence_builder.py
import numpy as np


def create_sequences(data: np.ndarray, window_size: int):
    """
    Generate sequential input-output pairs for time series modeling.

    This function transforms continuous time-series data into overlapping
    fixed-length sequences (`window_size`) suitable for recurrent models
    such as LSTMs or GRUs.

    Parameters
    ----------
    data : np.ndarray of shape (N, n_features)
        The full dataset containing both input and output features.
        Example structure: [[PWM, Tank_Level], ...]
    window_size : int
        Number of time steps in each sequence (e.g., 30).

    Returns
    -------
    X : np.ndarray of shape (num_seq, window_size, n_features)
        Array containing the input sequences.
    y : np.ndarray of shape (num_seq,)
        Array containing the target outputs corresponding to each sequence.

    Raises
    ------
    ValueError
        If `window_size` is greater than or equal to the number of samples in `data`.

    Notes
    -----
    - The function assumes the target variable is located in column index `1` of `data`.
      Modify this index if your target variable is stored elsewhere.
    - Commonly used for supervised learning on sequential data (e.g., LSTM, GRU, or TCN).

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : For feature scaling before building sequences.
    torch.utils.data.DataLoader : For batching and shuffling sequence data.

    Examples
    --------
    >>> import numpy as np
    >>> from sequence_builder import create_sequences
    >>> time = np.arange(0, 10, 0.1)
    >>> data_input = np.sin(time)
    >>> data_output = np.cos(time)
    >>> data = np.column_stack((data_input, data_output))
    >>> X, y = create_sequences(data, window_size=5)
    >>> X.shape, y.shape
    ((95, 5, 2), (95,))
    >>> X[0]
    array([[ 0.0000,  1.0000],
           [ 0.0998,  0.9950],
           [ 0.1987,  0.9801],
           [ 0.2955,  0.9553],
           [ 0.3894,  0.9211]])
    >>> y[0]
    0.8776
    """

    # --- Validation ---
    if window_size >= len(data):
        raise ValueError(
            f"window_size ({window_size}) must be smaller than data length ({len(data)})."
        )

    # --- Sequence generation ---
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size, 1])  # Column index 1 → output variable

    return np.array(X), np.array(y)


# =====================================================
# 🔽 Demonstration (Standalone Execution)
# =====================================================
if __name__ == "__main__":
    print("🧩 Testing create_sequences() ...")

    # Generate synthetic [input, output] data
    time = np.arange(0, 10, 0.1)
    data_input = np.sin(time)
    data_output = np.cos(time)
    data = np.column_stack((data_input, data_output))

    # Build sequences
    X, y = create_sequences(data, window_size=5)

    # Display results
    print(f"✅ X shape: {X.shape}  | y shape: {y.shape}")
    print("🔍 Example X[0]:\n", X[0])
    print("🔍 Example y[0]:", y[0])
