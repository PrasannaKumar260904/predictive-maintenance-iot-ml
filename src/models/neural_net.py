"""PyTorch Neural Network Architectures for Predictive Maintenance.

Includes PyTorch Multi-Layer Perceptron (MLP) and LSTM Recurrent Neural Network
for RUL regression and failure risk classification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PyTorchMLP(nn.Module):
    """Multi-Layer Perceptron for RUL Regression."""

    def __init__(self, input_dim: int, hidden_units: list = [128, 64, 32], dropout_rate: float = 0.2):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden in hidden_units:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden

        layers.append(nn.Linear(in_dim, 1))  # Regression output
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PyTorchLSTM(nn.Module):
    """LSTM Recurrent Neural Network for sequential sensor telemetry."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # Extract last time step output
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out).squeeze(-1)
        return out


class PyTorchNeuralNetRegressor:
    """Scikit-Learn style wrapper for PyTorch Neural Networks."""

    def __init__(
        self,
        model_type: str = "mlp",
        input_dim: int = 20,
        hidden_units: list = [128, 64, 32],
        lr: float = 0.001,
        batch_size: int = 64,
        epochs: int = 25,
        device: str | None = None,
    ):
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "lstm":
            self.model = PyTorchLSTM(input_dim=input_dim).to(self.device)
        else:
            self.model = PyTorchMLP(input_dim=input_dim, hidden_units=hidden_units).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchNeuralNetRegressor":
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        drop_last = len(dataset) > self.batch_size
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor)
        return preds.cpu().numpy()
