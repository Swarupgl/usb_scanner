import torch
import torch.nn as nn
import torch.nn.functional as F


class MalConv(nn.Module):
    """Kolosnjaji-inspired gated CNN for raw bytes.

    Input: LongTensor of shape (batch, length) with values 0..256
           where 256 is reserved for padding.
    Output: (batch, 1) probabilities in [0, 1]
    """

    def __init__(self, input_length: int = 1048576, window_size: int = 512):
        super().__init__()
        self.input_length = input_length
        self.window_size = window_size

        # 256 byte values + 1 for padding (index 256)
        self.embed = nn.Embedding(257, 8, padding_idx=256)

        self.conv_1 = nn.Conv1d(8, 128, kernel_size=window_size, stride=window_size)
        self.conv_2 = nn.Conv1d(8, 128, kernel_size=window_size, stride=window_size)

        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()

        # Safety: map any out-of-range values to padding
        x = torch.where((x >= 0) & (x <= 256), x, torch.full_like(x, 256))

        x = self.embed(x).transpose(1, 2)  # (B, C=8, L)

        cnn_value = self.conv_1(x)
        cnn_gate = torch.sigmoid(self.conv_2(x))
        gated = cnn_value * cnn_gate

        pooled = self.pooling(gated).squeeze(-1)
        out = F.relu(self.fc_1(pooled))
        return torch.sigmoid(self.fc_2(out))
