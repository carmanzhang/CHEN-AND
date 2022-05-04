# model = pickle.load(open(os.path.join(cached_dir, 'gpu-and-model.pkl'), 'rb'))
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.dropout = 0.1
        self.input_dim = input_dim
        self.linear = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(input_dim * 2, input_dim * 4),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(input_dim * 4, input_dim // 2),
            nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            nn.Linear(input_dim // 2, 1),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input):
        # Expect input to be shape (N, self.input_dim)
        output = self.linear(input)  # Shape (N, 1)
        output = output.flatten()  # Shape (N)
        output = torch.sigmoid(output)
        return output
