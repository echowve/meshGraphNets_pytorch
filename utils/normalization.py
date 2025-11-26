import torch
import torch.nn as nn

class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8, name='Normalizer', device='cuda'):
        super(Normalizer, self).__init__()
        self.name = name
        self._max_accumulations = max_accumulations

        self._std_epsilon = std_epsilon
        self.register_buffer('_acc_count', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('_num_accumulations', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('_acc_sum', torch.zeros((1, size), dtype=torch.float32))
        self.register_buffer('_acc_sum_squared', torch.zeros((1, size), dtype=torch.float32))
        self.to(device)

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate:
            # stop accumulating after a million updates, to prevent accuracy issues
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, dim=0, keepdim=True)
        squared_data_sum = torch.sum(batched_data ** 2, dim=0, keepdim=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        device = self._acc_count.device # type: ignore
        one_constant = torch.tensor(1.0, dtype=torch.float32, device=device)
        safe_count = torch.maximum(self._acc_count, one_constant)
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        device = self._acc_count.device # type: ignore
        one_constant = torch.tensor(1.0, dtype=torch.float32, device=device)
        safe_count = torch.maximum(self._acc_count, one_constant)
        mean = self._mean()
        variance = self._acc_sum_squared / safe_count - mean ** 2
        std = torch.sqrt(torch.clamp(variance, min=0.0))  # 防止数值误差导致负方差
        std_epsilon = torch.tensor(self._std_epsilon, dtype=torch.float32, device=device)
        return torch.maximum(std, std_epsilon)