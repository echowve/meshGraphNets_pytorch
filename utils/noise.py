import torch
from utils.utils import NodeType

def get_velocity_noise(graph, noise_std=2e-2, device='cuda'):
    # velocity is columns 1:4 in graph.x
    velocity = graph.x[:, 1:4]               # [N,3]
    noise = torch.randn_like(velocity) * noise_std

    type = graph.x[:, 0]
    # noise = torch.normal(std=noise_std, mean=0.0, size=velocity_sequence.shape).to(device)
    mask = type!=NodeType.NORMAL
    noise[mask]=0
    return noise.to(device)