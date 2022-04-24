import torch
from utils.utils import NodeType

def get_velocity_noise(graph, noise_std, device):
    velocity_sequence = graph.x[:, 1:3]
    type = graph.x[:, 0]
    noise = torch.normal(std=noise_std, mean=0.0, size=velocity_sequence.shape).to(device)
    mask = type!=NodeType.NORMAL
    noise[mask]=0
    return noise.to(device)