import torch.nn.init as init

import torch.nn as nn
import torch
from torch_geometric.data import Data

from .model import EncoderProcesserDecoder
from utils import normalization

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

class Simulator(nn.Module):
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        device: str,
    ) -> None:
        super(Simulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size

        self.model = EncoderProcesserDecoder(
            message_passing_num=message_passing_num,
            node_input_size=node_input_size,
            edge_input_size=edge_input_size
        ).to(device)

        self._output_normalizer = normalization.Normalizer(
            size=2, name='output_normalizer', device=device
        )
        self._node_normalizer = normalization.Normalizer(
            size=node_input_size, name='node_normalizer', device=device
        )
        self.edge_normalizer = normalization.Normalizer(
            size=edge_input_size, name='edge_normalizer', device=device
        )

        self.model.apply(init_weights)
        print('Simulator model initialized')

    def update_node_attr(self, frames: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """
        Construct and normalize node features from velocity and node type.
        
        Args:
            frames: [N, 2] — current velocity (or noisy velocity during training)
            types: [N, 1] — node type indices
        
        Returns:
            Normalized node attributes [N, node_input_size]
        """
        node_type = types.squeeze(-1).long()  # [N]
        one_hot = torch.nn.functional.one_hot(node_type, num_classes=9)  # [N, 9]
        node_feats = torch.cat([frames, one_hot], dim=-1)  # [N, 2 + 9 = 11]
        normalized_feats = self._node_normalizer(node_feats, self.training)
        return normalized_feats

    @staticmethod
    def velocity_to_acceleration(noised_frames: torch.Tensor, next_velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute acceleration as the difference between next velocity and current (noised) velocity.
        """
        return next_velocity - noised_frames

    def forward(self, graph: Data, velocity_sequence_noise: torch.Tensor):
        """
        Forward pass of the simulator.

        During training:
            - Inject noise into velocity
            - Predict normalized acceleration
            - Return prediction and normalized target acceleration

        During inference:
            - Use clean velocity
            - Denormalize predicted acceleration to get velocity update
            - Return predicted next-step velocity
        """
        node_type = graph.x[:, 0:1]      # [N, 1]
        frames = graph.x[:, 1:3]         # [N, 2] — current velocity

        if self.training:
            assert velocity_sequence_noise is not None, "Noise must be provided during training"
            noised_frames = frames + velocity_sequence_noise  # [N, 2]
            node_attr = self.update_node_attr(noised_frames, node_type)
            graph.x = node_attr

            edge_attr = graph.edge_attr  # [E, 3]
            edge_attr = self.edge_normalizer(edge_attr, self.training)
            graph.edge_attr = edge_attr

            predicted_acc_norm = self.model(graph)  # [N, 2]

            target_vel = graph.y  # [N, 2]
            target_acc = self.velocity_to_acceleration(noised_frames, target_vel) # type: ignore
            target_acc_norm = self._output_normalizer(target_acc, self.training)

            return predicted_acc_norm, target_acc_norm

        else:
            # Inference mode
            node_attr = self.update_node_attr(frames, node_type)
            graph.x = node_attr
            
            edge_attr = graph.edge_attr  # [E, 3]
            edge_attr = self.edge_normalizer(edge_attr, self.training)
            graph.edge_attr = edge_attr
            
            predicted_acc_norm = self.model(graph)  # [N, 2]
            acc_update = self._output_normalizer.inverse(predicted_acc_norm)  # [N, 2]
            predicted_velocity = frames + acc_update
            return predicted_velocity