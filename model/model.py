import torch.nn as nn
from .blocks import EdgeBlock, NodeBlock
from utils.utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = nn.Sequential(nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_size))
    if lay_norm: return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    return module


class Encoder(nn.Module):
    def __init__(self, hidden_size=128, device='cuda'):
        super(Encoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.eb_encoder = None
        self.nb_encoder = None

    def forward(self, graph):
        device = self.device
        node_attr, _, edge_attr, _ = decompose_graph(graph)

        # Move to same device
        if node_attr is not None:
            node_attr = node_attr.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)

        # Initialize MLPs dynamically if not already
        if self.nb_encoder is None and node_attr is not None:
            self.nb_encoder = build_mlp(node_attr.shape[1], self.hidden_size, self.hidden_size).to(device)
        if self.eb_encoder is None and edge_attr is not None:
            self.eb_encoder = build_mlp(edge_attr.shape[1], self.hidden_size, self.hidden_size).to(device)

        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        edge_index = graph.edge_index.to(device) if hasattr(graph, "edge_index") else None

        return Data(x=node_, edge_attr=edge_, edge_index=edge_index)




class GnBlock(nn.Module):

    def __init__(self, hidden_size=128):

        super(GnBlock, self).__init__()


        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
    
        graph_last = copy_geometric_data(graph)
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)
        edge_attr = graph_last.edge_attr + graph.edge_attr
        x = graph_last.x + graph.x
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)



class Decoder(nn.Module):
    def __init__(self, hidden_size=128, output_size=3):
        super(Decoder, self).__init__()
        # hidden_size = dimension of node features from encoder
        self.decode_module = build_mlp(in_size=hidden_size, hidden_size=hidden_size, out_size=output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, hidden_size=128):
        super().__init__()

        self.encoder = Encoder(hidden_size=hidden_size)  # no edge_input_size or node_input_size
        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)
        
        self.decoder = Decoder(hidden_size=hidden_size, output_size=3)  # output_size=3 for acceleration in 3D

    def forward(self, graph):
        graph = self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)
        return decoded





