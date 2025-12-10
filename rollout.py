import argparse
import os
import pickle

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm

from dataset import FpcDataset
from model.simulator import Simulator
from utils.utils import NodeType

def rollout_error(predicteds, targets):
    number_len = targets.shape[0]
    squared_diff = np.square(predicteds - targets).reshape(number_len, -1)
    loss = np.sqrt(np.cumsum(np.mean(squared_diff, axis=1), axis=0)/np.arange(1, number_len+1))
    for show_step in range(0, 1000000, 50):
        if show_step <number_len:
            print('testing rmse  @ step %d loss: %.2e'%(show_step, loss[show_step]))
        else: break
    return loss


@torch.no_grad()
def rollout(model, dataset, rollout_index=1):

    num_sampes_per_tra = dataset.num_sampes_per_tra
    predicted_velocity = None
    mask=None
    predicteds = []
    targets = []

    for i in range(num_sampes_per_tra):
        index = rollout_index * num_sampes_per_tra + i
        graph = dataset[index]
        graph = transformer(graph)
        graph = graph.cuda()

        if mask is None:
            node_type = graph.x[:, 0]
            mask = torch.logical_or(node_type==NodeType.NORMAL, node_type==NodeType.OUTFLOW)
            mask = torch.logical_not(mask)

        if predicted_velocity is not None:
            graph.x[:, 1:3] = predicted_velocity.detach()
        
        next_v = graph.y
        with torch.no_grad():
            predicted_velocity = model(graph, velocity_sequence_noise=None)

        predicted_velocity[mask] = next_v[mask]

        predicteds.append(predicted_velocity.detach().cpu().numpy())
        targets.append(next_v.detach().cpu().numpy())
        
    crds = graph.pos.cpu().numpy()
    result = [np.stack(predicteds), np.stack(targets)]

    os.makedirs('result', exist_ok=True)
    with open('result/result' + str(rollout_index) + '.pkl', 'wb') as f:
        pickle.dump([result, crds], f)
    
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Implementation of MeshGraphNets')
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="gpu number: 0 or 1")

    parser.add_argument("--model_dir",
                        type=str,
                        default='checkpoints/best_model.pth')

    parser.add_argument("--test_split", type=str, default='test')
    parser.add_argument("--rollout_num", type=int, default=1)

    args = parser.parse_args()

    # load model
    torch.cuda.set_device(args.gpu)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    simulator = Simulator(message_passing_num=15, node_input_size=11, edge_input_size=3, device=device)

    state_dict  = torch.load(args.model_dir, weights_only=False)
    simulator.load_state_dict(state_dict['model_state_dict'])
    simulator.eval()

    # prepare dataset
    dataset_dir = "data"
    dataset = FpcDataset(dataset_dir, split=args.test_split)
    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    for i in range(args.rollout_num):
        result = rollout(simulator, dataset, rollout_index=i)
        print('------------------------------------------------------------------')
        rollout_error(result[0], result[1])


    



