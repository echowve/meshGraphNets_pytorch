from dataset import FPC
from model.simulator import Simulator
import torch
from utils.noise import get_velocity_noise
import time
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from utils.normalization import Normalizer

dataset_dir = "/usagers3/nashe/cfd_dataset"
max_epochs = 1 #50
batch_size = 1 #20
noise_std=2e-2

print_batch = 1 #10
save_batch = 1 #200


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

simulator = Simulator(message_passing_num=2, device=device)
simulator._node_normalizer = Normalizer(size=6, device=device)
simulator._output_normalizer = Normalizer(size=3, device=device)  # 3D velocity output

optimizer= torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')


def train(model:Simulator, dataloader, optimizer):

    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.cuda()

        node_type = graph.x[:, 0] #"node_type, cur_v, pressure, time"
        # velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph)
        mask = torch.logical_or(node_type==NodeType.NORMAL, node_type==NodeType.OUTFLOW)
        
        errors = ((predicted_acc - target_acc)**2)[mask]
        loss = torch.mean(errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            print('batch %d [loss %.2e]'%(batch_index, loss.item()))

        if batch_index % save_batch == 0:
            model.save_checkpoint()

if __name__ == '__main__':

    dataset_fpc = FPC(dataset_dir=dataset_dir, split='train', max_epochs=max_epochs)

    train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=0)#10)
    transformer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])

    for i, graph in enumerate(train_loader):
        print(f"Graph {i}")
        print("x shape:", graph.x.shape)
        print("y shape:", graph.y.shape)
        print("pos shape:", graph.pos.shape)
        print("num edges:", graph.edge_index.shape[1])
        if hasattr(graph, 'face') and graph.face is not None:
            print("face shape:", graph.face.shape)

        if i > 2:  # just look at a few
            break
    print(simulator._node_normalizer._acc_sum.shape)

    train(simulator, train_loader, optimizer)
