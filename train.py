import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import numpy as np
from dataset import FpcDataset
from model.simulator import Simulator
from utils.noise import get_velocity_noise
from utils.utils import NodeType
import os
import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

# 配置
dataset_dir = "data"
batch_size = 20
noise_std = 2e-2
num_epochs = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
checkpoint_dir = "checkpoints"
log_dir = "runs"  # TensorBoard 日志目录
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 初始化模型与优化器
simulator = Simulator(message_passing_num=15, node_input_size=11, edge_input_size=3, device=device)
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')

# TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

# 数据预处理
transformer = T.Compose([
    T.FaceToEdge(),
    T.Cartesian(norm=False),
    T.Distance(norm=False)
])

def train_one_epoch(model: Simulator, dataloader, optimizer, transformer, device, noise_std):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for graph in tqdm.tqdm(dataloader):
        graph = transformer(graph)
        graph = graph.to(device)

        node_type = graph.x[:, 0]  # "node_type, cur_v"
        velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph, velocity_sequence_noise)

        mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)
        errors = ((predicted_acc - target_acc) ** 2)[mask]
        loss = torch.mean(errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model: Simulator, dataloader, transformer, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for graph in dataloader:
            graph = transformer(graph)
            graph = graph.to(device)

            node_type = graph.x[:, 0]
            predicted_velocity = model(graph, None)

            mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)
            errors = ((predicted_velocity - graph.y) ** 2)[mask]
            loss = torch.sqrt(torch.mean(errors))
            losses.append(loss.item())

    return np.mean(losses)


if __name__ == '__main__':
    # 加载训练和验证数据集
    train_dataset = FpcDataset(data_root=dataset_dir, split='train')
    valid_dataset = FpcDataset(data_root=dataset_dir, split='valid')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    simulator.to(device)

    best_valid_loss = float('inf')
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):

        train_loss = train_one_epoch(simulator, train_loader, optimizer, transformer, device, noise_std)
        valid_loss = evaluate(simulator, valid_loader, transformer, device)

        print(f"Epoch {epoch}/{num_epochs} Train Loss: {train_loss:.2e} Valid Loss: {valid_loss:.2e}")

        # 👇 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)

        # 保存最优模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': simulator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, checkpoint_path)
            print(f"  -> New best model saved at epoch {epoch} with valid loss {valid_loss:.2e}")

    writer.close()
    print(f"\nTraining finished. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")