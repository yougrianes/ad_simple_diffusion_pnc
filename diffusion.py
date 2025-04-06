import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder


# 简单的神经网络模型
class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleDiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 扩散过程
def diffusion_process(x, noise_level):
    noise = torch.randn_like(x) * noise_level
    return x + noise


# 逆扩散过程（训练模型）
def train_diffusion_model(model, dataloader, num_epochs, learning_rate, noise_level):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            noisy_batch = diffusion_process(batch, noise_level)
            optimizer.zero_grad()
            output = model(noisy_batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')


# 自定义数据集类
class NuPlanTrajectoryDataset(Dataset):
    def __init__(self, scenarios):
        self.scenarios = scenarios
        self.trajectories = []
        for scenario in scenarios:
            ego_states = scenario.get_ego_states()
            trajectory = np.array([[state.rear_axle.x, state.rear_axle.y] for state in ego_states])
            self.trajectories.append(trajectory.flatten())

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return torch.tensor(self.trajectories[idx], dtype=torch.float32)


# 加载NuPlan数据集
def load_nuplan_dataset(data_root, version):
    nuplan_db = NuPlanDB(data_root, version)
    scenario_builder = NuPlanScenarioBuilder(nuplan_db)
    scenarios = scenario_builder.get_scenarios()
    return scenarios


# 超参数设置
input_dim = 10  # 根据实际轨迹数据调整
hidden_dim = 20
output_dim = 10  # 根据实际轨迹数据调整
num_epochs = 100
learning_rate = 0.001
noise_level = 0.1
batch_size = 16

# 加载数据
data_root = "/home/li-ruiqin/nuplan/dataset"  # 替换为实际的数据根目录
version = "nuplan_v1.1-mini"  # 替换为实际的数据集版本
scenarios = load_nuplan_dataset(data_root, version)
dataset = NuPlanTrajectoryDataset(scenarios)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
model = SimpleDiffusionModel(input_dim, hidden_dim, output_dim)

# 训练模型
train_diffusion_model(model, dataloader, num_epochs, learning_rate, noise_level)
    