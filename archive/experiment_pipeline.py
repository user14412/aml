import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import numpy as np
import random
import time
import logging

# ================= 1. 基础配置与日志 =================
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 使用计算设备: {device}")

data_dir = "../TableShift Dataset/acsfoodstamps/" # 确保路径正确
batch_size = 512
epochs = 3
learning_rate = 0.01
num_runs = 3 # 满足作业要求：至少3次独立运行

# 固定随机种子函数，保证可复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ================= 2. 数据加载 (只加载一次以节省时间) =================
def load_and_tensorize(x_path, y_path):
    X_df = pd.read_csv(x_path, dtype='float32')
    y_df = pd.read_csv(y_path, dtype='float32')
    return TensorDataset(torch.tensor(X_df.values), torch.tensor(y_df.values))

logger.info("正在加载数据集 (仅加载一次)...")
train_dataset = load_and_tensorize(data_dir + "acsfoodstamps_Xtrain.csv", data_dir + "acsfoodstamps_ytrain.csv")
id_test_dataset = load_and_tensorize(data_dir + "acsfoodstamps_Xidtest.csv", data_dir + "acsfoodstamps_yidtest.csv")
ood_test_dataset = load_and_tensorize(data_dir + "acsfoodstamps_Xood.csv", data_dir + "acsfoodstamps_yood.csv")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
id_test_loader = DataLoader(id_test_dataset, batch_size=batch_size, shuffle=False)
ood_test_loader = DataLoader(ood_test_dataset, batch_size=batch_size, shuffle=False)
input_dim = train_dataset[0][0].shape[0]

# ================= 3. 模型定义 =================
class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.network(x)

# ================= 4. 核心评估函数 =================
def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            preds = (logits > 0).float().cpu()
            all_preds.extend(preds.numpy())
            all_targets.extend(batch_y.numpy())
            
    # 计算作业要求的三大指标
    acc = accuracy_score(all_targets, all_preds) * 100
    bacc = balanced_accuracy_score(all_targets, all_preds) * 100
    f1 = f1_score(all_targets, all_preds, average='macro') * 100
    return {"Acc": acc, "BAcc": bacc, "F1": f1}

# ================= 5. 单次实验完整流程 =================
def run_single_experiment(run_idx, seed):
    set_seed(seed)
    logger.info(f"\n" + "-"*15 + f" 开始第 {run_idx} 次运行 (Seed={seed}) " + "-"*15)
    
    # 每次运行必须重新初始化模型
    model = SimpleMLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 训练阶段
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    # 评估阶段 (此处模拟的是 Non-Adaptation Baseline)
    # 如果你后续写了TTA，就在这里替换成你的 TTA 评估函数
    id_metrics = evaluate(model, id_test_loader)
    ood_metrics = evaluate(model, ood_test_loader)
    
    logger.info(f"ID 测试集  -> Acc: {id_metrics['Acc']:.2f} | BAcc: {id_metrics['BAcc']:.2f} | F1: {id_metrics['F1']:.2f}")
    logger.info(f"OOD 测试集 -> Acc: {ood_metrics['Acc']:.2f} | BAcc: {ood_metrics['BAcc']:.2f} | F1: {ood_metrics['F1']:.2f}")
    
    return id_metrics, ood_metrics

# ================= 6. 主程序：多轮运行与结果汇总 =================
if __name__ == "__main__":
    seeds = [42, 3407, 2026] # 选用三个经典的随机种子
    
    all_id_results = {"Acc": [], "BAcc": [], "F1": []}
    all_ood_results = {"Acc": [], "BAcc": [], "F1": []}
    
    # 执行 3 次运行
    for i, seed in enumerate(seeds):
        id_res, ood_res = run_single_experiment(i+1, seed)
        for key in all_id_results.keys():
            all_id_results[key].append(id_res[key])
            all_ood_results[key].append(ood_res[key])
            
    # 计算均值和方差，生成报告
    logger.info("\n" + "="*15 + " 📄 最终学术报告 (可直接填入论文) " + "="*15)
    
    def print_stat(name, id_list, ood_list):
        id_mean, id_std = np.mean(id_list), np.std(id_list)
        ood_mean, ood_std = np.mean(ood_list), np.std(ood_list)
        gap = id_mean - ood_mean
        logger.info(f"{name: <18}: ID = {id_mean:.2f}±{id_std:.2f}  |  OOD = {ood_mean:.2f}±{ood_std:.2f}  |  Gap = {gap:.2f}")

    logger.info(f"Method: Non-Adaptation Baseline (MLP)")
    logger.info("-" * 65)
    print_stat("Accuracy", all_id_results["Acc"], all_ood_results["Acc"])
    print_stat("Balanced Accuracy", all_id_results["BAcc"], all_ood_results["BAcc"])
    print_stat("F1-score (Macro)", all_id_results["F1"], all_ood_results["F1"])
    logger.info("-" * 65)
    logger.info("注: Gap (Generalization Gap) = ID Mean - OOD Mean (数值越大说明模型在OOD上退化越严重)")