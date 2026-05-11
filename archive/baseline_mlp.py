import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import time
import copy
import torch.nn.functional as F
from utils.logger import logger

# 1. 基础设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 使用计算设备: {device}")

data_dir = "../TableShift Dataset/acsfoodstamps/" # 根据你的实际路径调整
batch_size = 512
epochs = 3 # 初步测试，只跑3个Epoch看看效果
learning_rate = 0.01

# 2. 数据加载函数 (把 Pandas 转成 PyTorch Tensor)
def load_and_tensorize(x_path, y_path):
    logger.info(f"正在加载 {x_path} ...")
    # 读取数据，并全部转换为 float32 格式
    X_df = pd.read_csv(x_path, dtype='float32')
    y_df = pd.read_csv(y_path, dtype='float32')
    
    # 转换为 Tensor
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_df.values, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)

logger.info("-" * 30)
# 加载 训练集、ID测试集(同分布)、OOD测试集(偏移)
train_dataset = load_and_tensorize(data_dir + "acsfoodstamps_Xtrain.csv", data_dir + "acsfoodstamps_ytrain.csv")
id_test_dataset = load_and_tensorize(data_dir + "acsfoodstamps_Xidtest.csv", data_dir + "acsfoodstamps_yidtest.csv")
ood_test_dataset = load_and_tensorize(data_dir + "acsfoodstamps_Xood.csv", data_dir + "acsfoodstamps_yood.csv")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
id_test_loader = DataLoader(id_test_dataset, batch_size=batch_size, shuffle=False)
ood_test_loader = DataLoader(ood_test_dataset, batch_size=batch_size, shuffle=False)

# 获取输入特征的维度 (截图里是239)
input_dim = train_dataset[0][0].shape[0]

# 3. 定义最简单的多层感知机 (MLP)
class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 输出1个神经元，用于二分类
        )

    def forward(self, x):
        return self.network(x)

model = SimpleMLP(input_dim).to(device)

# 4. 定义损失函数和优化器
# 因为是二分类任务，使用 BCEWithLogitsLoss (内部自带了Sigmoid，更稳定)
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 5. 训练循环
logger.info("\n" + "="*10 + " 开始训练 " + "="*10)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, 耗时: {time.time()-start_time:.2f}秒")

# 6. 测试评估函数
def evaluate_model(loader, dataset_name):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            # 输出的是logits，大于0相当于Sigmoid后大于0.5
            outputs = model(batch_X)
            preds = (outputs > 0).float().cpu() 
            all_preds.extend(preds.numpy())
            all_targets.extend(batch_y.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    logger.info(f"🎯 [{dataset_name}] Accuracy: {acc*100:.2f}% | F1 Score: {f1*100:.2f}%")

logger.info("\n" + "="*10 + " 开始评估 " + "="*10)
evaluate_model(id_test_loader, "ID_Test (同分布测试集)")
evaluate_model(ood_test_loader, "OOD_Test (偏移测试集)")

# ================= 新增：错题分析模块 =================
import matplotlib.pyplot as plt

# ================= 新增：置信度全景分析模块 =================
import matplotlib.pyplot as plt

def analyze_errors(loader, dataset_name):
    logger.info(f"\n🔍 开始分析 {dataset_name} 的预测置信度全景分布...")
    model.eval()
    errors_probs = []
    correct_probs = []  # 🌟 新增：存放正确预测的置信度
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = batch_y.numpy()
            
            # 记录模型当时给出的概率
            for i in range(len(probs)):
                p = probs[i][0]
                pred_label = 1 if p > 0.5 else 0
                # 记录它对预测类别的“自信程度”（总是大于等于0.5）
                confidence = p if pred_label == 1 else (1 - p)
                
                if pred_label != targets[i][0]:
                    errors_probs.append(confidence)
                else:
                    correct_probs.append(confidence)  # 🌟 收集正确的样本
                    
    # 画直方图
    plt.figure(figsize=(10, 6))
    
    # 🌟 核心修改：将错题（红）和对题（蓝）一起传给 hist，并开启堆叠 (stacked=True)
    # 把 errors_probs 放前面，红色块会在最底下，蓝色叠在上面
    plt.hist([errors_probs, correct_probs], bins=50, stacked=True, 
             color=['red', 'blue'], alpha=0.7, label=['WRONG', 'CORRECT'])
    
    plt.title(f"{dataset_name} - Confidence Distribution (Correct vs Wrong)")
    plt.xlabel("Model Confidence (0.5 = Hesitant, 1.0 = Blindly Confident)")
    plt.ylabel("Number of Samples")
    plt.axvline(x=0.5, color='gray', linestyle='--')
    plt.legend(loc='upper left')  # 显示图例
    plt.grid(axis='y', alpha=0.3)
    
    # 因为蓝色的数量大概率会碾压红色，导致红色看不清，
    # 你可以取消下面这行注释，把 Y 轴变成对数坐标 (Log Scale)，方便同时看清两者比例
    # plt.yscale('log') 

    plt.savefig("confidence_analysis.png") 
    logger.info("✅ 置信度全景分布图已保存为 confidence_analysis.png")

# 运行全景分析
analyze_errors(ood_test_loader, "OOD_Test")

# 7. TTA (测试时自适应) 评估函数
def evaluate_model_tta(dataset_name, dataloader, base_model, lr=0.001):
    logger.info(f"\n🚀 开始执行 TTA 评估: {dataset_name}")
    
    # 【极其重要】每次测试前，必须重新复制一份纯净的原始预训练模型！
    # 否则模型会在测试集上不断累积更新，甚至跨测试集污染
    tta_model = copy.deepcopy(base_model).to(device)
    
    # 定义专门用于测试阶段的优化器 (注意这里优化的是 tta_model 的参数)
    tta_optimizer = optim.AdamW(tta_model.parameters(), lr=lr)
    
    all_preds = []
    all_targets = []
    
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        
        # ==========================================
        # 🟢 第一步：自我适应 (Adaptation Step)
        # ==========================================
        tta_model.train() # 开启训练模式，允许梯度更新
        tta_optimizer.zero_grad()
        
        # 前向传播得到 logits
        logits = tta_model(batch_X)
        # 转换为 0~1 的概率
        probs = torch.sigmoid(logits)
        
        # 计算无监督损失：信息熵 (Entropy)
        # 限制下界防止 log(0) 报错
        p = torch.clamp(probs, 1e-6, 1.0 - 1e-6) 
        # 二分类的熵公式: - p*log(p) - (1-p)*log(1-p)
        entropy_loss = - (p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
        
        # 反向传播，更新模型参数！(注意：这里绝对没有用到 batch_y)
        entropy_loss.backward()
        tta_optimizer.step()
        
        # ==========================================
        # 🔵 第二步：最终预测 (Prediction Step)
        # ==========================================
        tta_model.eval() # 切换回评估模式
        with torch.no_grad():
            # 用刚刚更新过参数的模型，重新预测这批数据
            final_logits = tta_model(batch_X)
            preds = (final_logits > 0).float().cpu()
            
        all_preds.extend(preds.numpy())
        all_targets.extend(batch_y.numpy())
        
    # 计算指标
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    logger.info(f"🎯 [TTA - {dataset_name}] Accuracy: {acc*100:.2f}% | F1 Score: {f1*100:.2f}%")

# 运行 TTA 看看效果
evaluate_model_tta("OOD_Test (偏移测试集)", ood_test_loader, model)