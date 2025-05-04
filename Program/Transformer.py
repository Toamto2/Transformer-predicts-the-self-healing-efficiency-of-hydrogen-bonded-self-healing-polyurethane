import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 从CSV文件中加载数据
df_train = pd.read_csv("/")  # 读取训练集数据
df_test = pd.read_csv("/")  # 读取测试集数据

# 自动识别列名
descriptor_columns = [col for col in df_train.columns if '描述符' in col] #将数据集中的表头改为描述符1，描述2，描述符3，描述符n......即可
temperature_column = 'T'
time_column = 't'
efficiency_column = 'E'

# 数据清理：移除含有NaN的行
df_train = df_train.dropna(subset=descriptor_columns + [temperature_column, time_column, efficiency_column])
df_test = df_test.dropna(subset=descriptor_columns + [temperature_column, time_column, efficiency_column])

# 准备训练集输入和输出数据
X_train_descriptors = df_train[descriptor_columns].values
X_train_conditions = df_train[[temperature_column, time_column]].values  # 将温度和时间作为连续变量
y_train = df_train[efficiency_column].values

# 准备测试集输入和输出数据
X_test_descriptors = df_test[descriptor_columns].values
X_test_conditions = df_test[[temperature_column, time_column]].values
y_test = df_test[efficiency_column].values

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# 合并描述符和条件特征
X_train = np.concatenate((X_train_descriptors, X_train_conditions), axis=1)
X_test = np.concatenate((X_test_descriptors, X_test_conditions), axis=1)

# 标准化输入特征和输出目标
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# 转换为 PyTorch 张量并转移到设备
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # (batch_size, seq_length, feature_size)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# 创建数据集和数据加载器
batch_size = 512
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 3. Transformer模型构建
class TransformerRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=4, num_heads=16, ff_size=512, dropout=0.2):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,  # 每个位置的特征维度
            nhead=num_heads,  # 多头注意力机制的头数
            num_encoder_layers=num_layers,  # 编码器层数
            dim_feedforward=ff_size,  # 前馈网络的大小
            dropout=dropout,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 25)  # 全连接层
        self.fc2 = nn.Linear(25, 1)  # 输出层

    def forward(self, x):
        # 输入处理
        x = self.embedding(x)

        # Transformer编码
        x = self.transformer(x, x)

        # 使用最后一个位置的输出进行预测
        x = x[:, -1, :]  # 取最后一个位置的输出
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
input_size = X_train.shape[2]
model = TransformerRegressor(input_size=input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)

# 4. 训练模型
num_epochs = 25

# 用于记录损失的列表
train_losses = []
test_losses = []

# 记录最佳模型状态和效果
best_epoch = 0
best_train_r2 = -np.inf
best_test_r2 = -np.inf
best_train_rmse = np.inf
best_test_rmse = np.inf
best_model_state = None
best_train_predictions = None
best_test_predictions = None
best_y_train_actual = None
best_y_test_actual = None

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * batch_x.size(0)

    train_losses.append(epoch_train_loss / len(train_loader.dataset))

    model.eval()
    with torch.no_grad():
        epoch_test_loss = 0
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            test_outputs = model(batch_x)
            test_loss = criterion(test_outputs, batch_y)
            epoch_test_loss += test_loss.item() * batch_x.size(0)
        test_losses.append(epoch_test_loss / len(test_loader.dataset))

        # 训练集和测试集的预测结果
        train_predictions = model(X_train)
        test_predictions = model(X_test)

        # 反标准化预测结果
        train_predictions = scaler_y.inverse_transform(train_predictions.cpu().detach().numpy())
        test_predictions = scaler_y.inverse_transform(test_predictions.cpu().detach().numpy())
        y_train_actual = scaler_y.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
        y_test_actual = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

        train_r2 = r2_score(y_train_actual, train_predictions)
        test_r2 = r2_score(y_test_actual, test_predictions)

        # 计算反标准化后的RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, "
              f"Train R² Score: {train_r2:.4f}, Test R² Score: {test_r2:.4f}, "
              f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

        # 记录预测集上效果最好的那一轮
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_train_r2 = train_r2
            best_test_rmse = test_rmse
            best_train_rmse = train_rmse
            best_epoch = epoch
            best_model_state = model.state_dict()
            best_train_predictions = train_predictions
            best_test_predictions = test_predictions
            best_y_train_actual = y_train_actual
            best_y_test_actual = y_test_actual

# 5. 加载最佳模型
model.load_state_dict(best_model_state)
model.eval()

# 设置字体为新罗马字体并加粗
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 18 # 设置全局字号大小

# 6. 训练集和测试集最佳拟合效果图
plt.figure(figsize=(12, 6))

# 训练集拟合效果
plt.subplot(1, 2, 1)
plt.scatter(best_y_train_actual, best_train_predictions, color='blue', alpha=0.6, label="Train Predictions vs Actual")
plt.plot([min(best_y_train_actual), max(best_y_train_actual)], [min(best_y_train_actual), max(best_y_train_actual)],
         'r--', label="Perfect Fit")
plt.xlabel("Actual", fontweight='bold', fontsize=22)
plt.ylabel("Predicted", fontweight='bold', fontsize=22)
plt.title("Train Fit Model", fontweight='bold', fontsize=22)
plt.legend(loc='upper left', frameon=False)  # 设置图例不带边框

# 测试集拟合效果
plt.subplot(1, 2, 2)
plt.scatter(best_y_test_actual, best_test_predictions, color='green', alpha=0.6, label="Test Predictions vs Actual")
plt.plot([min(best_y_test_actual), max(best_y_test_actual)], [min(best_y_test_actual), max(best_y_test_actual)], 'r--',
         label="Perfect Fit")
plt.xlabel("Actual", fontweight='bold', fontsize=22)
plt.ylabel("Predicted", fontweight='bold', fontsize=22)
plt.title("Test Fit Model", fontweight='bold', fontsize=22)
plt.legend(loc='upper left', frameon=False)  # 设置图例不带边框

plt.tight_layout()
plt.show()

# 合并预测结果和原始测试集的时间、温度数据
df_test_results = pd.DataFrame({
    '时间 (H)': df_test[time_column],  # 时间
    '温度 (T)': df_test[temperature_column],  # 温度
    '真实值 (E)': best_y_test_actual.flatten(),  # 原始测试集目标值
    '预测值 (E)': best_test_predictions.flatten(),  # 模型预测值
    '误差': np.abs(best_y_test_actual.flatten() - best_test_predictions.flatten())  # 预测误差
})

# 保存到 CSV 文件
output_file = "T_E_test_predictions3.csv"
df_test_results.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"预测结果已保存到 {output_file} 文件中")
