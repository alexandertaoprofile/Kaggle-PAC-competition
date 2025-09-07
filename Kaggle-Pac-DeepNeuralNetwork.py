import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# 读取数据
file_path = 'C:/Users/10946/OneDrive/Desktop/columbia/note/analysisData.csv'
data = pd.read_csv(file_path)

# 特征选择 (与xgboost代码中的特征一致)
features = [
    'make_name', 'model_name', 'body_type', 'fuel_tank_volume_gallons', 'fuel_type', 
            'highway_fuel_economy', 'city_fuel_economy', 'torque', 'transmission', 
            'transmission_display', 'wheel_system', 'wheelbase_inches', 'length_inches', 
            'width_inches', 'height_inches', 'engine_type', 'engine_displacement', 'horsepower', 
            'daysonmarket', 'maximum_seating', 'year', 'fleet', 'frame_damaged', 
            'franchise_dealer', 'franchise_make', 'has_accidents', 'isCab', 'is_cpo', 'is_new', 
            'mileage', 'owner_count', 'salvage', 'seller_rating'
]

# 选择需要的特征列和价格目标列
X = data[features].copy()
y = data['price']

# 处理类别变量，转换为数值变量
X = pd.get_dummies(X)

# 填补缺失值（均值填补）
X = X.fillna(X.mean())

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换为 PyTorch tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

# 定义 DNN 模型
class DNNModel(nn.Module):
    def __init__(self, input_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 初始化模型，定义损失函数和优化器
input_size = X_train.shape[1]
model = DNNModel(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 预测和计算 RMSE
model.eval()
y_train_pred = model(X_train_tensor).detach().numpy()
y_val_pred = model(X_val_tensor).detach().numpy()

# 计算 RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
