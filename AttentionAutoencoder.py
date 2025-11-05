import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# 检测 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 启用 CuDNN 加速
torch.backends.cudnn.benchmark = True

# 自定义 Transformer Encoder 层
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # 比 ReLU 更平滑
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed Forward
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

# 自编码器模型
class AttentionAutoencoder(nn.Module):
    def __init__(self, timesteps, n_features, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.embedding = nn.Linear(n_features, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, ff_dim)
        self.decoder = TransformerEncoder(embed_dim, num_heads, ff_dim)
        self.output_layer = nn.Linear(embed_dim, n_features)
        self.timesteps = timesteps

    def forward(self, x):
        # Encoder
        x = self.embedding(x)
        x = self.encoder(x)
        
        # Global Average Pooling
        encoded = x.mean(dim=1)  # [batch_size, embed_dim]
        
        # Decoder
        encoded_expanded = encoded.unsqueeze(1).expand(-1, self.timesteps, -1)
        decoded = self.decoder(encoded_expanded)
        reconstructed = self.output_layer(decoded)
        return reconstructed, encoded

# 读取 Excel 文件
def read_excel_file(file_path):
    return pd.read_excel(file_path)

# 数据预处理
def preprocess_data(df):
    features = df.iloc[:, 1:].values.astype(np.float32)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_tensor = torch.FloatTensor(features_scaled).unsqueeze(1)  # [samples, 1, features]
    return features_tensor, scaler, df.iloc[:, 0]

# 保存结果到 Excel
def save_to_excel(original_df, encoded_data, output_path):
    encoded_df = pd.DataFrame(encoded_data)
    result_df = pd.concat([original_df.iloc[:, 0], encoded_df], axis=1)
    result_df.to_excel(output_path, index=False)

# 主函数
def main():
    # 参数设置
    file_path = "E:\\SKY\\data.xlsx"
    
    output_path = "encoded_data.xlsx"
    embed_dim = 6                   # 输出的特征数
    num_heads = 6                   # 更多注意力头
    ff_dim = 64                     # 更大的前馈网络
    epochs = 100                     # 更多训练轮次
    batch_size = 64                 # 更大的批量（适合 GPU）
    learning_rate = 0.001
    patience = 10                    # 早停耐心值

    # 读取数据
    df = read_excel_file(file_path)
    data_scaled, scaler, first_column = preprocess_data(df)
    dataset = TensorDataset(data_scaled, data_scaled)
    
    # DataLoader 配置（pin_memory 加速 GPU 传输）
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=4  # 多线程加载数据
    )

    # 初始化模型并移至 GPU
    timesteps = 1
    n_features = data_scaled.shape[2]
    model = AttentionAutoencoder(timesteps, n_features, embed_dim, num_heads, ff_dim).to(device)
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    # 混合精度训练
    scaler_amp = GradScaler()
    best_loss = float('inf')
    patience_counter = 0

    # 训练循环
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                reconstructed, _ = model(batch_x)
                loss = criterion(reconstructed, batch_y)
            
            # 混合精度反向传播
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型并推理
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    with torch.no_grad():
        # 分批处理大数据集避免内存不足
        encoded_data = []
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            _, encoded = model(batch_x)
            encoded_data.append(encoded.cpu())
        encoded_data = torch.cat(encoded_data, dim=0).numpy()

    # 保存结果
    save_to_excel(df, encoded_data, output_path)
    print(f"降维后的数据已保存到 {output_path}")

if __name__ == "__main__":
    main()
