import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from TurtleNeckMLP.mlp_model import TurtleNeckMLP

def train_model(data_path="data/processed/mlp", weight_path="weights/mlp.pth"):
    # 데이터 로드
    try:
        X = np.load(os.path.join(data_path, "X.npy")).astype(np.float32)
        y = np.load(os.path.join(data_path, "y.npy")).astype(np.int64)
        print(f"데이터 로드 완료: {X.shape}")
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return

    # 간단한 Train/Val 분리 (8:2)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * 0.8)
    train_x, val_x = X[indices[:split]], X[indices[split:]]
    train_y, val_y = y[indices[:split]], y[indices[split:]]

    config = {'batch_size': 32, 'lr': 0.001, 'epochs': 50}

    # 1. 데이터셋 준비 (Numpy -> Tensor)
    train_dataset = TensorDataset(
        torch.FloatTensor(train_x), 
        torch.LongTensor(train_y)  # CrossEntropyLoss는 Long 타입을 사용
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # 2. 모델 및 최적화 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TurtleNeckMLP(input_dim=99, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 3. 학습 루프
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {total_loss/len(train_loader):.4f}")

    # 4. 모델 저장
    torch.save(model.state_dict(), weight_path)
    print(f"Model saved to {weight_path}")

if __name__ == "__main__":
    # 예시 설정값
    cfg = {
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 50
    }
    # 데이터 로드 로직 (실제 구현 시 필요)
    # train_model(X_train, y_train, X_val, y_val, cfg)