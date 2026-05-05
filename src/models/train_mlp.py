import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.mlp_model import TurtleNeckMLP

def train_model(train_x, train_y, val_x, val_y, config):
    # 1. 데이터셋 준비 (Numpy -> Tensor)
    train_dataset = TensorDataset(
        torch.FloatTensor(train_x), 
        torch.FloatTensor(train_y).view(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # 2. 모델 및 최적화 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TurtleNeckMLP(input_dim=99).to(device)
    criterion = nn.BCELoss()
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
    torch.save(model.state_dict(), "weights/mlp.pth")
    print("Model saved to weights/mlp.pth")

if __name__ == "__main__":
    # 예시 설정값
    cfg = {
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 50
    }
    # 데이터 로드 로직 (실제 구현 시 필요)
    # train_model(X_train, y_train, X_val, y_val, cfg)