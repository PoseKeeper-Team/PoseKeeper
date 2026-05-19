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

    hyper_params = {'batch_size': 32, 'lr': 0.001, 'epochs': 50}

    # 1. 데이터셋 준비 (Numpy -> Tensor)
    train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
    val_dataset = TensorDataset(torch.FloatTensor(val_x), torch.LongTensor(val_y))

    train_loader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyper_params['batch_size'], shuffle=False)

    # 2. 모델 및 최적화 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TurtleNeckMLP(input_dim=99, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyper_params['lr'])

    best_val_acc = 0.0
    print(f"\n[학습 시작] Device: {device} | Epochs: {hyper_params['epochs']}")

    # 3. 학습 루프
    for epoch in range(hyper_params['epochs']):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # 4. 에폭별 검증 수행
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # 최고 성능 갱신 시 모델 저장
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_data = {
                "model_state_dict": model.state_dict(),
                "model_name": "mlp",
                "input_dim": 99,
                "num_classes": 3,
                "class_names": ["normal", "turtle_neck", "severe_turtle_neck"],
                "best_val_accuracy": best_val_acc,
                "config": hyper_params
            }
            torch.save(save_data, weight_path)
        
        print(f"Epoch [{epoch+1:2d}/{hyper_params['epochs']}] "
              f"Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:5.2f}% {'[Best!]' if is_best else ''}")

    print(f"\n[학습 종료] 최고 검증 정확도: {best_val_acc:.2f}%")
    print(f"모델 가중치가 '{weight_path}' 에 저장되었습니다.")

if __name__ == "__main__":
    # 예시 설정값
    cfg = {
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 50
    }
    # 데이터 로드 로직 (실제 구현 시 필요)
    # train_model(X_train, y_train, X_val, y_val, cfg)