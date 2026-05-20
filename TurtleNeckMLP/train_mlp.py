import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from TurtleNeckMLP.mlp_model import TurtleNeckMLP

class AugmentedPoseDataset(Dataset):
    """데이터 증강을 포함한 포즈 데이터셋"""
    def __init__(self, x_data, y_data, augment=False):
        self.x = x_data
        self.y = y_data
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].copy()
        y = self.y[idx]

        if self.augment:
            # 1. 가우시안 노이즈 추가 (MediaPipe의 미세한 떨림 모사)
            noise = np.random.normal(0, 0.005, x.shape).astype(np.float32)
            x += noise
            
            # 2. 미세한 전체 스케일 변화 (사용자와 카메라 거리 변화 모사)
            scale = np.random.uniform(0.98, 1.02)
            x *= scale

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def train_model(data_path="data/processed", weight_path="weights/mlp.pth"):
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

    # 하이퍼파라미터 조정
    hyper_params = {
        'batch_size': 64,      # 배치 사이즈를 키워 학습 안정성 확보
        'lr': 0.001, 
        'epochs': 100,         # 더 충분히 학습하도록 증가
        'weight_decay': 1e-4   # 과적합 방지를 위한 가중치 감쇠
    }

    # 1. 데이터셋 준비 (학습셋에만 증강 적용)
    train_dataset = AugmentedPoseDataset(train_x, train_y, augment=True)
    val_dataset = AugmentedPoseDataset(val_x, val_y, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyper_params['batch_size'], shuffle=False)

    # 2. 모델 및 최적화 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TurtleNeckMLP(input_dim=99, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay'])
    
    # 학습률 스케줄러: 검증 정확도가 7번 동안 안 오르면 학습률을 절반으로 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)

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
        
        # 에폭 끝날 때마다 스케줄러에게 현재 상태 보고
        scheduler.step(val_acc)

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