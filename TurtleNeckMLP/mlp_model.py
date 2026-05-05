import torch
import torch.nn as nn

class TurtleNeckMLP(nn.Module):
    def __init__(self, input_dim=99, hidden_dims=[128, 64, 32]):
        super(TurtleNeckMLP, self).__init__()
        
        layers = []
        last_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            last_dim = h_dim
        
        # 이진 분류를 위한 최종 출력층
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch_size, 99) 형태의 Pose landmark tensor
        return: 0~1 사이의 거북목 확률
        """
        return self.model(x)