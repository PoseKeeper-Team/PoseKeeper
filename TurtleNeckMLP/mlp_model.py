import torch
import torch.nn as nn

class TurtleNeckMLP(nn.Module):
    def __init__(self, input_dim=99, hidden_dims=[128, 64, 32], num_classes=3):
        super(TurtleNeckMLP, self).__init__()
        
        layers = []
        last_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            last_dim = h_dim
        
        # 다중 분류(3개 클래스: 정상, 거북목, 심한 거북목)를 위한 출력층
        layers.append(nn.Linear(last_dim, num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch_size, 99) 형태의 Pose landmark tensor
        return: (batch_size, num_classes) 형태의 Logits
        """
        return self.model(x)