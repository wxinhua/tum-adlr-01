import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        return x    
    

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn_block1 = CNNBlock(4, 32)
        self.cnn_block2 = CNNBlock(32, 64)
        self.cnn_block3 = CNNBlock(64, 128)
        self.flatten = nn.Flatten(start_dim=2)
        self.reshape = nn.Linear(4 * 128 * 2 * 2, 2048)
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 自动调整输入以添加批量大小维度
        if x.dim() == 4:  # 检查是否缺少批量维度
            x = x.unsqueeze(0)  # 添加批量大小维度
        # Permute the input to match the expected format (batch_size, channels, depth, height, width)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        x = self.flatten(x)
        x = x.reshape(x.size(0), 4, -1)
        x, _ = self.lstm(x)
        x = self.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x


