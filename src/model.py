import torch 
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16) -> None:
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MLSTM_FCN(nn.Module):
    def __init__(self, num_classes, max_seq_len, num_features, num_lstm_out=128, num_lstm_layers=1, conv1_nf=128, conv2_nf=256, conv3_nf=128, lstm_drop_p=0.8, fc_drop_p=0.3, kernels = [8, 5, 3]):
        super(MLSTM_FCN, self).__init__()
        
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        
        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers
        
        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf
        
        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p
        
        self.lstm = nn.LSTM(input_size=self.num_features, hidden_size=self.num_lstm_out, num_layers=self.num_lstm_layers, batch_first=True)

        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, kernels[0])
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, kernels[1])
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, kernels[2])
        
        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)
        
        self.se1 = SELayer(self.conv1_nf)
        self.se2 = SELayer(self.conv2_nf)
        
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.lstm_drop_p)
        
        self.fc = nn.Linear(self.conv3_nf + self.num_lstm_out, self. num_classes)

    def forward(self, x):
        x1, (_, _) = self.lstm(x)
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2, 1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)
        
        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)
        
        return x_out
    

def main():
    nclasses = 2
    
    device = 'cuda:0'
    input = torch.randn(20, 20, 2).to(device)  
    model = MLSTM_FCN(num_classes=nclasses, max_seq_len=20, num_features=2).to(device)
    
    output = model(input)
    
    print(output.shape)
    

if __name__ == '__main__':
    main()