import torch
from torch import nn, optim
import torch.nn.functional as F


class cross_attn(nn.Module):
    def __init__(self, Input_len, seq_len,dropout,num_head):
        super(cross_attn, self).__init__()
        self.query = nn.Conv1d(in_channels=Input_len,out_channels = Input_len,kernel_size=1)
        self.key = nn.Conv1d(in_channels=Input_len,out_channels=Input_len,kernel_size=1)
        self.value = nn.Conv1d(in_channels=Input_len,out_channels=Input_len,kernel_size=1)
        self.laynorm = nn.LayerNorm([Input_len])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv2d(in_channels=Input_len,out_channels=Input_len,kernel_size=(1,num_head))

    def forward(self, x, x2):
        x = x.transpose(-2, -1)
    
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x2))
            k = self.dropout(self.key(x)).transpose(-2, -1)
            v = self.dropout(self.value(x))
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32)/self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result,line.unsqueeze(-1)],dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x + result
        x = x.transpose(-2, -1)
        x = self.laynorm(x)
        return x


class cross_attn_2(nn.Module):
    def __init__(self, Input_len, seq_len, dropout,num_head):
        super(cross_attn_2, self).__init__()
        self.query = nn.Conv1d(in_channels=Input_len,out_channels = Input_len,kernel_size=1)
        self.key = nn.Conv1d(in_channels=Input_len,out_channels=Input_len,kernel_size=1)
        self.value = nn.Conv1d(in_channels=Input_len,out_channels=Input_len,kernel_size=1)
        self.laynorm = nn.LayerNorm([Input_len])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv2d(in_channels=Input_len,out_channels=Input_len,kernel_size=(1,num_head))

    def forward(self, x, x2):
        x = x.transpose(-2, -1)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x))
            k = self.dropout(self.key(x2)).transpose(-2, -1)
            v = self.dropout(self.value(x2))
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32)/self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result,line.unsqueeze(-1)],dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x + result
        x = x.transpose(-2, -1)
        x = self.laynorm(x)
        return x