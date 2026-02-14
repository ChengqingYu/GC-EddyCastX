import torch
from torch import nn, optim
import torch.nn.functional as F

class back_encorder(nn.Module):
    def __init__(self, seq_len, back_size, num_id, hidden_dim, dropout, num_head, num_layer):
        super(back_encorder, self).__init__()

        self.seq_len = seq_len
        self.embed_back = embed_back(seq_len, back_size, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead = num_head,dropout=dropout, dim_feedforward = hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layer)
        self.MLP = nn.Linear(hidden_dim, num_id)
        self.droupt = nn.Dropout(dropout)


    def forward(self, feature, background):

        b, T,_ = feature.shape
        hidden = self.embed_back(feature, background).transpose(-2, -1)
        hidden = self.transformer_encoder(hidden)
        hidden = hidden[:, -1,:]
        hidden = hidden.reshape(b, T, -1)
        hidden = self.MLP(hidden)

        return hidden # [b, T, hidden_size]
    
class embed_back(nn.Module):
    def __init__(self, seq_len, back_size, hidden_dim):
        super(embed_back, self).__init__()
        self.embed_layer = nn.Conv1d(in_channels = back_size[0] + 3, out_channels=hidden_dim, kernel_size=1)
        self.Location_emb = nn.Parameter(torch.empty(back_size[1], back_size[2]))
        nn.init.xavier_uniform_(self.Location_emb)

        self.clas_emb = nn.Parameter(torch.empty(seq_len, back_size[0] + 3))
        nn.init.xavier_uniform_(self.clas_emb)

    def forward(self, feature, background):
        b, T, c , w, h =  background.shape
        background = background.reshape(b, T, c, -1)
        feature = feature[:,:,[-2,-1]].unsqueeze(-1).expand(-1, -1, -1, w * h)

        ### Position encoding
        Location_emb = self.Location_emb.reshape(-1)
        Location_emb = Location_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, T, -1, -1)
        hidden = torch.cat([background,feature,Location_emb], dim = 2)

        ### [cls] token        
        clas_emb = self.clas_emb.unsqueeze(0).unsqueeze(-1).expand(b, T, -1, -1)
        clas_emb = clas_emb.reshape(b * T, c + 3, 1)

        # background information
        hidden = hidden.reshape(b * T, c + 3, w * h)
        hidden = torch.cat([clas_emb, hidden], dim = -1)
        hidden = self.embed_layer(hidden)

        return hidden  # [b * T, hidden_size , w * h] 
