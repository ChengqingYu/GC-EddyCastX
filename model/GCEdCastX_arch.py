import torch
from torch import nn

from .embed_block import embed
from .tva_block import TVA_block_att, TVA_block_att2
from .decoder_block import TVADE_block, TVADE_block2
from .background_encoder import back_encorder
from .cross_att import cross_attn, cross_attn_2
    

class GCEdCastX(nn.Module):

    def __init__(self, Input_len, out_len, num_id, input_size, back_size, hiden_size,  num_layer, vit_layer, dropout, muti_head, num_samp, IF_node,IF_REVIN):
        super(GCEdCastX, self).__init__()

        if IF_node:
            self.inputlen = 2 * Input_len // num_samp
        else:
            self.inputlen = Input_len // num_samp

        self.IF_REVIN = IF_REVIN

        self.Input_len = Input_len

        self.out_len = out_len

        ### embed and encoder
        self.time_embed = nn.Linear(input_size,1)
        self.embed_layer = embed(Input_len,num_id,num_samp,IF_node)
        self.encoder = TVA_block_att(self.inputlen,num_id,num_layer,dropout, muti_head,num_samp)
        self.laynorm = nn.LayerNorm([self.inputlen])
        self.laynorm2 = nn.LayerNorm([self.Input_len])

        self.vis_tran = back_encorder(Input_len, back_size, num_id, hiden_size, dropout, muti_head, vit_layer)
        self.cross_attn1 = cross_attn(Input_len, num_id, dropout, muti_head)
        self.cross_attn2 = cross_attn_2(Input_len, num_id, dropout, muti_head)

        ### decorder
        self.decoder = TVADE_block(self.inputlen, num_id, dropout, muti_head)
        self.output = nn.Conv1d(in_channels = self.inputlen, out_channels=out_len, kernel_size=1)

        ### Adaptive normalization

        self.mean_input = nn.Conv2d(in_channels = input_size, out_channels= 1, kernel_size=1, bias =False)
        self.mean_in4out = nn.Conv1d(in_channels = Input_len, out_channels= out_len, kernel_size=1, bias =False)
        self.mean_output = nn.Conv2d(in_channels = input_size, out_channels= 1, kernel_size=1, bias =False)

        self.stdev_input = nn.Conv2d(in_channels = input_size, out_channels= 1, kernel_size=1, bias =False)
        self.stdev_in4out = nn.Conv1d(in_channels = Input_len, out_channels= out_len, kernel_size=1, bias =False)
        self.stdev_output = nn.Conv2d(in_channels = input_size, out_channels= 1, kernel_size=1, bias =False)
        

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, background_data, target_num, epoch) -> torch.Tensor:
        # Input [B,H,N,1]: B is batch size. N is the number of variables. H is the history length
        # Output [B,L,N,1]: B is batch size. N is the number of variables. L is the future length

        # Timestamp information
        history_time = history_data[:, :, :, 1:]
        future_time = future_data[:,:,:,1:]

        # temporal features
        x = history_data[:, :, :, 0]

        # Background encoder
        back_ground = self.vis_tran(x, background_data)

        # Adaptive normalization
        if self.IF_REVIN:
        #   raw mean and std
            means = x.mean(1, keepdim=True) # [B,1,N]
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)

            # the mean and std of input
            means_input = torch.cat([means.repeat(1, self.Input_len, 1).unsqueeze(-1),history_time],dim=-1) # [B,L,N, 3]
            stdev_input = torch.cat([stdev.repeat(1, self.Input_len, 1).unsqueeze(-1),history_time],dim=-1)

            means_input = self.mean_input(means_input.transpose(-3,-1)).transpose(-3,-1)
            means_input = means_input[:,:,:,0]

            stdev_input = self.stdev_input(stdev_input.transpose(-3,-1)).transpose(-3,-1)
            stdev_input = stdev_input[:,:,:,0]

            # the mean and std of input
            means_out = torch.cat([self.mean_in4out(means_input).unsqueeze(-1),future_time],dim=-1)
            stdev_out = torch.cat([self.stdev_in4out(stdev_input).unsqueeze(-1),future_time],dim=-1)

            means_out = self.mean_output(means_out.transpose(-3,-1)).transpose(-3,-1)
            means_out = means_out[:,:,:,0]

            stdev_out = self.stdev_output(stdev_out.transpose(-3,-1)).transpose(-3,-1)
            stdev_out = stdev_out[:,:,:,0]


            means_input = means_input.mean(1, keepdim=True)
            stdev_input = stdev_input.mean(1, keepdim=True)
            
            means_out = means_out.mean(1, keepdim=True)
            stdev_out = stdev_out.mean(1, keepdim=True)

            if epoch == 1:
                x = (x - means) / stdev
            else:
                x = (x - means_input) / stdev_input
        else:
            means_input = torch.cat([means.repeat(1, self.Input_len, 1).unsqueeze(-1),history_time],dim=-1) # [B,L,N, 3]
            stdev_input = torch.cat([stdev.repeat(1, self.Input_len, 1).unsqueeze(-1),history_time],dim=-1)


        # cross attention
        x = torch.cat([x.unsqueeze(-1), history_time],dim=-1)
        x = self.time_embed(x)[:,:,:,0]
        x = x.transpose(-2,-1)

        x = self.cross_attn1(x, back_ground) + self.cross_attn2(x, back_ground)
        x = self.laynorm2(x)
        x_1, x_2 = self.embed_layer(x)

        ### the encoder of Multi-scale Transformer
        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        x = x_1 + x_2
        x = self.laynorm(x)

        ### the decorder of Multi-scale Transformer
        x = self.decoder(x)
        x = self.output(x.transpose(-2,-1))

        ### De-Normalization
        if self.IF_REVIN:
            if epoch == 1:
                x = x * stdev + means
            else:
                x = x * stdev_out + means_out
        else:
            means_out = future_data[:,:,:,0].mean(1, keepdim=True)
            stdev_out = torch.sqrt(torch.var(future_data[:,:,:,0], dim=1, keepdim=True, unbiased=False) + 1e-5)

        return x[:,:,target_num], means_input, stdev_input, means_out, stdev_out

