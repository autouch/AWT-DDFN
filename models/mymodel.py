import torch
import torch.nn as nn
from models.LearnableMorletWavelet import LearnableMorlet1D
from ptflops import get_model_complexity_info

class TDB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[2, 3, 5]):
        super().__init__()
        self.conv=nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels,in_channels,5,stride=1,groups=in_channels,padding=2),nn.Conv1d(in_channels,in_channels,2,2,groups=in_channels),
            nn.ReLU(),nn.Conv1d(in_channels, out_channels, kernel_size,  
                dilation=dilation,groups=in_channels,padding=(kernel_size - 1) * dilation//2))
                for dilation in dilations])
    def forward(self, x):
        conv_result=[]
        for layer in self.conv:
            temp = layer(x)
            conv_result.append(temp)
        output=torch.stack(conv_result,dim=0).sum(dim=0)
        return output
    
    
class SDB(nn.Module):
    def __init__(self,in_channels,out_channels,a=[16,4,1]):
        super().__init__()
        self.conv=nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels, a[i], 1,1  
                ),nn.Conv1d( a[i], a[i], 2,2,groups=a[i]),nn.ReLU(),nn.Conv1d(a[i],out_channels,  1
                ))
                for i in range(len(a))])
    def forward(self, x):
        conv_result=[]
        for layer in self.conv:
            temp = layer(x)
            conv_result.append(temp)
        output=torch.stack(conv_result,dim=0).sum(dim=0)
        return output

class DDFB(nn.Module):
    def __init__(
        self,inchannel,outchannel
    ):
        super().__init__()
        self.TDB=TDB(inchannel,outchannel,3)
        self.SDB=SDB(inchannel,outchannel)
        self.bn=nn.BatchNorm1d(outchannel)
    def forward(self,x):
        
        x1=self.TDB(x)
        x2=self.SDB(x)
        x=x1+x2
        x=self.bn(x)
        return x

class AWT_DDFN(nn.Module):
    def __init__(
        self,numlayer=2,channel=64,classes=11
    ):
        super().__init__()
        self.morlet_conv = LearnableMorlet1D(1,channel,127,sample_rate=64000,use_complex=True)
        self.maxpool=nn.MaxPool1d(5,4,2)
        self.DDFB=[]
        self.bn=nn.BatchNorm1d(channel)
        for num in range(numlayer):
            self.DDFB.append(DDFB(channel,channel))
        self.DDFB=nn.ModuleList(self.DDFB)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(in_features = channel, out_features = classes,bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.morlet_conv(x)
        x=self.maxpool(x)
        x=self.bn(x)
        for layer in self.DDFB:
            x=layer(x)
        self.out=x
        output = self.gap(x).permute(0,2,1)
        tsne=output.squeeze(1)
        output=self.fc2(tsne)
        return output

if __name__ == "__main__":
    device='cuda:1'
    L=2048
    input_shape=()
    model =AWT_DDFN().to(device)
    macs, params = get_model_complexity_info(model, (1, L), as_strings=False, print_per_layer_stat=False)
    inputshape=torch.rand(32,1,2048).to(device)
    output,_=model(inputshape)
    print(f"FLOPs: {macs}")
    print(f"参数量: {params}")
    print(output.shape,inputshape.shape)