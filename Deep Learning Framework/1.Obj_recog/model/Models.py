'''
Spatiotemporal Touch Perception network
==============
**Author**: `zhibin Li`__
'''
#python ./model/Models.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from thop import profile, clever_format






# def gn(c, g=4):
#     g = min(g, c)
#     return nn.GroupNorm(g, c)

def gn(c, g=4):
    while g > 1 and c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, spatial_stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, stride=(1, spatial_stride, spatial_stride))
        self.gn1   = gn(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.gn2   = gn(out_ch)
        self.skip  = (in_ch == out_ch and spatial_stride == 1)
        if not self.skip:
            self.proj = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=(1, spatial_stride, spatial_stride)),
                gn(out_ch)
            )
    def forward(self, x):
        y = self.conv1(x); y = self.gn1(y); y = self.act(y)
        y = self.conv2(y); y = self.gn2(y)
        s = x if hasattr(self, 'proj') is False else self.proj(x)
        return self.act(y + (s if hasattr(self, 'proj') else x))

class TemporalHead(nn.Module):
    def __init__(self, embed_dim=64, hid=96):
        super().__init__()
        self.pre_ln = nn.LayerNorm(embed_dim)
        self.tcn = nn.Sequential(
            nn.Conv1d(embed_dim, hid, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(hid, hid, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(hid, embed_dim, kernel_size=3, padding=4, dilation=4),
        )
        self.attn_score = nn.Linear(embed_dim, 1)
    def forward(self, x):          # (B, T, C)
        x = self.pre_ln(x)
        y = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        x = x + y
        w = torch.softmax(self.attn_score(x), dim=1)   # (B,T,1)
        return (x * w).sum(dim=1)                      # (B,C)

class CNN_3D(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 4, 3, padding=1),
            gn(4),
            nn.ReLU(inplace=True),
        )
        # 两次空间下采样：48->24->12，保留更多细节
        self.layer2 = ResBlock3D(4, 8, spatial_stride=2)
        self.layer3 = ResBlock3D(8, 16, spatial_stride=2)

        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))   # (B,16,T,1,1)
        self.embed = nn.Conv3d(16, embed_dim, 1)         # (B,C,T,1,1)

        self.temporal = TemporalHead(embed_dim=embed_dim, hid=embed_dim*3//2)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_feature = False):                  # x: (B, T, 48, 48)
        x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
        x = x.unsqueeze(1)                 # (B,1,T,H,W)
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.embed(x)                  # (B,C,T,1,1)
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B,T,C)
        x = self.temporal(x)               # (B,C)

        if return_feature:
            return x  # 提取特征

        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        return x




# ------------------ MicroLite ResBlock ------------------
class ResBlock3D_lite(nn.Module):
    def __init__(self, in_ch, out_ch, spatial_stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1,
                              stride=(1, spatial_stride, spatial_stride), bias=False)
        self.norm = gn(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.skip = (in_ch == out_ch and spatial_stride == 1)
        if not self.skip:
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1,
                                  stride=(1, spatial_stride, spatial_stride), bias=False)

    def forward(self, x):
        y = self.act(self.norm(self.conv(x)))
        s = x if self.skip else self.proj(x)
        return self.act(y + s)

# ------------------ MicroLite Temporal Head ------------------
class TemporalHeadlite(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.pre_ln = nn.LayerNorm(embed_dim)
        self.dw = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.pw = nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=1)  # 减小 hidden
        self.out = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=1)
        self.attn_score = nn.Linear(embed_dim, 1)

    def forward(self, x):  # (B,T,C)
        x = self.pre_ln(x)
        y = self.dw(x.transpose(1, 2))
        y = F.gelu(self.pw(y))
        y = self.out(y).transpose(1, 2)
        x = x + y
        w = torch.softmax(self.attn_score(x), dim=1)  # (B,T,1)
        return (x * w).sum(dim=1)  # (B,C)

# ------------------ MicroLite CNN_3D ------------------
class CNN_3D_Lite(nn.Module):
    def __init__(self, num_classes=10, embed_dim=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1, bias=False),
            gn(4),
            nn.ReLU(inplace=True),
        )
        self.layer2 = ResBlock3D_lite(4, 6, spatial_stride=2)   # 48→24
        self.layer3 = ResBlock3D_lite(6, 6, spatial_stride=2)   # 24→12

        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # (B,6,T,1,1)
        self.embed = nn.Conv3d(6, embed_dim, 1, bias=False)

        self.temporal = TemporalHeadlite(embed_dim=embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_feature = False):  # x: (B,T,H,W)
        x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
        x = x.unsqueeze(1)  # (B,1,T,H,W)
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.embed(x)  # (B,C,T,1,1)
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B,T,C)
        x = self.temporal(x)  # (B,C)

        if return_feature:
            return x  # 提取特征

        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        return x






if __name__ == '__main__':
    net = CNN_3D_Lite()

    print(net)
    #
    input1 = torch.rand(10, 1 , 48, 48)  #
    # #
    out1 = net(input1)
    print(out1.size())

    flops, params = profile(net, inputs=(input1,))
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
