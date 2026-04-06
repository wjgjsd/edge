import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU)를 2번 반복하는 U-Net의 기본 블록"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class EdgeUNet(nn.Module):
    def __init__(self):
        super(EdgeUNet, self).__init__()
        
        # [변경] 초기 채널을 64로 확장하고 더 깊은 층(512까지) 추가
        self.inc = DoubleConv(4, 64)
        
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256) # cat(256 + 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128) # cat(128 + 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)   # cat(64 + 64)
        
        # 최종 출력은 항상 1채널 (흑백 엣지)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x_up1 = self.up1(x4)
        # [수정] 스킵 커넥션과 크기 맞추기 (홀수 사이즈 대응)
        diffY = x3.size()[2] - x_up1.size()[2]
        diffX = x3.size()[3] - x_up1.size()[3]
        x_up1 = F.pad(x_up1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x_up1_cat = torch.cat([x3, x_up1], dim=1) 
        x_up1_out = self.conv_up1(x_up1_cat)
        
        x_up2 = self.up2(x_up1_out)
        diffY = x2.size()[2] - x_up2.size()[2]
        diffX = x2.size()[3] - x_up2.size()[3]
        x_up2 = F.pad(x_up2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x_up2_cat = torch.cat([x2, x_up2], dim=1)
        x_up2_out = self.conv_up2(x_up2_cat)

        x_up3 = self.up3(x_up2_out)
        diffY = x1.size()[2] - x_up3.size()[2]
        diffX = x1.size()[3] - x_up3.size()[3]
        x_up3 = F.pad(x_up3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x_up3_cat = torch.cat([x1, x_up3], dim=1)
        x_up3_out = self.conv_up3(x_up3_cat)
        
        residual = self.outc(x_up3_out)
        
        # [변경] 잔차 연결(+x)를 완전히 지우고 백지에서 날카롭게 선을 그리도록 유도!
        out = self.sigmoid(residual)
        
        # 테스트 모드에서는 모델 스스로 0.5(50%) 이상만 선으로 처리
        if not self.training:
            out = (out >= 0.5).float()
            
        return out

if __name__ == "__main__":
    model = EdgeUNet()
    # 확인용 4채널 텐서
    dummy_input = torch.randn(1, 4, 128, 128) 
    output = model(dummy_input)
    print(f"입력 크기: {dummy_input.shape}")
    print(f"출력 크기: {output.shape}") 
