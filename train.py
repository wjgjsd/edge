import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import EdgeUNet
from dataset import SRDataset

# ---------- [새로 추가된 로스 함수] ----------
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 1. 픽셀별 일치도 (BCE)
        bce_loss = self.bce(inputs, targets)
        
        # 2. 선명한 얇은 선 추출 일치도 (Dice)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.*intersection + self.smooth)/(inputs_flat.sum() + targets_flat.sum() + self.smooth)  
        
        # BCE와 Dice를 합산!
        return bce_loss + dice_loss
# ---------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 사용 중인 학습 기기: {device}")

    # 새롭게 바뀐 구조 모델 장착
    model = EdgeUNet().to(device)
    
    train_dir = "./DIV2K/DIV2K_train_HR"
    
    print("데이터셋을 불러오는 중입니다...")
    dataset = SRDataset(hr_dir=train_dir, patch_size=128, scale_factor=4)
    # 메모리 여유가 있다면 batch_size를 조정 가능
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"총 데이터 개수: {len(dataset)}장, 1에포크당 배치 개수: {len(dataloader)}")

    # [수정] BCE의 단점을 보완한 하이브리드 로스 채택
    criterion = DiceBCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            
            # 하이브리드 로스로 오차 계산
            loss = criterion(outputs, hr_imgs)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 4 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss (BCE+Dice): {loss.item():.6f}")
                
        epoch_loss = running_loss / len(dataloader)
        print(f"===> Epoch {epoch+1} 완료! 이번 에포크 평균 오차: {epoch_loss:.6f}")
        
        torch.save(model.state_dict(), f"edge_unet_epoch_{epoch+1}.pth")
        print(f"모델 체크포인트 edge_unet_epoch_{epoch+1}.pth 저장 완료!\n")

if __name__ == "__main__":
    train()
