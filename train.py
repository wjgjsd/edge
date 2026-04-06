import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from model import EdgeUNet
from dataset import SRDataset

# ---------- [새롭게 정의된 로스 함수들] ----------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(9)])   # relu2_2
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs, targets):
        # 1채널 엣지를 3채널로 복사 (VGG 입력용)
        inputs_3c = inputs.repeat(1, 3, 1, 1)
        targets_3c = targets.repeat(1, 3, 1, 1)
        
        h_relu1_2 = self.slice1(inputs_3c)
        h_relu2_2 = self.slice2(inputs_3c)
        
        h_target_relu1_2 = self.slice1(targets_3c)
        h_target_relu2_2 = self.slice2(targets_3c)
        
        loss = nn.MSELoss()(h_relu1_2, h_target_relu1_2) + \
               nn.MSELoss()(h_relu2_2, h_target_relu2_2)
        return loss

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()
        self.smooth = smooth

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.*intersection + self.smooth)/(inputs_flat.sum() + targets_flat.sum() + self.smooth)  
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

    # [수정] 복합 로스 함수 구성
    criterion_dice = DiceBCELoss()
    criterion_perceptual = PerceptualLoss().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # [신규] 학습률 스케줄러 추가 (성능 향상 포인트)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            
            # [수정] 하이브리드 + 퍼셉추얼 로스로 학습 정교화
            loss_dice = criterion_dice(outputs, hr_imgs)
            loss_perceptual = criterion_perceptual(outputs, hr_imgs)
            
            loss = loss_dice + 0.1 * loss_perceptual
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 4 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss (BCE+Dice): {loss.item():.6f}")
                
        epoch_loss = running_loss / len(dataloader)
        scheduler.step(epoch_loss) # 스케줄러 업데이트
        print(f"===> Epoch {epoch+1} 완료! 이번 에포크 평균 오차: {epoch_loss:.6f} / LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        torch.save(model.state_dict(), f"edge_unet_epoch_{epoch+1}.pth")
        print(f"모델 체크포인트 edge_unet_epoch_{epoch+1}.pth 저장 완료!\n")

if __name__ == "__main__":
    train()
