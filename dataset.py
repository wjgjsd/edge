import os
import glob
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SRDataset(Dataset):
    def __init__(self, hr_dir, patch_size=128, scale_factor=4):
        self.image_paths = glob.glob(os.path.join(hr_dir, "*.png"))
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        crop_transform = transforms.RandomCrop(self.patch_size)
        hr_image = crop_transform(image)
        
        # 1. 원본을 축소하여 LR 이미지 생성 (작은 사이즈)
        lr_image_small = hr_image.resize(
            (self.patch_size // self.scale_factor, self.patch_size // self.scale_factor), 
            Image.BICUBIC
        )
        
        # 2. 작은 LR 이미지를 원본 크기(patch_size)로 다시 Bicubic 업샘플링
        lr_image_bicubic = lr_image_small.resize(
            (self.patch_size, self.patch_size), 
            Image.BICUBIC
        )
        
        hr_np = np.array(hr_image)
        lr_np_bicubic = np.array(lr_image_bicubic)
        
        # 3. 동일한 크기로 맞춰진 상태에서 윤곽선 추출 (Canny)
        hr_edge = cv2.Canny(hr_np, 30, 200)
        lr_edge = cv2.Canny(lr_np_bicubic, 30, 200)
        
        # 4. 차원 추가 및 4채널(RGB + Edge) 결합
        hr_edge_1c = np.expand_dims(hr_edge, axis=-1)
        lr_edge_1c = np.expand_dims(lr_edge, axis=-1)
        
        # LR 모델 입력: 컬러 이미지(3채널) + 거친 윤곽선(1채널) 결합 -> 총 4채널
        lr_combined = np.concatenate([lr_np_bicubic, lr_edge_1c], axis=-1)
        
        # ToTensor()가 (H,W,C) 형태의 numpy 배열을 (C,H,W) 텐서 [0.0 ~ 1.0] 로 변환해줌
        hr_tensor = self.to_tensor(hr_edge_1c)
        lr_tensor = self.to_tensor(lr_combined)
        
        return lr_tensor, hr_tensor

# 데이터로더 동작 테스트
if __name__ == "__main__":
    train_hr_dir = "./DIV2K/DIV2K_train_HR" 
    my_dataset = SRDataset(hr_dir=train_hr_dir, patch_size=256, scale_factor=4)
    print(f"불러온 이미지 개수: {len(my_dataset)}장")
    
    my_loader = DataLoader(dataset=my_dataset, batch_size=4, shuffle=True)
    
    for lr_batch, hr_batch in my_loader:
        print(f"입력(LR) 4채널 결합 텐서 크기: {lr_batch.shape}")
        print(f"정답(HR) 1채널 엣지 텐서 크기: {hr_batch.shape}")
        break 
