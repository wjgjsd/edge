import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import math
import glob
import cv2
import numpy as np

from model import EdgeUNet

def calculate_psnr(img1, img2):
    mse = nn.MSELoss()(img1, img2).item()
    if mse == 0:
        return float('inf') 
    return 10 * math.log10(1.0 / mse)

def calculate_metrics(pred, target, threshold=0.5):
    # 이진화 처리 (0.5 기준)
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    intersection = (pred_bin * target_bin).sum()
    sum_pred = pred_bin.sum()
    sum_target = target_bin.sum()
    
    # Dice Coefficient (F1-score)
    dice = (2. * intersection) / (sum_pred + sum_target + 1e-8)
    
    # IoU (Jaccard Index)
    iou = intersection / (sum_pred + sum_target - intersection + 1e-8)
    
    return dice.item(), iou.item()

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"테스트 환경: {device}")

    weight_files = glob.glob("edge_unet_epoch_*.pth")
    if len(weight_files) == 0:
        print("가중치 파일을 찾을 수 없어요.")
        return
        
    weight_path = sorted(weight_files)[-1]
    model = EdgeUNet().to(device)
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"성공적으로 빙의 완료 (가중치: {weight_path})")
    except RuntimeError:
        print("\n[알림] 모델 아키텍처가 4채널로 완전히 변경되어 과거 가중치와 호환되지 않습니다!")
        print("터미널에서 'python train.py'를 실행하여 모델을 새롭게 학습시켜주세요.\n")
        return

    model.eval()

    hr_img_path = "./DIV2K/DIV2K_valid_HR/0801.png"
    lr_img_path = "./DIV2K/DIV2K_valid_LR_bicubic/X4/0801x4.png" 
    
    if not (os.path.exists(hr_img_path) and os.path.exists(lr_img_path)):
        print("이미지 경로를 확인해주세요.")
        return
        
    original_hr = Image.open(hr_img_path).convert("RGB")
    original_lr = Image.open(lr_img_path).convert("RGB") 
    
    # 1. Bicubic 확대
    lr_image_bicubic = original_lr.resize(original_hr.size, Image.BICUBIC)
    
    hr_np = np.array(original_hr)
    lr_np_bicubic = np.array(lr_image_bicubic)
    
    # 2. 동일선상에서 엣지 추출
    hr_edge = cv2.Canny(hr_np, 30, 200)
    lr_edge = cv2.Canny(lr_np_bicubic, 30, 200)
    
    # 3. 차원 변경 및 3+1 채널 결합
    hr_edge_1c = np.expand_dims(hr_edge, axis=-1)
    lr_edge_1c = np.expand_dims(lr_edge, axis=-1)
    
    lr_combined = np.concatenate([lr_np_bicubic, lr_edge_1c], axis=-1)
    
    to_tensor = transforms.ToTensor()
    # 4채널 텐서
    lr_tensor = to_tensor(lr_combined).unsqueeze(0).to(device) 
    # 1채널 정답 텐서
    hr_tensor = to_tensor(hr_edge_1c).unsqueeze(0).to(device)

    # 4. 모델 구동 (내부에서 알아서 0.5 임계값 계산하여 0과 1로 나뉨)
    with torch.no_grad():
        output_sr_tensor = model(lr_tensor)
        
    output_sr_tensor = output_sr_tensor.clamp(0.0, 1.0)
    
    # [수정] 4채널 전부 비교하면 안 되므로, 처음에 추출한 단일 lr_edge만 테스트에 사용
    lr_edge_tensor_only = to_tensor(lr_edge_1c).unsqueeze(0).to(device)
    
    psnr_nearest = calculate_psnr(lr_edge_tensor_only, hr_tensor)
    psnr_unet = calculate_psnr(output_sr_tensor, hr_tensor)
    
    dice_unet, iou_unet = calculate_metrics(output_sr_tensor, hr_tensor)
    dice_lr, iou_lr = calculate_metrics(lr_edge_tensor_only, hr_tensor)
    
    print("\n" + "=" * 60)
    print(f"👉 [Baseline]  PSNR: {psnr_nearest:.2f} dB | F1: {dice_lr:.4f} | IoU: {iou_lr:.4f}")
    print(f"👉 [UNet Deep] PSNR: {psnr_unet:.2f} dB | F1: {dice_unet:.4f} | IoU: {iou_unet:.4f}")
    print("=" * 60)
    
    to_pil = transforms.ToPILImage()
    
    hr_edge_img = to_pil(hr_tensor.squeeze(0).cpu())
    lr_edge_img = to_pil(lr_edge_tensor_only.squeeze(0).cpu())
    sr_edge_img = to_pil(output_sr_tensor.squeeze(0).cpu())
    
    hr_edge_img.save("result_1_Original_HR_Edge.png")
    lr_edge_img.save("result_2_Pixelated_LR_Edge.png")
    sr_edge_img.save("result_3_UNet_Advanced_Restored_Edge.png")
    print("\n결과 이미지 3장이 성공적으로 폴더에 저장되었습니다!")

if __name__ == "__main__":
    test()
