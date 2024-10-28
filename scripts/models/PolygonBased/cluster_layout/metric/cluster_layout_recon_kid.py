import os
import shutil
import tempfile
import argparse
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm  # Import tqdm for progress bars

def separate_images(source_dir, predict_suffix='_predict.png', gt_suffix='_gt.png'):
    """
    주어진 디렉토리에서 예측 이미지와 실제 이미지를 분리하여 임시 디렉토리에 복사합니다.
    
    Args:
        source_dir (str): 원본 이미지들이 저장된 디렉토리 경로.
        predict_suffix (str): 예측 이미지 파일의 접미사.
        gt_suffix (str): 실제 이미지 파일의 접미사.
    
    Returns:
        tuple: (예측 이미지 임시 디렉토리 경로, 실제 이미지 임시 디렉토리 경로, temp_dir)
    """
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    predict_dir = os.path.join(temp_dir, 'predictions')
    gt_dir = os.path.join(temp_dir, 'ground_truth')
    os.makedirs(predict_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    # 파일 분류 및 복사
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if os.path.isfile(file_path):
            if filename.endswith(predict_suffix):
                shutil.copy(file_path, predict_dir)
            elif filename.endswith(gt_suffix):
                shutil.copy(file_path, gt_dir)
    
    return predict_dir, gt_dir, temp_dir

class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.filenames = sorted([
            os.path.join(folder, fname) 
            for fname in os.listdir(folder) 
            if os.path.isfile(os.path.join(folder, fname))
        ])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def calculate_kid_score(source_dir, predict_suffix='_predict.png', gt_suffix='_gt.png', batch_size=50, device=None, dims=2048):
    """
    KID 스코어를 계산합니다.
    
    Args:
        source_dir (str): 원본 이미지들이 저장된 디렉토리 경로.
        predict_suffix (str): 예측 이미지 파일의 접미사.
        gt_suffix (str): 실제 이미지 파일의 접미사.
        batch_size (int): 배치 사이즈.
        device (str): 'cuda' 또는 'cpu'. 자동으로 설정할 수도 있습니다.
        dims (int): Inception 네트워크의 출력 차원.
    
    Returns:
        tuple: (KID 평균값, KID 분산값)
    """
    # 이미지 분리
    predict_dir, gt_dir, temp_dir = separate_images(source_dir, predict_suffix, gt_suffix)
    
    try:
        # 장치 설정
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 데이터 전처리
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        # 데이터셋 및 데이터로더
        gt_dataset = ImageFolderDataset(gt_dir, transform=transform)
        pred_dataset = ImageFolderDataset(predict_dir, transform=transform)
        
        gt_loader = DataLoader(gt_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # KID 계산기 초기화 (num_subsets 제거)
        kid = KernelInceptionDistance(subset_size=1000, normalize=True).to(device)
        
        # Ground Truth 이미지 추가 with tqdm progress bar
        print("Processing Ground Truth images...")
        for batch in tqdm(gt_loader, desc="Ground Truth Batches", unit="batch"):
            batch = batch.to(device)
            kid.update(batch, real=True)
        
        # 예측 이미지 추가 with tqdm progress bar
        print("Processing Predicted images...")
        for batch in tqdm(pred_loader, desc="Predicted Batches", unit="batch"):
            batch = batch.to(device)
            kid.update(batch, real=False)
        
        # KID 계산 (반환값 언팩)
        kid_mean, kid_var = kid.compute()
        kid_score = kid_mean.item()
        kid_variance = kid_var.item()
    
    finally:
        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)
    
    return kid_score, kid_variance

def main():
    parser = argparse.ArgumentParser(description='Calculate KID score between predict and ground truth images in the same directory.')
    parser.add_argument('--source_dir', type=str, default='./visualizations/d_256_cb_512_st_9', help='Path to the directory containing both predict and gt images.')
    parser.add_argument('--predict_suffix', type=str, default='_predict.png', help='Suffix of predict images. Default: _predict.png')
    parser.add_argument('--gt_suffix', type=str, default='_gt.png', help='Suffix of ground truth images. Default: _gt.png')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for KID calculation. Default: 50')
    parser.add_argument('--device', type=str, default=None, help="Computation device: 'cuda' or 'cpu'. If not set, automatically determined.")
    parser.add_argument('--dims', type=int, default=2048, help='Dimensionality of Inception features to use. Default: 2048')
    parser.add_argument('--save_dir', type=str, default='metric_result/kid', help='Directory to save the KID score result.')
    
    args = parser.parse_args()
    
    model_name = os.path.basename(os.path.normpath(args.source_dir))
    args.save_dir = os.path.join(args.save_dir, model_name)
    
    # KID 계산
    kid_mean, kid_var = calculate_kid_score(
        args.source_dir,
        predict_suffix=args.predict_suffix,
        gt_suffix=args.gt_suffix,
        batch_size=args.batch_size,
        device=args.device,
        dims=args.dims
    )
    
    print(f'KID score (mean): {kid_mean:.4f}')
    print(f'KID score (variance): {kid_var:.4f}')
    
    # KID 점수를 result.txt 파일에 저장
    result_path = os.path.join(args.save_dir, 'result.txt')
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        with open(result_path, 'w') as f:
            f.write(f'KID score (mean): {kid_mean:.4f}\n')
            f.write(f'KID score (variance): {kid_var:.4f}\n')
        print(f'KID scores saved to {result_path}')
    except Exception as e:
        print(f'Error saving KID scores to {result_path}: {e}')

if __name__ == '__main__':
    main()
