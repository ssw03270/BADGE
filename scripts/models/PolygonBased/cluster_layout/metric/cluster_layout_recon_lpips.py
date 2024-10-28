import os
import shutil
import tempfile
import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lpips

def separate_images(source_dir, predict_suffix='_predict.png', gt_suffix='_gt.png'):
    """
    주어진 디렉토리에서 예측 이미지와 실제 이미지를 분리하여 임시 디렉토리에 복사합니다.
    
    Args:
        source_dir (str): 원본 이미지들이 저장된 디렉토리 경로.
        predict_suffix (str): 예측 이미지 파일의 접미사.
        gt_suffix (str): 실제 이미지 파일의 접미사.
    
    Returns:
        tuple: (예측 이미지 임시 디렉토리 경로, 실제 이미지 임시 디렉토리 경로, 임시 디렉토리 전체 경로)
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

class ImagePairDataset(Dataset):
    """
    예측 이미지와 실제 이미지의 쌍을 제공하는 데이터셋.
    """
    def __init__(self, predict_dir, gt_dir, transform=None):
        self.predict_dir = predict_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.predict_images = sorted(os.listdir(predict_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
        
        assert len(self.predict_images) == len(self.gt_images), "예측 이미지와 실제 이미지의 수가 일치하지 않습니다."
        
        # 대응되는 이미지 쌍을 보장하기 위해 파일 이름을 비교
        self.image_pairs = []
        predict_set = set(self.predict_images)
        gt_set = set(self.gt_images)
        common_bases = set(os.path.splitext(f.replace('_predict.png', '').replace('_gt.png', '')) for f in self.predict_images).intersection(
            set(os.path.splitext(f.replace('_gt.png', '').replace('_predict.png', '')) for f in self.gt_images))
        
        for base in common_bases:
            predict_file = f"{base}{'_predict.png'}"
            gt_file = f"{base}{'_gt.png'}"
            if predict_file in predict_set and gt_file in gt_set:
                self.image_pairs.append((os.path.join(predict_dir, predict_file),
                                         os.path.join(gt_dir, gt_file)))
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        predict_path, gt_path = self.image_pairs[idx]
        predict_image = Image.open(predict_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')
        
        if self.transform:
            predict_image = self.transform(predict_image)
            gt_image = self.transform(gt_image)
        
        return predict_image, gt_image

def calculate_lpips(source_dir, predict_suffix='_predict.png', gt_suffix='_gt.png', batch_size=50, device=None, resize=True):
    """
    LPIPS 스코어를 계산합니다.
    
    Args:
        source_dir (str): 원본 이미지들이 저장된 디렉토리 경로.
        predict_suffix (str): 예측 이미지 파일의 접미사.
        gt_suffix (str): 실제 이미지 파일의 접미사.
        batch_size (int): 배치 사이즈.
        device (str): 'cuda' 또는 'cpu'. 자동으로 설정할 수도 있습니다.
        resize (bool): 입력 이미지를 Inception 네트워크에 맞게 리사이즈할지 여부.
    
    Returns:
        float: 계산된 LPIPS 스코어 (평균 거리).
    """
    # 이미지 분리
    predict_dir, gt_dir, temp_dir = separate_images(source_dir, predict_suffix, gt_suffix)
    
    try:
        # 장치 설정
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 데이터셋 및 데이터로더 준비
        dataset = ImagePairDataset(predict_dir, gt_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # LPIPS 모델 초기화
        loss_fn = lpips.LPIPS(net='alex').to(device)
        
        total_distance = 0.0
        total_pairs = 0
        
        for batch in tqdm(dataloader, desc='Calculating LPIPS'):
            img1, img2 = batch
            if resize:
                # LPIPS expects images to be at least 256x256
                img1 = torch.nn.functional.interpolate(img1, size=(256, 256), mode='bilinear', align_corners=False)
                img2 = torch.nn.functional.interpolate(img2, size=(256, 256), mode='bilinear', align_corners=False)
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            with torch.no_grad():
                distance = loss_fn(img1, img2)
                total_distance += distance.sum().item()
                total_pairs += distance.size(0)
        
        avg_lpips = total_distance / total_pairs
    finally:
        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)
    
    return avg_lpips

def main():
    parser = argparse.ArgumentParser(description='Calculate LPIPS score between predict and ground truth images in the same directory.')
    parser.add_argument('--source_dir', type=str, default='./visualizations/d_256_cb_512_st_9', help='Path to the directory containing both predict and gt images.')
    parser.add_argument('--predict_suffix', type=str, default='_predict.png', help='Suffix of predict images. Default: _predict.png')
    parser.add_argument('--gt_suffix', type=str, default='_gt.png', help='Suffix of ground truth images. Default: _gt.png')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for LPIPS calculation. Default: 50')
    parser.add_argument('--device', type=str, default=None, help="Computation device: 'cuda' or 'cpu'. If not set, automatically determined.")
    parser.add_argument('--resize', action='store_true', help='Resize images to 256x256 for LPIPS calculation.')
    parser.add_argument('--save_dir', type=str, default='metric_result/lpips', help='Directory to save the LPIPS result.')
    
    args = parser.parse_args()
    
    model_name = os.path.basename(os.path.normpath(args.source_dir))
    args.save_dir = os.path.join(args.save_dir, model_name)
    
    # LPIPS 계산
    lpips_score = calculate_lpips(args.source_dir,
                                  predict_suffix=args.predict_suffix,
                                  gt_suffix=args.gt_suffix,
                                  batch_size=args.batch_size,
                                  device=args.device,
                                  resize=args.resize)
    
    print(f'LPIPS score: {lpips_score:.4f}')
    
    # LPIPS 점수를 result.txt 파일에 저장
    result_path = os.path.join(args.save_dir, 'result.txt')
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        with open(result_path, 'w') as f:
            f.write(f'LPIPS score: {lpips_score:.4f}\n')
        print(f'LPIPS score saved to {result_path}')
    except Exception as e:
        print(f'Error saving LPIPS score to {result_path}: {e}')

if __name__ == '__main__':
    main()
