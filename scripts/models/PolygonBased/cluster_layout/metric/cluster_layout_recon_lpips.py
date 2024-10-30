import os
import shutil
import tempfile
import argparse
from PIL import Image
import torch
import lpips
from tqdm import tqdm

def separate_images(source_dir, predict_suffix='_predict.png', gt_suffix='_gt.png'):
    """
    주어진 디렉토리에서 예측 이미지와 실제 이미지를 분리하여 임시 디렉토리에 복사합니다.
    
    Args:
        source_dir (str): 원본 이미지들이 저장된 디렉토리 경로.
        predict_suffix (str): 예측 이미지 파일의 접미사.
        gt_suffix (str): 실제 이미지 파일의 접미사.
    
    Returns:
        tuple: (예측 이미지 임시 디렉토리 경로, 실제 이미지 임시 디렉토리 경로, 임시 디렉토리 경로)
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

def calculate_lpips(source_dir, predict_suffix='_predict.png', gt_suffix='_gt.png', device=None):
    """
    LPIPS 점수를 계산합니다.
    
    Args:
        source_dir (str): 원본 이미지들이 저장된 디렉토리 경로.
        predict_suffix (str): 예측 이미지 파일의 접미사.
        gt_suffix (str): 실제 이미지 파일의 접미사.
        device (str): 'cuda' 또는 'cpu'. 자동으로 설정할 수도 있습니다.
    
    Returns:
        float: 평균 LPIPS 점수.
    """
    # 이미지 분리
    predict_dir, gt_dir, temp_dir = separate_images(source_dir, predict_suffix, gt_suffix)
    
    try:
        # 장치 설정
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # LPIPS 모델 초기화
        loss_fn = lpips.LPIPS(net='alex').to(device)
        
        # 예측 이미지와 실제 이미지 목록 가져오기
        predict_images = sorted(os.listdir(predict_dir))
        gt_images = sorted(os.listdir(gt_dir))
        
        # 대응되는 이미지 쌍 확인
        assert len(predict_images) == len(gt_images), "예측 이미지와 실제 이미지의 수가 일치하지 않습니다."
        
        total_lpips = 0.0
        count = 0
        
        for pred_name, gt_name in tqdm(zip(predict_images, gt_images), total=len(predict_images), desc='Calculating LPIPS'):
            # 파일 이름 매칭 확인 (필요시 수정 가능)
            if pred_name.replace(predict_suffix, '') != gt_name.replace(gt_suffix, ''):
                print(f"Warning: Mismatched pair {pred_name} and {gt_name}")
                continue
            
            pred_path = os.path.join(predict_dir, pred_name)
            gt_path = os.path.join(gt_dir, gt_name)
            
            # 이미지 로드 및 전처리
            pred_img = Image.open(pred_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
            
            # 텐서로 변환 및 정규화
            pred_tensor = lpips.im2tensor(lpips.load_image(pred_path)).to(device)
            gt_tensor = lpips.im2tensor(lpips.load_image(gt_path)).to(device)
            
            # LPIPS 계산
            with torch.no_grad():
                lpips_score = loss_fn(pred_tensor, gt_tensor).item()
            
            total_lpips += lpips_score
            count += 1
        
        average_lpips = total_lpips / count if count > 0 else float('nan')
    
    finally:
        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)
    
    return average_lpips

def main():
    parser = argparse.ArgumentParser(description='Calculate LPIPS score between predict and ground truth images in the same directory.')
    parser.add_argument('--source_dir', type=str, default='./visualizations/d_256_cb_512_coords_continuous_norm_blk', help='Path to the directory containing both predict and gt images.')
    parser.add_argument('--predict_suffix', type=str, default='_predict.png', help='Suffix of predict images. Default: _predict.png')
    parser.add_argument('--gt_suffix', type=str, default='_gt.png', help='Suffix of ground truth images. Default: _gt.png')
    parser.add_argument('--device', type=str, default=None, help="Computation device: 'cuda' or 'cpu'. If not set, automatically determined.")
    parser.add_argument('--save_dir', type=str, default='metric_result/lpips', help='Directory to save the LPIPS result.')
    
    args = parser.parse_args()
    
    model_name = os.path.basename(os.path.normpath(args.source_dir))
    args.save_dir = os.path.join(args.save_dir, model_name)

    # LPIPS 계산
    lpips_score = calculate_lpips(args.source_dir,
                                  predict_suffix=args.predict_suffix,
                                  gt_suffix=args.gt_suffix,
                                  device=args.device)
    
    print(f'Average LPIPS score: {lpips_score:.4f}')
    
    # LPIPS 점수를 result.txt 파일에 저장
    result_path = os.path.join(args.save_dir, 'result.txt')
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        with open(result_path, 'w') as f:
            f.write(f'Average LPIPS score: {lpips_score:.4f}\n')
        print(f'LPIPS score saved to {result_path}')
    except Exception as e:
        print(f'Error saving LPIPS score to {result_path}: {e}')

if __name__ == '__main__':
    main()
