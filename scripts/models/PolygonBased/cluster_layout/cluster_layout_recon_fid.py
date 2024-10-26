import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import shutil
import tempfile
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths
import torch
from tqdm import tqdm

def separate_images(source_dir, predict_suffix='_predict.png', gt_suffix='_gt.png'):
    """
    주어진 디렉토리에서 예측 이미지와 실제 이미지를 분리하여 임시 디렉토리에 복사합니다.
    
    Args:
        source_dir (str): 원본 이미지들이 저장된 디렉토리 경로.
        predict_suffix (str): 예측 이미지 파일의 접미사.
        gt_suffix (str): 실제 이미지 파일의 접미사.
    
    Returns:
        tuple: (예측 이미지 임시 디렉토리 경로, 실제 이미지 임시 디렉토리 경로)
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

def calculate_fid(source_dir, predict_suffix='_predict.png', gt_suffix='_gt.png', batch_size=50, device=None, dims=2048):
    """
    FID 스코어를 계산합니다.
    
    Args:
        source_dir (str): 원본 이미지들이 저장된 디렉토리 경로.
        predict_suffix (str): 예측 이미지 파일의 접미사.
        gt_suffix (str): 실제 이미지 파일의 접미사.
        batch_size (int): 배치 사이즈.
        device (str): 'cuda' 또는 'cpu'. 자동으로 설정할 수도 있습니다.
        dims (int): Inception 네트워크의 출력 차원.
    
    Returns:
        float: 계산된 FID 스코어.
    """
    # 이미지 분리
    predict_dir, gt_dir, temp_dir = separate_images(source_dir, predict_suffix, gt_suffix)
    
    try:
        # 장치 설정
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # FID 계산
        fid_value = calculate_fid_given_paths([gt_dir, predict_dir],
                                             batch_size=batch_size,
                                             device=device,
                                             dims=dims)
    finally:
        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)
    
    return fid_value

def main():
    parser = argparse.ArgumentParser(description='Calculate FID score between predict and ground truth images in the same directory.')
    parser.add_argument('--source_dir', type=str, default='./visualizations/d_256_cb_512_st_9', help='Path to the directory containing both predict and gt images.')
    parser.add_argument('--predict_suffix', type=str, default='_predict.png', help='Suffix of predict images. Default: _predict.png')
    parser.add_argument('--gt_suffix', type=str, default='_gt.png', help='Suffix of ground truth images. Default: _gt.png')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for FID calculation. Default: 50')
    parser.add_argument('--device', type=str, default=None, help="Computation device: 'cuda' or 'cpu'. If not set, automatically determined.")
    parser.add_argument('--dims', type=int, default=2048, help='Dimensionality of Inception features to use. Default: 2048')
    
    args = parser.parse_args()
    
    # FID 계산
    fid = calculate_fid(args.source_dir,
                        predict_suffix=args.predict_suffix,
                        gt_suffix=args.gt_suffix,
                        batch_size=args.batch_size,
                        device=args.device,
                        dims=args.dims)
    
    print(f'FID score: {fid:.4f}')
    
    # FID 점수를 result.txt 파일에 저장
    result_path = os.path.join(args.source_dir, 'result.txt')
    try:
        with open(result_path, 'w') as f:
            f.write(f'FID score: {fid:.4f}\n')
        print(f'FID score saved to {result_path}')
    except Exception as e:
        print(f'Error saving FID score to {result_path}: {e}')

if __name__ == '__main__':
    main()