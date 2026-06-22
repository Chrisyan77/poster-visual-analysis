import numpy as np
import pandas as pd
import os

def shannon_entropy(hist, total):
    """计算香农熵 H(C)"""
    p = hist / total
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def find_best_split(matrix):
    """
    实现Rigau et al.的信息论分割算法，结合LUB优化思想。
    返回使互信息 I(C,R) 最大的分割方向(H/V)、位置和比例rc。
    """
    H, W = matrix.shape
    total_pixels = H * W
    total_hist = np.bincount(matrix.flatten(), minlength=8)
    H_total = shannon_entropy(total_hist, total_pixels)
    
    best_score = -1
    best_dir = None
    best_rc = 0.5
    
    # --- 检查水平分割 ---
    # LUB优化：使用累加直方图代替每行重新计算
    row_hists = np.apply_along_axis(lambda x: np.bincount(x, minlength=8), 1, matrix)
    cum_hist_h = np.cumsum(row_hists, axis=0)
    
    for i in range(1, H):
        h1 = cum_hist_h[i-1]
        h2 = total_hist - h1
        s1 = np.sum(h1)
        s2 = np.sum(h2)
        
        H1 = shannon_entropy(h1, s1)
        H2 = shannon_entropy(h2, s2)
        
        # I(C,R) = H(C) - [pi_1 * H(C|R1) + pi_2 * H(C|R2)]
        mi = H_total - ( (s1/total_pixels)*H1 + (s2/total_pixels)*H2 )
        
        if mi > best_score:
            best_score = mi
            best_dir = 'H'
            best_rc = i / H
            
    # --- 检查垂直分割 ---
    col_hists = np.apply_along_axis(lambda x: np.bincount(x, minlength=8), 0, matrix)
    cum_hist_v = np.cumsum(col_hists, axis=1)
    
    for j in range(1, W):
        h1 = cum_hist_v[:, j-1]
        h2 = total_hist - h1
        s1 = np.sum(h1)
        s2 = np.sum(h2)
        
        H1 = shannon_entropy(h1, s1)
        H2 = shannon_entropy(h2, s2)
        
        mi = H_total - ( (s1/total_pixels)*H1 + (s2/total_pixels)*H2 )
        
        if mi > best_score:
            best_score = mi
            best_dir = 'V'
            best_rc = j / W
            
    return best_dir, best_rc

def get_second_split(region):
    """对子区域进行第二次分割"""
    return find_best_split(region)

def analyze_poster_composition(matrix_path):
    """提取前两次分割特征"""
    matrix = np.load(matrix_path)
    
    # 第一次分割
    dir1, rc1 = find_best_split(matrix)
    
    # 根据第一次分割获取子区域
    H, W = matrix.shape
    if dir1 == 'H':
        split_pos = int(rc1 * H)
        region2 = matrix[split_pos:, :] # 取下半部分继续分割
    else:
        split_pos = int(rc1 * W)
        region2 = matrix[:, split_pos:] # 取右半部分继续分割
        
    # 第二次分割
    dir2, rc2 = get_second_split(region2)
    
    return dir1, rc1, dir2, rc2, f"{dir1}-{dir2}"

def run_segmentation_analysis(metadata_csv, matrix_folder, output_csv):
    df = pd.read_csv(metadata_csv)
    results = []
    
    for idx, row in df.iterrows():
        img_id = row['image_id']
        mat_path = os.path.join(matrix_folder, f"{img_id}.npy")
        
        if os.path.exists(mat_path):
            dir1, rc1, dir2, rc2, split_type = analyze_poster_composition(mat_path)
            results.append({
                'image_id': img_id,
                'year': row['year'],
                'category': row['category'],
                'style': row['stylistic_tags'],
                'split_type': split_type,
                'rc1': rc1,
                'rc2': rc2
            })
            
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Segmentation analysis saved to {output_csv}")

if __name__ == "__main__":
    run_segmentation_analysis("data/poster_metadata.csv", "data/3bit_matrices/", "data/segmentation_results.csv")
