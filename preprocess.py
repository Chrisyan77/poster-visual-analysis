import cv2
import numpy as np
import os
import pandas as pd

def to_3bit_median_threshold(image_path):
    """
    将图像转换为3-bit色深 (8 colors)。
    使用每个RGB通道的中位数作为阈值 (对应参考文献SI中的 Fig.S7 方法)。
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 计算每个通道的中位数
    medians = np.median(img_rgb.reshape(-1, 3), axis=0)
    
    # 二值化: >= median 为 1, 否则为 0
    bit_img = (img_rgb >= medians).astype(np.uint8)
    
    # 将RGB 3个bit合并为一个0-7的整数值，方便后续计算直方图
    # R*4 + G*2 + B*1
    combined_matrix = bit_img[:,:,0] * 4 + bit_img[:,:,1] * 2 + bit_img[:,:,2]
    
    return combined_matrix

def process_dataset(metadata_csv, image_folder, output_folder):
    """
    批量处理数据集
    """
    df = pd.read_csv(metadata_csv)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for idx, row in df.iterrows():
        img_id = row['image_id']
        img_path = os.path.join(image_folder, f"{img_id}.jpg")
        
        matrix = to_3bit_median_threshold(img_path)
        if matrix is not None:
            # 保存为.npy格式以供后续快速读取
            np.save(os.path.join(output_folder, f"{img_id}.npy"), matrix)
            
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(df)}")

if __name__ == "__main__":
    process_dataset("data/poster_metadata.csv", "data/images/", "data/3bit_matrices/")
