import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_comparison(image_path, complexity_path, gt_path, pred_path, output_path):
    # 读取图片
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    complexity = cv2.imread(complexity_path)
    if complexity is None:
        complexity = np.zeros_like(img)
    else:
        complexity = cv2.cvtColor(complexity, cv2.COLOR_BGR2RGB)
    
    gt = cv2.imread(gt_path)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB) if gt is not None else np.zeros_like(img)
    
    pred = cv2.imread(pred_path)
    if pred is None:
        pred = np.zeros_like(img)
    else:
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    
    # 统一尺寸
    h, w = img.shape[:2]
    complexity = cv2.resize(complexity, (w, h))
    gt = cv2.resize(gt, (w, h))
    pred = cv2.resize(pred, (w, h))
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    
    # 在每张图顶部添加标题
    for im, title in zip([img, complexity, gt, pred], 
                         ["Input Image", "Complexity Map", "Ground Truth", "ICTNet-S"]):
        # 白色文字带黑色边框
        cv2.putText(im, title, (20, 45), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(im, title, (20, 45), font, font_scale, (255, 255, 255), thickness)
    
    # 2x2 布局
    top_row = np.hstack([img, complexity])
    bottom_row = np.hstack([gt, pred])
    result = np.vstack([top_row, bottom_row])
    
    # 保存
    plt.figure(figsize=(16, 12))
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    base_name = "frankfurt_000000_000294"
    
    image_path = f"/root/autodl-tmp/datasets/cityscapes/leftImg8bit/val/frankfurt/{base_name}_leftImg8bit.png"
    complexity_path = f"/root/autodl-tmp/ICTNet-main/output/cityscapes/ictednet_small_city_train/complexity_maps_color/{base_name}_complexity_color.png"
    gt_path = f"/root/autodl-tmp/datasets/cityscapes/gtFine/val/frankfurt/{base_name}_gtFine_color.png"
    pred_path = f"/root/autodl-tmp/ICTNet-main/output/cityscapes/ictednet_small_city_train/val_vis_results/{base_name}_gtFine_labelIds.png"
    output_path = f"/root/autodl-tmp/ICTNet-main/output/cityscapes/ictednet_small_city_train/comparison_{base_name}.png"
    
    create_comparison(image_path, complexity_path, gt_path, pred_path, output_path)
    print("Done!")
