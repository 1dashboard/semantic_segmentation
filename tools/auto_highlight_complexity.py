import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_high_complexity_regions(complexity_img, num_boxes=2, min_size=5000):
    """
    从复杂度图中找到高复杂度区域（使用 OpenCV 实现）
    """
    # 转为灰度图
    if len(complexity_img.shape) == 3:
        gray = cv2.cvtColor(complexity_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = complexity_img
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # 二值化：取最高 15% 的区域
    threshold = np.percentile(blurred, 85)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    
    # 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 提取矩形框
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > min_size:
            boxes.append((x, y, w, h, area))
    
    # 按面积排序，取前 num_boxes 个
    boxes.sort(key=lambda r: r[4], reverse=True)
    
    return [(x, y, w, h) for x, y, w, h, _ in boxes[:num_boxes]]

def add_highlight_box(image, x, y, w, h, color, linewidth=3):
    """在图像上添加矩形框"""
    img_copy = image.copy()
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, linewidth)
    return img_copy

def create_comparison_highlight(image_path, complexity_path, gt_path, pred_path, output_path):
    # 读取图片
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    complexity_color = cv2.imread(complexity_path)
    if complexity_color is None:
        complexity_color = np.zeros_like(img)
    else:
        complexity_color = cv2.cvtColor(complexity_color, cv2.COLOR_BGR2RGB)
    
    # 自动检测高复杂度区域
    boxes = find_high_complexity_regions(complexity_color, num_boxes=2)
    print(f"检测到 {len(boxes)} 个高复杂度区域: {boxes}")
    
    # 读取 GT 和预测
    gt = cv2.imread(gt_path)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB) if gt is not None else np.zeros_like(img)
    
    pred = cv2.imread(pred_path)
    if pred is None:
        pred = np.zeros_like(img)
    else:
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    
    # 统一尺寸
    h, w = img.shape[:2]
    complexity_color = cv2.resize(complexity_color, (w, h))
    gt = cv2.resize(gt, (w, h))
    pred = cv2.resize(pred, (w, h))
    
    # 添加矩形框
    for (x, y, bw, bh) in boxes:
        img = add_highlight_box(img, x, y, bw, bh, (255, 165, 0))   # 橙色
        complexity_color = add_highlight_box(complexity_color, x, y, bw, bh, (255, 165, 0))
        gt = add_highlight_box(gt, x, y, bw, bh, (255, 0, 0))       # 红色
        pred = add_highlight_box(pred, x, y, bw, bh, (255, 0, 0))
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    for im, title in zip([img, complexity_color, gt, pred], 
                         ["Input Image", "Complexity Map", "Ground Truth", "ICTNet-S"]):
        cv2.putText(im, title, (20, 45), font, 1.2, (0, 0, 0), 3)
        cv2.putText(im, title, (20, 45), font, 1.2, (255, 255, 255), 2)
    
    # 2x2 布局
    top_row = np.hstack([img, complexity_color])
    bottom_row = np.hstack([gt, pred])
    result = np.vstack([top_row, bottom_row])
    
    plt.figure(figsize=(16, 12))
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存到: {output_path}")

if __name__ == "__main__":
    base_name = "frankfurt_000000_000294"
    
    image_path = f"/root/autodl-tmp/datasets/cityscapes/leftImg8bit/val/frankfurt/{base_name}_leftImg8bit.png"
    complexity_path = f"/root/autodl-tmp/ICTNet-main/output/cityscapes/ictednet_small_city_train/complexity_maps_color/{base_name}_complexity_color.png"
    gt_path = f"/root/autodl-tmp/datasets/cityscapes/gtFine/val/frankfurt/{base_name}_gtFine_color.png"
    pred_path = f"/root/autodl-tmp/ICTNet-main/output/cityscapes/ictednet_small_city_train/val_vis_results/{base_name}_gtFine_labelIds.png"
    output_path = f"/root/autodl-tmp/ICTNet-main/output/cityscapes/ictednet_small_city_train/comparison_auto_highlight.png"
    
    create_comparison_highlight(image_path, complexity_path, gt_path, pred_path, output_path)
    print("Done!")
