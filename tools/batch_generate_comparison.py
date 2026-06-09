import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_high_complexity_regions(complexity_img, num_boxes=2, min_size=5000):
    """从复杂度图检测高复杂度区域"""
    if len(complexity_img.shape) == 3:
        gray = cv2.cvtColor(complexity_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = complexity_img
    
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    threshold = np.percentile(blurred, 85)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > min_size:
            boxes.append((x, y, w, h, area))
    
    boxes.sort(key=lambda r: r[4], reverse=True)
    return [(x, y, w, h) for x, y, w, h, _ in boxes[:num_boxes]]

def add_highlight_box(image, x, y, w, h, color, linewidth=3):
    img_copy = image.copy()
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, linewidth)
    return img_copy

def create_comparison(image_path, complexity_path, gt_path, pred_path, output_path, scene_name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  跳过: {scene_name} - 原图不存在")
        return False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    complexity = cv2.imread(complexity_path)
    if complexity is None:
        print(f"  跳过: {scene_name} - 复杂度图不存在")
        return False
    complexity = cv2.cvtColor(complexity, cv2.COLOR_BGR2RGB)
    
    # 检测高复杂度区域
    boxes = find_high_complexity_regions(complexity, num_boxes=2)
    
    gt = cv2.imread(gt_path)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB) if gt is not None else np.zeros_like(img)
    
    pred = cv2.imread(pred_path)
    if pred is None:
        pred = np.zeros_like(img)
    else:
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    complexity = cv2.resize(complexity, (w, h))
    gt = cv2.resize(gt, (w, h))
    pred = cv2.resize(pred, (w, h))
    
    # 添加矩形框
    for (x, y, bw, bh) in boxes:
        img = add_highlight_box(img, x, y, bw, bh, (255, 165, 0))
        complexity = add_highlight_box(complexity, x, y, bw, bh, (255, 165, 0))
        gt = add_highlight_box(gt, x, y, bw, bh, (255, 0, 0))
        pred = add_highlight_box(pred, x, y, bw, bh, (255, 0, 0))
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    for im, title in zip([img, complexity, gt, pred], 
                         ["Input Image", "Complexity Map", "Ground Truth", "ICTNet-S"]):
        cv2.putText(im, title, (20, 45), font, 1.2, (0, 0, 0), 3)
        cv2.putText(im, title, (20, 45), font, 1.2, (255, 255, 255), 2)
    
    # 2x2 布局
    top_row = np.hstack([img, complexity])
    bottom_row = np.hstack([gt, pred])
    result = np.vstack([top_row, bottom_row])
    
    plt.figure(figsize=(16, 12))
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {scene_name} - 检测到 {len(boxes)} 个区域")
    return True

# ============ 配置 ============
# 验证集所有城市
cities = {
    'frankfurt': '/root/autodl-tmp/datasets/cityscapes/leftImg8bit/val/frankfurt',
    'lindau': '/root/autodl-tmp/datasets/cityscapes/leftImg8bit/val/lindau',
    'munster': '/root/autodl-tmp/datasets/cityscapes/leftImg8bit/val/munster'
}

# 输出目录
output_base = 'output/cityscapes/ictednet_small_city_train/comparison_all'
os.makedirs(output_base, exist_ok=True)

# 每个城市选择的前 N 张图片
images_per_city = 5

print("=" * 50)
print("开始批量生成对比图")
print("=" * 50)

total_generated = 0

for city, img_dir in cities.items():
    print(f"\n处理城市: {city}")
    
    # 获取该城市的图片列表
    img_files = [f for f in os.listdir(img_dir) if f.endswith('_leftImg8bit.png')]
    img_files.sort()  # 按文件名排序
    selected = img_files[:images_per_city]
    
    print(f"  找到 {len(img_files)} 张图片，选择前 {len(selected)} 张")
    
    for img_file in selected:
        # 提取 base name
        base_name = img_file.replace('_leftImg8bit.png', '')
        
        # 构造路径
        image_path = os.path.join(img_dir, img_file)
        complexity_path = f"/root/autodl-tmp/ICTNet-main/output/cityscapes/ictednet_small_city_train/complexity_maps_color/{base_name}_complexity_color.png"
        gt_path = f"/root/autodl-tmp/datasets/cityscapes/gtFine/val/{city}/{base_name}_gtFine_color.png"
        pred_path = f"/root/autodl-tmp/ICTNet-main/output/cityscapes/ictednet_small_city_train/val_vis_results/{base_name}_gtFine_labelIds.png"
        output_path = os.path.join(output_base, f"comparison_{city}_{base_name}.png")
        
        if create_comparison(image_path, complexity_path, gt_path, pred_path, output_path, f"{city}_{base_name}"):
            total_generated += 1

print("\n" + "=" * 50)
print(f"完成！共生成 {total_generated} 张对比图")
print(f"保存位置: {output_base}")
print("=" * 50)
