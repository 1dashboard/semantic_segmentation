import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.insert(0, '.')

from configs import config, update_config
import argparse
from datasets.cityscapes import Cityscapes
from datasets.camvid import CamVid
import models

# ---------- 颜色映射 ----------
cityscapes_colors = [
    (128,64,128), (244,35,232), (70,70,70), (102,102,156), (190,153,153),
    (153,153,153), (250,170,30), (220,220,0), (107,142,35), (152,251,152),
    (70,130,180), (220,20,60), (255,0,0), (0,0,142), (0,0,70),
    (0,60,100), (0,80,100), (0,0,230), (119,11,32)
]

camvid_colors = [
    (192,128,0), (0,0,128), (128,0,64), (128,192,192), (128,64,64),
    (0,64,64), (128,64,128), (192,0,0), (128,128,192), (0,128,128), (128,128,128)
]

def apply_heatmap(ic_np):
    """连续热力图（JET）"""
    ic_uint8 = (ic_np * 255).astype(np.uint8)
    heat = cv2.applyColorMap(ic_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

def label_to_color(label_np, colors):
    h, w = label_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for i, col in enumerate(colors):
        color[label_np == i] = col
    return color

def safe_put_text(img, text, org, font, font_scale, color, thickness):
    """确保图像连续且类型正确后再绘制文字"""
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return cv2.putText(img, text, org, font, font_scale, color, thickness)

def create_comparison(img_vis, comp_vis, gt_vis, pred_vis, out_path):
    # 统一尺寸
    h, w = img_vis.shape[:2]
    comp_vis = cv2.resize(comp_vis, (w, h))
    gt_vis = cv2.resize(gt_vis, (w, h))
    pred_vis = cv2.resize(pred_vis, (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    titles = ["Input Image", "Complexity Map", "Ground Truth", "ICTNet-S"]
    images = [img_vis, comp_vis, gt_vis, pred_vis]
    for im, title in zip(images, titles):
        safe_put_text(im, title, (15, 35), font, 0.8, (0,0,0), 3)
        safe_put_text(im, title, (15, 35), font, 0.8, (255,255,255), 2)

    top = np.hstack([images[0], images[1]])
    bottom = np.hstack([images[2], images[3]])
    result = np.vstack([top, bottom])

    # 确保最终结果连续且类型正确
    if not result.flags['C_CONTIGUOUS']:
        result = np.ascontiguousarray(result)
    if result.dtype != np.uint8:
        result = result.astype(np.uint8)

    plt.figure(figsize=(16,12))
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cityscapes', 'camvid'], required=True)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    # 更新配置（必须添加 opts 属性）
    args.opts = []
    update_config(config, args)

    # 数据集相关
    if args.dataset == 'cityscapes':
        DS = Cityscapes
        colors = cityscapes_colors
        num_classes = 19
        list_path = 'list/cityscapes/val.lst'
        crop_size = (1024, 2048)
        base_size = 2048
    else:
        DS = CamVid
        colors = camvid_colors
        num_classes = 11
        list_path = 'list/camvid/val.lst'
        crop_size = (720, 960)
        base_size = 960

    dataset = DS(
        root=config.DATASET.ROOT,
        list_path=list_path,
        num_classes=num_classes,
        multi_scale=False, flip=False,
        ignore_label=255,
        base_size=base_size,
        crop_size=crop_size
    )

    img_data, label_np, _, size, name = dataset[args.index]
    print(f'Processing: {name}')

    # ----- 原始图像（反归一化）-----
    if torch.is_tensor(img_data):
        img_np = img_data.cpu().numpy()
    else:
        img_np = img_data
    if img_np.shape[0] == 3:
        img_np = img_np.transpose(1,2,0)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    img_vis = (img_np * 255).astype(np.uint8)

    h, w = label_np.shape
    if img_vis.shape[:2] != (h, w):
        img_vis = cv2.resize(img_vis, (w, h))
    img_vis = np.ascontiguousarray(img_vis)

    # ----- 加载模型 -----
    model_name = getattr(models, config.MODEL.SUBNAME)
    model = model_name.get_seg_model(config, imgnet_pretrained=False)
    state_dict = torch.load(args.model_file, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model_dict = model.state_dict()
    state_dict = {k[6:]: v for k, v in state_dict.items() if k[6:] in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model = model.cuda().eval()

    # ----- 推理：得到分割预测和复杂度图 -----
    if torch.is_tensor(img_data):
        img_input = img_data.unsqueeze(0).cuda()
    else:
        img_tensor = torch.from_numpy(img_data).float()
        if img_tensor.shape[0] == 3:
            img_input = img_tensor.unsqueeze(0).cuda()
        else:
            img_input = img_tensor.permute(2,0,1).unsqueeze(0).cuda()

    with torch.no_grad():
        seg_out, ic_out = model(img_input)   # seg_out 可能是 list（训练模式）或 tensor
        if isinstance(seg_out, (list, tuple)):
            seg_logits = seg_out[-1]
        else:
            seg_logits = seg_out
        pred_label = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        ic_map = ic_out.squeeze().cpu().numpy()

    # ----- 复杂度图（连续热力图）-----
    comp_vis = apply_heatmap(ic_map)
    if comp_vis.shape[:2] != (h, w):
        comp_vis = cv2.resize(comp_vis, (w, h))
    comp_vis = np.ascontiguousarray(comp_vis)

    # ----- 真实分割结果（彩色）-----
    pred_color = label_to_color(pred_label, colors)
    # ----- Ground Truth（彩色）-----
    gt_color = label_to_color(label_np, colors)

    # ----- 保存对比图 -----
    out_dir = f'output/{args.dataset}/ictednet_small_{args.dataset}'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/real_comparison_{name}.png'
    create_comparison(img_vis, comp_vis, gt_color, pred_color, out_path)

if __name__ == '__main__':
    main()
