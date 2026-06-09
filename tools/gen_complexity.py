import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '/root/autodl-tmp/ICTNet-main')

from configs import config, update_config
import argparse
import models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', default='./complexity_maps_color')
    args = parser.parse_args()
    
    args.opts = []
    update_config(config, args)
    
    model_name = getattr(models, config.MODEL.SUBNAME)
    model = model_name.get_seg_model(config, imgnet_pretrained=False)
    
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model_dict = model.state_dict()
    state_dict = {k[6:]: v for k, v in state_dict.items() if k[6:] in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model = model.cuda().eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(args.input_dir) if f.endswith('.png')]
    
    for f in tqdm(img_files):
        img = cv2.imread(os.path.join(args.input_dir, f))
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (1024, 512))
        img_tensor = torch.from_numpy(img_resized).float().permute(2,0,1).unsqueeze(0).cuda()
        
        with torch.no_grad():
            _, ic_map = model(img_tensor)
        
        ic_map_np = ic_map.squeeze().cpu().numpy()
        ic_map_np = cv2.resize(ic_map_np, (w, h))
        ic_map_np = (ic_map_np * 255).astype(np.uint8)
        ic_map_color = cv2.applyColorMap(ic_map_np, cv2.COLORMAP_JET)
        
        base = os.path.splitext(f)[0].replace('_leftImg8bit', '')
        out = os.path.join(args.output_dir, f'{base}_complexity_color.png')
        cv2.imwrite(out, ic_map_color)
    
    print(f'Done! Saved to {args.output_dir}')

if __name__ == '__main__':
    main()
