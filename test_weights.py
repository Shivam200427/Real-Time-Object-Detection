"""Quick script to test weight loading and inference."""
import os, sys, torch, numpy as np
sys.path.insert(0, '.')

from src.utils import load_config, resize_image, get_device
from src.model import build_model
from PIL import Image

WEIGHTS_PATH = 'weights/best_trained.pth'
print(f'Weights exist: {os.path.exists(WEIGHTS_PATH)}')
print(f'Weights size: {os.path.getsize(WEIGHTS_PATH) / 1e6:.1f} MB')

ckpt = torch.load(WEIGHTS_PATH, map_location='cpu')
print(f'Checkpoint keys: {list(ckpt.keys())}')
cfg = ckpt['config']
backbone = cfg['model']['backbone']
num_cls = cfg['model']['num_classes']
img_sz = cfg['dataset']['img_size']
print(f'Config: backbone={backbone}, num_classes={num_cls}, img_size={img_sz}')

print('\nBuilding model...')
model = build_model(cfg)
print('Loading state dict...')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print('Model loaded OK!')

# Test with real LLVIP image
rgb_path = 'data/llvip_processed/rgb/img_0000.jpg'
th_path = 'data/llvip_processed/thermal/img_0000.jpg'
img_size = tuple(img_sz)

rgb_np = np.array(Image.open(rgb_path).convert('RGB'))
th_np = np.array(Image.open(th_path).convert('L'))
th_3ch = np.stack([th_np]*3, -1)

rgb_r = resize_image(rgb_np, img_size)
th_r = resize_image(th_3ch, img_size)

rgb_t = torch.from_numpy(rgb_r).float().permute(2,0,1).unsqueeze(0) / 255.0
th_t = torch.from_numpy(th_r).float().permute(2,0,1).unsqueeze(0) / 255.0

print(f'\nReal LLVIP image test (img_0000):')
print(f'  RGB: {rgb_t.shape}, Thermal: {th_t.shape}')

for thresh in [0.3, 0.15, 0.1, 0.05, 0.01]:
    results = model.predict(rgb_t, th_t, score_thresh=thresh, img_size=img_size)
    n = len(results[0]['boxes'])
    scores = results[0]['scores']
    if n > 0:
        top = [f'{float(s):.3f}' for s in scores[:5]]
        print(f'  thresh={thresh}: {n} detections, top scores: {top}')
    else:
        print(f'  thresh={thresh}: 0 detections')

# Try a few more images
for idx in [1, 5, 10, 50]:
    rp = f'data/llvip_processed/rgb/img_{idx:04d}.jpg'
    tp = f'data/llvip_processed/thermal/img_{idx:04d}.jpg'
    if os.path.exists(rp):
        r = resize_image(np.array(Image.open(rp).convert('RGB')), img_size)
        t = resize_image(np.stack([np.array(Image.open(tp).convert('L'))]*3, -1), img_size)
        rt = torch.from_numpy(r).float().permute(2,0,1).unsqueeze(0) / 255.0
        tt = torch.from_numpy(t).float().permute(2,0,1).unsqueeze(0) / 255.0
        res = model.predict(rt, tt, score_thresh=0.05, img_size=img_size)
        n = len(res[0]['boxes'])
        sc = [f'{float(s):.3f}' for s in res[0]['scores'][:3]] if n > 0 else []
        print(f'  img_{idx:04d}: {n} detections at thresh=0.05, scores={sc}')
