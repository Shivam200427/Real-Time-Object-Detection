"""Analyze FLIR ADAS v2 dataset structure."""
import json, os
from collections import Counter

base = 'data/FLIR_ADAS_v2'

# Check thermal train
tcoco = json.load(open(os.path.join(base, 'images_thermal_train/coco.json')))
cat_map = {c['id']: c['name'] for c in tcoco['categories']}
cat_counts = Counter(a['category_id'] for a in tcoco['annotations'])
print('=== Thermal Train class distribution (top 15) ===')
for cid, count in cat_counts.most_common(15):
    name = cat_map.get(cid, f'id_{cid}')
    print(f'  {name:20s} (id={cid:2d}): {count:6d}')

# Check RGB train
rcoco = json.load(open(os.path.join(base, 'images_rgb_train/coco.json')))
print(f'\nRGB train: {len(rcoco["images"])} images, {len(rcoco["annotations"])} annotations')
print(f'Thermal train: {len(tcoco["images"])} images, {len(tcoco["annotations"])} annotations')

# Video IDs
r_vids = set(i['extra_info']['video_id'] for i in rcoco['images'])
t_vids = set(i['extra_info']['video_id'] for i in tcoco['images'])
shared = r_vids & t_vids
print(f'\nRGB videos: {len(r_vids)}, Thermal videos: {len(t_vids)}, Shared: {len(shared)}')

# Frame-level pairing: check if same video_id + frame number exists in both
import re
def extract_frame_info(fname):
    m = re.search(r'video-(.+?)-frame-(\d+)', fname)
    if m:
        return m.group(1), int(m.group(2))
    return None, None

# Build thermal frame index
t_frame_idx = {}
for img in tcoco['images']:
    vid, frame = extract_frame_info(img['file_name'])
    if vid:
        t_frame_idx[(vid, frame)] = img

# Try to match RGB frames to thermal
matched = 0
for img in rcoco['images'][:500]:
    vid, frame = extract_frame_info(img['file_name'])
    if vid and (vid, frame) in t_frame_idx:
        matched += 1

print(f'\nFrame-level RGB-Thermal matches (first 500 RGB): {matched}/500')

# Image sizes
r_sizes = Counter((i['width'], i['height']) for i in rcoco['images'])
t_sizes = Counter((i['width'], i['height']) for i in tcoco['images'])
print(f'\nRGB sizes: {r_sizes.most_common(3)}')
print(f'Thermal sizes: {t_sizes.most_common(3)}')

# File existence check
f = rcoco['images'][0]['file_name']
full = os.path.join(base, 'images_rgb_train', f)
print(f'\nRGB file exists: {os.path.exists(full)} ({full})')
f2 = tcoco['images'][0]['file_name']
full2 = os.path.join(base, 'images_thermal_train', f2)
print(f'Thermal file exists: {os.path.exists(full2)} ({full2})')

# Count files in data dirs
rgb_data = os.path.join(base, 'images_rgb_train/data')
thermal_data = os.path.join(base, 'images_thermal_train/data')
print(f'\nRGB data files: {len(os.listdir(rgb_data)) if os.path.isdir(rgb_data) else "NOT FOUND"}')
print(f'Thermal data files: {len(os.listdir(thermal_data)) if os.path.isdir(thermal_data) else "NOT FOUND"}')

# Check val sets too
for split in ['images_rgb_val', 'images_thermal_val']:
    p = os.path.join(base, split, 'coco.json')
    if os.path.exists(p):
        c = json.load(open(p))
        print(f'{split}: {len(c["images"])} images, {len(c["annotations"])} annotations')
