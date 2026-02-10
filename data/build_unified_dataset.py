"""
Convert FLIR ADAS v2 + LLVIP into a unified dataset for multi-sensor detection.

Strategy:
- FLIR ADAS: Use thermal images (640x512) as both thermal AND pseudo-RGB input
  since RGB/Thermal are not paired 1:1. This gives us multi-class detections.
- LLVIP: Use real RGB+Thermal paired images for person class.
- Unified classes: person, bike, car, motor, bus, truck, light, sign, dog, other_vehicle
  (top 10 classes with meaningful annotation counts)
- Output: data/unified_dataset/ with rgb/, thermal/, annotations.json (COCO format)
"""

import json
import os
import shutil
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
import argparse

# ── Unified class mapping (0-indexed for model) ──
UNIFIED_CLASSES = [
    'person',        # 0
    'bike',          # 1 
    'car',           # 2
    'motor',         # 3
    'bus',           # 4
    'truck',         # 5
    'light',         # 6
    'sign',          # 7
    'dog',           # 8
    'other_vehicle', # 9
]

# FLIR category_id -> unified class index
FLIR_TO_UNIFIED = {
    1: 0,   # person
    2: 1,   # bike
    3: 2,   # car
    4: 3,   # motor
    6: 4,   # bus
    8: 5,   # truck
    10: 6,  # light
    12: 7,  # sign
    17: 8,  # dog
    79: 9,  # other vehicle
}

# LLVIP has only person (category_id=0 -> unified 0)
LLVIP_TO_UNIFIED = {
    0: 0,   # person
    1: 0,   # person (in case some use 1-indexed)
}


def process_flir(flir_base, output_dir, split='train', max_images=None):
    """Process FLIR ADAS thermal images into unified format."""
    
    thermal_dir = os.path.join(flir_base, f'images_thermal_{split}')
    coco_path = os.path.join(thermal_dir, 'coco.json')
    
    if not os.path.exists(coco_path):
        print(f"  WARNING: {coco_path} not found, skipping FLIR {split}")
        return [], [], 0
    
    coco = json.load(open(coco_path))
    
    # Build annotation index: image_id -> list of annotations
    ann_by_img = defaultdict(list)
    for ann in coco['annotations']:
        cat_id = ann['category_id']
        if cat_id in FLIR_TO_UNIFIED:
            ann_by_img[ann['image_id']].append(ann)
    
    # Only use images that have at least one valid annotation
    valid_images = [img for img in coco['images'] if img['id'] in ann_by_img]
    
    if max_images and len(valid_images) > max_images:
        # Sample diverse images
        np.random.seed(42)
        indices = np.random.choice(len(valid_images), max_images, replace=False)
        valid_images = [valid_images[i] for i in sorted(indices)]
    
    images_out = []
    annotations_out = []
    processed = 0
    
    for img_info in valid_images:
        src_path = os.path.join(thermal_dir, img_info['file_name'])
        if not os.path.exists(src_path):
            continue
        
        # Copy thermal image
        out_name = f"flir_{split}_{processed:05d}.jpg"
        dst_thermal = os.path.join(output_dir, 'thermal', out_name)
        dst_rgb = os.path.join(output_dir, 'rgb', out_name)
        
        # For FLIR: use thermal as thermal, and convert thermal to 3-channel pseudo-RGB
        try:
            img = Image.open(src_path)
            # Save as thermal (grayscale)
            if img.mode != 'L':
                img_gray = img.convert('L')
            else:
                img_gray = img
            img_gray.save(dst_thermal)
            
            # Save as pseudo-RGB (thermal colorized or just 3-channel gray)
            img_rgb = img.convert('RGB')
            img_rgb.save(dst_rgb)
        except Exception as e:
            print(f"  Error processing {src_path}: {e}")
            continue
        
        # Get image dimensions
        w, h = img.size
        
        # Process annotations
        img_anns = ann_by_img[img_info['id']]
        for ann in img_anns:
            cat_id = ann['category_id']
            unified_cls = FLIR_TO_UNIFIED[cat_id]
            x, y, bw, bh = ann['bbox']  # COCO format: x, y, width, height
            
            # Skip tiny annotations
            if bw < 5 or bh < 5:
                continue
                
            annotations_out.append({
                'image_id': processed,
                'category_id': unified_cls,
                'bbox': [float(x), float(y), float(bw), float(bh)],
                'area': float(bw * bh),
                'iscrowd': 0
            })
        
        images_out.append({
            'id': processed,
            'file_name': out_name,
            'width': w,
            'height': h,
            'source': 'flir_adas_v2'
        })
        
        processed += 1
        if processed % 200 == 0:
            print(f"    FLIR {split}: {processed}/{len(valid_images)} processed")
    
    print(f"  FLIR {split}: {processed} images, {len(annotations_out)} annotations")
    return images_out, annotations_out, processed


def process_llvip(llvip_dir, output_dir, start_id=0, max_images=None):
    """Process LLVIP paired RGB+Thermal images into unified format."""
    
    ann_path = os.path.join(llvip_dir, 'annotations.json')
    if not os.path.exists(ann_path):
        print(f"  WARNING: {ann_path} not found, skipping LLVIP")
        return [], [], 0
    
    coco = json.load(open(ann_path))
    
    # Build annotation index
    ann_by_img = defaultdict(list)
    for ann in coco['annotations']:
        ann_by_img[ann['image_id']].append(ann)
    
    valid_images = [img for img in coco['images'] if img['id'] in ann_by_img]
    
    if max_images and len(valid_images) > max_images:
        np.random.seed(43)
        indices = np.random.choice(len(valid_images), max_images, replace=False)
        valid_images = [valid_images[i] for i in sorted(indices)]
    
    images_out = []
    annotations_out = []
    processed = 0
    
    for img_info in valid_images:
        rgb_src = os.path.join(llvip_dir, 'rgb', img_info['file_name'])
        thermal_src = os.path.join(llvip_dir, 'thermal', img_info['file_name'])
        
        if not (os.path.exists(rgb_src) and os.path.exists(thermal_src)):
            continue
        
        out_id = start_id + processed
        out_name = f"llvip_{processed:05d}.jpg"
        
        dst_rgb = os.path.join(output_dir, 'rgb', out_name)
        dst_thermal = os.path.join(output_dir, 'thermal', out_name)
        
        try:
            # Copy RGB as-is
            shutil.copy2(rgb_src, dst_rgb)
            # Copy thermal (ensure grayscale)
            th = Image.open(thermal_src)
            if th.mode != 'L':
                th = th.convert('L')
            th.save(dst_thermal)
        except Exception as e:
            print(f"  Error: {e}")
            continue
        
        w, h = Image.open(rgb_src).size
        
        # LLVIP annotations -> unified (all are person = class 0)
        for ann in ann_by_img[img_info['id']]:
            old_cat = ann.get('category_id', 1)
            if old_cat in LLVIP_TO_UNIFIED:
                x, y, bw, bh = ann['bbox']
                if bw < 5 or bh < 5:
                    continue
                annotations_out.append({
                    'image_id': out_id,
                    'category_id': LLVIP_TO_UNIFIED[old_cat],
                    'bbox': [float(x), float(y), float(bw), float(bh)],
                    'area': float(bw * bh),
                    'iscrowd': 0
                })
        
        images_out.append({
            'id': out_id,
            'file_name': out_name,
            'width': w,
            'height': h,
            'source': 'llvip'
        })
        
        processed += 1
        if processed % 500 == 0:
            print(f"    LLVIP: {processed}/{len(valid_images)} processed")
    
    print(f"  LLVIP: {processed} images, {len(annotations_out)} annotations")
    return images_out, annotations_out, processed


def main():
    parser = argparse.ArgumentParser(description='Build unified FLIR+LLVIP dataset')
    parser.add_argument('--flir-dir', default='data/FLIR_ADAS_v2')
    parser.add_argument('--llvip-dir', default='data/llvip_processed')
    parser.add_argument('--output-dir', default='data/unified_dataset')
    parser.add_argument('--max-flir-train', type=int, default=3000,
                        help='Max FLIR training images (default: 3000)')
    parser.add_argument('--max-flir-val', type=int, default=500,
                        help='Max FLIR validation images')
    parser.add_argument('--max-llvip', type=int, default=2000,
                        help='Max LLVIP images (default: 2000)')
    parser.add_argument('--all', action='store_true',
                        help='Use ALL available images (no limit)')
    args = parser.parse_args()
    
    if args.all:
        args.max_flir_train = None
        args.max_flir_val = None
        args.max_llvip = None
    
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'thermal'), exist_ok=True)
    
    print("=" * 60)
    print("Building Unified FLIR ADAS v2 + LLVIP Dataset")
    print("=" * 60)
    print(f"  FLIR dir:   {args.flir_dir}")
    print(f"  LLVIP dir:  {args.llvip_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Max FLIR train: {'ALL' if args.max_flir_train is None else args.max_flir_train}")
    print(f"  Max FLIR val:   {'ALL' if args.max_flir_val is None else args.max_flir_val}")
    print(f"  Max LLVIP:      {'ALL' if args.max_llvip is None else args.max_llvip}")
    print(f"  Classes: {UNIFIED_CLASSES}")
    print()
    
    all_images = []
    all_annotations = []
    ann_id = 0
    img_id_offset = 0
    
    # ── 1. Process FLIR train ──
    print("[1/3] Processing FLIR ADAS thermal train...")
    flir_imgs, flir_anns, flir_count = process_flir(
        args.flir_dir, output_dir, split='train', max_images=args.max_flir_train
    )
    # Assign annotation IDs
    for ann in flir_anns:
        ann['id'] = ann_id
        ann_id += 1
    all_images.extend(flir_imgs)
    all_annotations.extend(flir_anns)
    img_id_offset += flir_count
    
    # ── 2. Process FLIR val (as additional data) ──
    print("\n[2/3] Processing FLIR ADAS thermal val...")
    # Remap image IDs for val
    flir_val_imgs, flir_val_anns, flir_val_count = process_flir(
        args.flir_dir, output_dir, split='val', max_images=args.max_flir_val
    )
    for img in flir_val_imgs:
        img['id'] += img_id_offset
    for ann in flir_val_anns:
        ann['image_id'] += img_id_offset
        ann['id'] = ann_id
        ann_id += 1
    all_images.extend(flir_val_imgs)
    all_annotations.extend(flir_val_anns)
    img_id_offset += flir_val_count
    
    # ── 3. Process LLVIP ──
    print("\n[3/3] Processing LLVIP paired data...")
    llvip_imgs, llvip_anns, llvip_count = process_llvip(
        args.llvip_dir, output_dir, start_id=img_id_offset, max_images=args.max_llvip
    )
    for ann in llvip_anns:
        ann['id'] = ann_id
        ann_id += 1
    all_images.extend(llvip_imgs)
    all_annotations.extend(llvip_anns)
    
    # ── Build unified COCO JSON ──
    categories = [{'id': i, 'name': name} for i, name in enumerate(UNIFIED_CLASSES)]
    
    unified_coco = {
        'images': all_images,
        'annotations': all_annotations,
        'categories': categories,
        'info': {
            'description': 'Unified FLIR ADAS v2 + LLVIP dataset for multi-sensor object detection',
            'num_classes': len(UNIFIED_CLASSES),
            'class_names': UNIFIED_CLASSES,
        }
    }
    
    ann_path = os.path.join(output_dir, 'annotations.json')
    with open(ann_path, 'w') as f:
        json.dump(unified_coco, f)
    
    # ── Summary ──
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total images:      {len(all_images)}")
    print(f"Total annotations: {len(all_annotations)}")
    print(f"  FLIR train:      {flir_count} images")
    print(f"  FLIR val:        {flir_val_count} images")
    print(f"  LLVIP:           {llvip_count} images")
    print(f"Classes ({len(UNIFIED_CLASSES)}):")
    
    class_counts = Counter(a['category_id'] for a in all_annotations)
    for cls_id, cls_name in enumerate(UNIFIED_CLASSES):
        count = class_counts.get(cls_id, 0)
        print(f"  [{cls_id}] {cls_name:20s}: {count:6d}")
    
    print(f"\nSaved to: {ann_path}")
    print(f"RGB dir:     {os.path.join(output_dir, 'rgb')}")
    print(f"Thermal dir: {os.path.join(output_dir, 'thermal')}")
    print("Done!")


if __name__ == '__main__':
    main()
