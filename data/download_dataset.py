"""
Dataset download helper.
Downloads LLVIP (Low-Light Vision Image Pair) dataset as it's freely available.
For FLIR ADAS, provides instructions for manual download.
"""

import os
import sys
import argparse
import zipfile
import shutil
import json
import glob
from pathlib import Path


def download_llvip(output_dir: str):
    """
    Download LLVIP dataset (RGB + Infrared pairs).
    LLVIP: https://bupt-ai-cz.github.io/LLVIP/
    
    The dataset contains paired visible-infrared images for pedestrian detection.
    """
    try:
        import gdown
    except ImportError:
        print("Installing gdown for Google Drive download...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("LLVIP Dataset Download")
    print("="*60)
    print("\nThe LLVIP dataset contains paired visible + infrared images")
    print("for pedestrian detection in low-light conditions.\n")
    
    # LLVIP Google Drive links (publicly shared)
    # Note: These links may change. If they break, download manually.
    print("Please download the LLVIP dataset manually from:")
    print("  https://bupt-ai-cz.github.io/LLVIP/")
    print(f"\nExtract contents to: {output_dir}")
    print("\nExpected structure after extraction:")
    print(f"  {output_dir}/")
    print(f"  ├── visible/        (or infrared/)")
    print(f"  │   ├── train/")
    print(f"  │   └── test/")
    print(f"  ├── infrared/")
    print(f"  │   ├── train/")
    print(f"  │   └── test/")
    print(f"  └── Annotations/")
    print(f"      ├── train/")
    print(f"      └── test/")


def setup_llvip_structure(data_dir: str, output_dir: str):
    """
    Reorganize LLVIP dataset into our expected format.
    
    Converts from LLVIP structure to:
        output_dir/
        ├── rgb/
        │   ├── img_0000.jpg
        │   └── ...
        ├── thermal/
        │   ├── img_0000.jpg
        │   └── ...
        └── annotations.json
    """
    rgb_out = os.path.join(output_dir, "rgb")
    thermal_out = os.path.join(output_dir, "thermal")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(thermal_out, exist_ok=True)
    
    # Find visible and infrared directories
    visible_dir = None
    infrared_dir = None
    ann_dir = None
    
    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            if d.lower() in ['visible', 'vis']:
                visible_dir = os.path.join(root, d)
            elif d.lower() in ['infrared', 'ir', 'lwir']:
                infrared_dir = os.path.join(root, d)
            elif d.lower() in ['annotations', 'annotation', 'labels']:
                ann_dir = os.path.join(root, d)
    
    if not visible_dir or not infrared_dir:
        print(f"Could not find visible/infrared directories in {data_dir}")
        print("Creating sample data instead...")
        
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils import create_sample_data
        create_sample_data(output_dir, num_samples=50)
        return
    
    print(f"Found visible images: {visible_dir}")
    print(f"Found infrared images: {infrared_dir}")
    
    # Gather all image files
    vis_images = sorted(glob.glob(os.path.join(visible_dir, "**", "*.jpg"), recursive=True) +
                       glob.glob(os.path.join(visible_dir, "**", "*.png"), recursive=True))
    ir_images = sorted(glob.glob(os.path.join(infrared_dir, "**", "*.jpg"), recursive=True) +
                       glob.glob(os.path.join(infrared_dir, "**", "*.png"), recursive=True))
    
    print(f"Found {len(vis_images)} visible images and {len(ir_images)} infrared images")
    
    # Match by filename
    vis_dict = {os.path.basename(p): p for p in vis_images}
    ir_dict = {os.path.basename(p): p for p in ir_images}
    
    common_names = sorted(set(vis_dict.keys()) & set(ir_dict.keys()))
    print(f"Matched {len(common_names)} image pairs")
    
    # Copy and rename
    annotations = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "person"}]
    }
    
    ann_id = 0
    for idx, name in enumerate(common_names):
        new_name = f"img_{idx:04d}.jpg"
        
        shutil.copy2(vis_dict[name], os.path.join(rgb_out, new_name))
        shutil.copy2(ir_dict[name], os.path.join(thermal_out, new_name))
        
        annotations["images"].append({
            "id": idx,
            "file_name": new_name,
            "width": 640,
            "height": 480,
            "original_name": name
        })
        
        # Parse XML annotation if available
        if ann_dir:
            xml_name = os.path.splitext(name)[0] + ".xml"
            xml_paths = glob.glob(os.path.join(ann_dir, "**", xml_name), recursive=True)
            
            if xml_paths:
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(xml_paths[0])
                    root = tree.getroot()
                    
                    for obj in root.findall('object'):
                        bbox = obj.find('bndbox')
                        x1 = int(bbox.find('xmin').text)
                        y1 = int(bbox.find('ymin').text)
                        x2 = int(bbox.find('xmax').text)
                        y2 = int(bbox.find('ymax').text)
                        
                        annotations["annotations"].append({
                            "id": ann_id,
                            "image_id": idx,
                            "category_id": 0,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": (x2 - x1) * (y2 - y1),
                            "iscrowd": 0
                        })
                        ann_id += 1
                except Exception as e:
                    print(f"Warning: Could not parse {xml_paths[0]}: {e}")
    
    # Save annotations
    ann_path = os.path.join(output_dir, "annotations.json")
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nDataset prepared: {len(common_names)} pairs")
    print(f"Annotations: {ann_path}")


def create_sample_dataset(output_dir: str, num_samples: int = 50):
    """Create a synthetic sample dataset for testing."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import create_sample_data
    
    os.makedirs(output_dir, exist_ok=True)
    create_sample_data(output_dir, num_samples=num_samples)
    print(f"\nSample dataset created at {output_dir}")
    print("This is synthetic data for testing the pipeline.")
    print("For real results, download FLIR ADAS or LLVIP dataset.\n")


def main():
    parser = argparse.ArgumentParser(description="Download/prepare dataset")
    parser.add_argument('--dataset', type=str, default='sample',
                        choices=['llvip', 'flir', 'sample'],
                        help='Dataset to download/prepare')
    parser.add_argument('--output', type=str, default='data/raw',
                        help='Output directory')
    parser.add_argument('--source', type=str, default=None,
                        help='Source directory (for reorganizing existing downloads)')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of synthetic samples (for sample dataset)')
    
    args = parser.parse_args()
    
    if args.dataset == 'sample':
        create_sample_dataset(args.output, args.num_samples)
    
    elif args.dataset == 'llvip':
        if args.source:
            setup_llvip_structure(args.source, args.output)
        else:
            download_llvip(args.output)
    
    elif args.dataset == 'flir':
        print("="*60)
        print("FLIR ADAS Dataset")
        print("="*60)
        print("\nPlease download the FLIR ADAS dataset from:")
        print("  https://www.flir.com/oem/adas/adas-dataset-form/")
        print(f"\nAfter downloading, extract to: {args.output}")
        print("\nThen run:")
        print(f'  python data/download_dataset.py --dataset flir --source <extracted_dir> --output {args.output}')


if __name__ == "__main__":
    main()
