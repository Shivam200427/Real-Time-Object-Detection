"""
Finish LLVIP dataset setup: copy remaining files and generate annotations.json.
Skips already-copied files for speed.
"""
import os, sys, shutil, json, glob
import xml.etree.ElementTree as ET

DATA_DIR = os.path.join(os.path.dirname(__file__), "LLVIP")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "llvip_processed")

rgb_out = os.path.join(OUTPUT_DIR, "rgb")
thermal_out = os.path.join(OUTPUT_DIR, "thermal")
os.makedirs(rgb_out, exist_ok=True)
os.makedirs(thermal_out, exist_ok=True)

visible_dir = os.path.join(DATA_DIR, "visible")
infrared_dir = os.path.join(DATA_DIR, "infrared")
ann_dir = os.path.join(DATA_DIR, "Annotations")

# Gather all image files
vis_images = sorted(glob.glob(os.path.join(visible_dir, "**", "*.jpg"), recursive=True) +
                    glob.glob(os.path.join(visible_dir, "**", "*.png"), recursive=True))
ir_images = sorted(glob.glob(os.path.join(infrared_dir, "**", "*.jpg"), recursive=True) +
                   glob.glob(os.path.join(infrared_dir, "**", "*.png"), recursive=True))

vis_dict = {os.path.basename(p): p for p in vis_images}
ir_dict = {os.path.basename(p): p for p in ir_images}
common_names = sorted(set(vis_dict.keys()) & set(ir_dict.keys()))

print(f"Total matched pairs: {len(common_names)}")

# Copy remaining files (skip if destination exists)
copied = 0
skipped = 0
for idx, name in enumerate(common_names):
    new_name = f"img_{idx:04d}.jpg"
    rgb_dst = os.path.join(rgb_out, new_name)
    th_dst = os.path.join(thermal_out, new_name)
    
    if not os.path.exists(rgb_dst):
        shutil.copy2(vis_dict[name], rgb_dst)
        copied += 1
    else:
        skipped += 1
    
    if not os.path.exists(th_dst):
        shutil.copy2(ir_dict[name], th_dst)
    
    if (idx + 1) % 2000 == 0:
        print(f"  Progress: {idx+1}/{len(common_names)} (copied {copied}, skipped {skipped})")

print(f"\nCopy done: {copied} new, {skipped} skipped")

# Now generate annotations.json
print("\nGenerating annotations.json...")
annotations = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 0, "name": "person"}]
}

ann_id = 0
parsed = 0
for idx, name in enumerate(common_names):
    new_name = f"img_{idx:04d}.jpg"
    
    # Get actual image dimensions from the first annotation
    img_w, img_h = 1280, 1024  # LLVIP default
    
    annotations["images"].append({
        "id": idx,
        "file_name": new_name,
        "width": img_w,
        "height": img_h,
        "original_name": name
    })
    
    # Parse XML annotation
    xml_name = os.path.splitext(name)[0] + ".xml"
    xml_path = os.path.join(ann_dir, xml_name)
    
    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get real size from XML
            size_el = root.find('size')
            if size_el is not None:
                w_el = size_el.find('width')
                h_el = size_el.find('height')
                if w_el is not None:
                    img_w = int(w_el.text)
                if h_el is not None:
                    img_h = int(h_el.text)
                annotations["images"][-1]["width"] = img_w
                annotations["images"][-1]["height"] = img_h
            
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
            parsed += 1
        except Exception as e:
            print(f"  Warning: {xml_path}: {e}")
    
    if (idx + 1) % 3000 == 0:
        print(f"  Annotations progress: {idx+1}/{len(common_names)}")

ann_path = os.path.join(OUTPUT_DIR, "annotations.json")
with open(ann_path, 'w') as f:
    json.dump(annotations, f)

print(f"\nDone!")
print(f"  Images: {len(annotations['images'])}")
print(f"  Annotations: {len(annotations['annotations'])}")
print(f"  XMLs parsed: {parsed}")
print(f"  Saved to: {ann_path}")
