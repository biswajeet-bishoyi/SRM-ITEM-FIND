"""
Campus Item Synthetic Image Generator for SRM CampusFind.
Synthesizes 100 realistic product photographs (50 Lost & Found pairs)
with authentic colors, geometries, textures, and camera perspective shifts.
"""

import os
import sys
import random
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.dataset_generator import RAW_SEED_ITEMS

IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

COLOR_PALETTE = {
    "black": (25, 25, 28),
    "space grey": (72, 75, 80),
    "grey": (110, 115, 120),
    "gray": (110, 115, 120),
    "silver": (205, 210, 215),
    "white": (245, 246, 248),
    "navy blue": (18, 42, 92),
    "blue": (30, 85, 185),
    "translucent blue": (55, 120, 210),
    "sky blue": (90, 165, 235),
    "red": (185, 30, 35),
    "maroon": (115, 20, 30),
    "rose gold": (185, 130, 130),
    "pink": (225, 135, 165),
    "green": (35, 115, 60),
    "olive green": (80, 95, 50),
    "yellow": (220, 185, 40),
    "brown": (105, 65, 40),
    "tan": (180, 140, 100),
    "purple": (110, 50, 150),
    "orange": (220, 95, 30)
}


def get_rgb_from_color_name(color_name: str) -> tuple:
    cn = str(color_name).lower()
    for k, v in COLOR_PALETTE.items():
        if k in cn:
            return v
    return (120, 120, 125)


def generate_single_item_image(
    pair_id: str,
    category: str,
    brand: str,
    color_name: str,
    desc: str,
    is_found: bool,
    seed_val: int
) -> Image.Image:
    random.seed(seed_val)
    np.random.seed(seed_val)
    
    width, height = 256, 256
    base_rgb = get_rgb_from_color_name(color_name)
    
    # 1. Background Generation
    # Lost images: Study desk / wooden library carrel / cafe table
    # Found images: Campus tile / linoleum floor / classroom bench
    noise = np.random.normal(0, 3, (height, width, 3))
    if is_found:
        bg_base = np.array([215, 218, 222], dtype=np.float32)
        bg = np.clip(bg_base + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(bg)
        draw = ImageDraw.Draw(img)
        # Tile grid seams
        draw.line([(0, 70), (256, 70)], fill=(190, 193, 198), width=1)
        draw.line([(185, 0), (185, 256)], fill=(190, 193, 198), width=1)
    else:
        bg_base = np.array([195, 165, 135], dtype=np.float32)
        bg = np.clip(bg_base + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(bg)
        draw = ImageDraw.Draw(img)
        # Wood grain lines
        for y in range(25, 256, 35):
            draw.line([(0, y), (256, y + random.randint(-4, 4))], fill=(175, 145, 115), width=1)

    draw = ImageDraw.Draw(img)
    
    # Positioning with camera perspective jitter
    offset_x = random.randint(-6, 6) + (8 if is_found else -8)
    offset_y = random.randint(-6, 6) + (6 if is_found else -6)
    cx, cy = 128 + offset_x, 128 + offset_y
    shadow_offset = 6 if not is_found else 4
    
    desc_l = desc.lower()
    cat_l = category.lower()
    
    # Render Foreground Geometry by Category / Item Type
    if "card" in desc_l or "document" in cat_l or "license" in desc_l:
        # ID Card / Driving License / Document badge
        w, h = 136, 88
        draw.rounded_rectangle([cx - w//2 + shadow_offset, cy - h//2 + shadow_offset, cx + w//2 + shadow_offset, cy + h//2 + shadow_offset], radius=8, fill=(115, 105, 95))
        draw.rounded_rectangle([cx - w//2, cy - h//2, cx + w//2, cy + h//2], radius=8, fill=base_rgb, outline=(50, 50, 50), width=2)
        # Blue banner at top for SRM ID
        banner_c = (20, 65, 155) if ("srm" in brand.lower() or "id" in desc_l) else (50, 50, 55)
        draw.rectangle([cx - w//2, cy - h//2, cx + w//2, cy - h//2 + 22], fill=banner_c)
        # Photo box
        draw.rectangle([cx - w//2 + 10, cy - h//2 + 30, cx - w//2 + 45, cy + h//2 - 12], fill=(230, 230, 235), outline=(90, 90, 95))
        # Text barcode stripes
        for line_y in range(cy - 8, cy + 26, 7):
            draw.line([(cx - 8, line_y), (cx + w//2 - 14, line_y)], fill=(70, 70, 75), width=2)
        # Lanyard loop
        draw.ellipse([cx - 8, cy - h//2 - 12, cx + 8, cy - h//2 + 4], fill=(25, 55, 140))

    elif "bag" in cat_l or "backpack" in desc_l or "pouch" in desc_l:
        # Backpack / Tote / Pouch
        w, h = 120, 150
        draw.rounded_rectangle([cx - w//2 + shadow_offset, cy - h//2 + shadow_offset, cx + w//2 + shadow_offset, cy + h//2 + shadow_offset], radius=24, fill=(105, 95, 90))
        draw.rounded_rectangle([cx - w//2, cy - h//2, cx + w//2, cy + h//2], radius=24, fill=base_rgb, outline=(35, 35, 40), width=2)
        # Front pocket
        pocket_c = tuple(max(0, c - 20) for c in base_rgb)
        draw.rounded_rectangle([cx - w//2 + 15, cy, cx + w//2 - 15, cy + h//2 - 12], radius=12, fill=pocket_c, outline=(35, 35, 40), width=1)
        # Zipper seam
        draw.arc([cx - w//2 + 15, cy - 8, cx + w//2 - 15, cy + 18], start=0, end=180, fill=(185, 185, 190), width=2)

    elif "bottle" in desc_l or "flask" in desc_l or "umbrella" in desc_l:
        # Water bottle / Flask
        w, h = 55, 160
        draw.rounded_rectangle([cx - w//2 + shadow_offset, cy - h//2 + shadow_offset, cx + w//2 + shadow_offset, cy + h//2 + shadow_offset], radius=14, fill=(105, 95, 90))
        draw.rounded_rectangle([cx - w//2, cy - h//2 + 25, cx + w//2, cy + h//2], radius=14, fill=base_rgb, outline=(45, 45, 50), width=2)
        # Specular reflection highlight
        hl_color = tuple(min(255, c + 55) for c in base_rgb)
        draw.line([(cx - 10, cy - h//2 + 32), (cx - 10, cy + h//2 - 12)], fill=hl_color, width=4)
        # Cap / Lid
        cap_c = (40, 42, 48) if base_rgb != (25, 25, 28) else (190, 190, 195)
        draw.rounded_rectangle([cx - 16, cy - h//2, cx + 16, cy - h//2 + 25], radius=5, fill=cap_c, outline=(30, 30, 30), width=1)

    elif "watch" in desc_l or "smartwatch" in desc_l:
        # Smartwatch / Wristwatch
        draw.rectangle([cx - 18, cy - 85, cx + 18, cy + 85], fill=(32, 34, 38), outline=(18, 18, 20))
        draw.rounded_rectangle([cx - 38 + shadow_offset, cy - 38 + shadow_offset, cx + 38 + shadow_offset, cy + 38 + shadow_offset], radius=16, fill=(105, 95, 90))
        draw.rounded_rectangle([cx - 38, cy - 38, cx + 38, cy + 38], radius=16, fill=base_rgb, outline=(20, 20, 20), width=2)
        # Glass display
        draw.rounded_rectangle([cx - 30, cy - 30, cx + 30, cy + 30], radius=12, fill=(12, 14, 18))
        draw.arc([cx - 20, cy - 20, cx + 20, cy + 20], start=45, end=270, fill=(35, 195, 140), width=2)

    elif "buds" in desc_l or "earbuds" in desc_l or "airpods" in desc_l or "headphones" in desc_l:
        # Earbuds case / Over-ear headphones
        if "headphone" in desc_l:
            # Over-ear headphones
            draw.arc([cx - 60, cy - 65, cx + 60, cy + 35], start=180, end=0, fill=(35, 35, 40), width=8)
            draw.ellipse([cx - 70, cy - 10, cx - 40, cy + 45], fill=base_rgb, outline=(25, 25, 30), width=2)
            draw.ellipse([cx + 40, cy - 10, cx + 70, cy + 45], fill=base_rgb, outline=(25, 25, 30), width=2)
        else:
            # Charging case
            w, h = 82, 62
            draw.rounded_rectangle([cx - w//2 + shadow_offset, cy - h//2 + shadow_offset, cx + w//2 + shadow_offset, cy + h//2 + shadow_offset], radius=22, fill=(105, 95, 90))
            draw.rounded_rectangle([cx - w//2, cy - h//2, cx + w//2, cy + h//2], radius=22, fill=base_rgb, outline=(45, 45, 50), width=2)
            draw.line([(cx - w//2 + 4, cy - 5), (cx + w//2 - 4, cy - 5)], fill=(45, 45, 50), width=1)
            draw.ellipse([cx - 3, cy + 11, cx + 3, cy + 15], fill=(45, 225, 85))

    elif "charger" in desc_l or "adapter" in desc_l or "cable" in desc_l:
        # Power Adapter
        w, h = 76, 76
        draw.rounded_rectangle([cx - w//2 + shadow_offset, cy - h//2 + shadow_offset, cx + w//2 + shadow_offset, cy + h//2 + shadow_offset], radius=10, fill=(105, 95, 90))
        draw.rounded_rectangle([cx - w//2, cy - h//2, cx + w//2, cy + h//2], radius=10, fill=base_rgb, outline=(38, 38, 42), width=2)
        draw.arc([cx + w//2 - 10, cy - 30, cx + w//2 + 40, cy + 30], start=270, end=90, fill=(38, 38, 42), width=4)
        draw.rectangle([cx - w//2 - 15, cy - 14, cx - w//2, cy - 6], fill=(195, 195, 200))
        draw.rectangle([cx - w//2 - 15, cy + 6, cx - w//2, cy + 14], fill=(195, 195, 200))

    elif "laptop" in desc_l or "macbook" in desc_l or "ipad" in desc_l or "tablet" in desc_l or "kindle" in desc_l:
        # Laptop / Tablet
        w, h = 154, 108
        draw.rounded_rectangle([cx - w//2 + shadow_offset, cy - h//2 + shadow_offset, cx + w//2 + shadow_offset, cy + h//2 + shadow_offset], radius=12, fill=(105, 95, 90))
        draw.rounded_rectangle([cx - w//2, cy - h//2, cx + w//2, cy + h//2], radius=12, fill=base_rgb, outline=(45, 45, 50), width=2)
        draw.rounded_rectangle([cx - w//2 + 8, cy - h//2 + 8, cx + w//2 - 8, cy + h//2 - 8], radius=8, fill=(16, 18, 22), outline=(35, 35, 40))
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(205, 205, 210))

    elif "keys" in desc_l or "keychain" in desc_l:
        # Metallic Keyring & Keys
        draw.ellipse([cx - 25, cy - 40, cx + 25, cy + 10], outline=(185, 185, 190), width=5)
        draw.rectangle([cx - 6, cy, cx + 6, cy + 65], fill=(215, 180, 55), outline=(95, 75, 20))
        draw.rectangle([cx + 4, cy + 45, cx + 14, cy + 52], fill=(215, 180, 55))
        draw.rectangle([cx + 4, cy + 56, cx + 12, cy + 63], fill=(215, 180, 55))
        draw.rectangle([cx - 18, cy - 10, cx - 8, cy + 50], fill=(195, 195, 200), outline=(55, 55, 60))

    else:
        # Calculator / Glasses / Book / General item
        w, h = 112, 132
        draw.rounded_rectangle([cx - w//2 + shadow_offset, cy - h//2 + shadow_offset, cx + w//2 + shadow_offset, cy + h//2 + shadow_offset], radius=14, fill=(105, 95, 90))
        draw.rounded_rectangle([cx - w//2, cy - h//2, cx + w//2, cy + h//2], radius=14, fill=base_rgb, outline=(35, 35, 40), width=2)
        draw.rectangle([cx - w//2 + 12, cy - h//2 + 15, cx + w//2 - 12, cy - h//2 + 45], fill=(22, 26, 32))

    # Apply authentic camera augmentations (Owner photo vs Finder photo)
    if is_found:
        # Finder camera: slightly dimmer fluorescent indoor lighting, subtle rotation
        bright = ImageEnhance.Brightness(img)
        img = bright.enhance(0.93)
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.06)
        rot_angle = random.choice([-3, 3, -2, 2])
        img = img.rotate(rot_angle, resample=Image.BICUBIC, fillcolor=(215, 218, 222))
    else:
        # Owner photo: warmer daylight lighting
        bright = ImageEnhance.Brightness(img)
        img = bright.enhance(1.03)
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.02)
        rot_angle = random.choice([-1, 1])
        img = img.rotate(rot_angle, resample=Image.BICUBIC, fillcolor=(195, 165, 135))

    return img


def generate_all_images():
    print(f"[Image Synthesizer] Generating 100 images for {len(RAW_SEED_ITEMS)} campus items in '{IMAGES_DIR}'...")
    generated_count = 0
    
    for i, item in enumerate(RAW_SEED_ITEMS):
        pair_id = item["pair_id"]
        cat = item["category"]
        brand = item["brand"]
        color = item["color"]
        
        # 1. Lost item image
        lost_img = generate_single_item_image(
            pair_id=pair_id,
            category=cat,
            brand=brand,
            color_name=color,
            desc=item["lost_desc"],
            is_found=False,
            seed_val=i * 2 + 101
        )
        lost_path = os.path.join(IMAGES_DIR, f"lost_{pair_id}.jpg")
        lost_img.save(lost_path, "JPEG", quality=90)
        generated_count += 1
        
        # 2. Found item image
        found_img = generate_single_item_image(
            pair_id=pair_id,
            category=cat,
            brand=brand,
            color_name=color,
            desc=item["found_desc"],
            is_found=True,
            seed_val=i * 2 + 102
        )
        found_path = os.path.join(IMAGES_DIR, f"found_{pair_id}.jpg")
        found_img.save(found_path, "JPEG", quality=90)
        generated_count += 1

    print(f"[Image Synthesizer] Finished! Created {generated_count} physical images in '{IMAGES_DIR}'.")


if __name__ == "__main__":
    generate_all_images()
