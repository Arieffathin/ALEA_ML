import os
from PIL import Image, ImageDraw, ImageFont


characters = {
    "a": "ا", "b": "ب", "t": "ت", "ts": "ث", "j": "ج", "h": "ح", "kh": "خ",
    "d": "د", "dz": "ذ", "r": "ر", "z": "ز", "s": "س", "sy": "ش", "sh": "ص",
    "dh": "ض", "th": "ط", "zh": "ظ", "ain": "ع", "gh": "غ", "f": "ف", "q": "ق",
    "k": "ك", "l": "ل", "m": "م", "n": "ن", "w": "و", "h2": "ه", "hamzah": "ء",
    "y": "ي", "ng": "ڠ", "ny": "ڽ", "p": "ڤ", "g": "ݢ"
}


output_dir = "huruf_pegon"
os.makedirs(output_dir, exist_ok=True)


fonts = ["Lateef-Regular.ttf", "Scheherazade-Regular.ttf", "Amiri-Regular.ttf"]
font_sizes = [100, 120, 140]
bg_colors = ["white", "#fff5cc", "#fdfdf5"]  
text_colors = ["black", "#003366", "#4b2e2e"] 

image_size = (300, 300)
counter = 0

for label, char in characters.items():
    for font_name in fonts:
        for size in font_sizes:
            for bg in bg_colors:
                for fg in text_colors:
                    try:
                        font = ImageFont.truetype(font_name, size)
                    except OSError:
                        print(f"⚠️ Font '{font_name}' tidak ditemukan. Lewatkan.")
                        continue

                    img = Image.new("RGB", image_size, bg)
                    draw = ImageDraw.Draw(img)

                    bbox = draw.textbbox((0, 0), char, font=font)
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    x = (image_size[0] - text_w) // 2 - bbox[0]
                    y = (image_size[1] - text_h) // 2 - bbox[1]

                    draw.text((x, y), char, font=font, fill=fg)

                    filename = f"{label}_{counter}.jpg"
                    img.save(os.path.join(output_dir, filename))
                    counter += 1

print(f"✅ Dataset huruf Pegon selesai. Total gambar: {counter}")
print(f"📁 Disimpan di folder: {output_dir}")
