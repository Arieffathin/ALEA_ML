
import os
import shutil

# Path dataset pegon
pegon_path = "huruf_pegon"  # ganti dengan path ekstraksi jika diperlukan
output_path = "kelompokkan_berdasarkan_awalan"

# Pastikan output path ada
os.makedirs(output_path, exist_ok=True)

# Mapping huruf
pegon_map = {
    "a": "ا", "b": "ب", "t": "ت", "ts": "ث", "j": "ج", "h": "ح", "kh": "خ",
    "d": "د", "dz": "ذ", "r": "ر", "z": "ز", "s": "س", "sy": "ش", "sh": "ص",
    "dh": "ض", "th": "ط", "zh": "ظ", "ain": "ع", "gh": "غ", "f": "ف", "q": "ق",
    "k": "ك", "l": "ل", "m": "م", "n": "ن", "w": "و", "h2": "ه", "hamzah": "ء",
    "y": "ي", "ng": "ڠ", "ny": "ڽ", "p": "ڤ", "g": "ݢ"
}

# Urutkan prefix agar yang lebih panjang seperti "ain_" tidak ditangkap oleh "a_"
sorted_prefixes = sorted(pegon_map.keys(), key=lambda x: -len(x))

# Loop semua file
for root, dirs, files in os.walk(pegon_path):
    for file in files:
        for prefix in sorted_prefixes:
            if file.startswith(prefix + "_"):
                folder_path = os.path.join(output_path, prefix)
                os.makedirs(folder_path, exist_ok=True)
                src = os.path.join(root, file)
                dst = os.path.join(folder_path, file)
                shutil.copy(src, dst)
                break
