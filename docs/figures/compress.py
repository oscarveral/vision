from pathlib import Path
from PIL import Image

#!/usr/bin/env python3

QUALITY = 90

def convert_png_to_jpeg(png_path: Path, quality: int = QUALITY):
    img = Image.open(png_path)
    # Ensure we have an alpha-aware image to detect transparency
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # Paste onto white background to remove alpha
        background = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.convert("RGBA").split()[-1]
        background.paste(img.convert("RGBA"), mask=alpha)
        rgb = background
    else:
        rgb = img.convert("RGB")

    out_path = png_path.with_suffix(".jpg")
    rgb.save(out_path, "JPEG", quality=quality, optimize=True, progressive=True)
    return out_path

def main():
    folder = Path(__file__).resolve().parent
    png_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    if not png_files:
        print("No PNG files found in", folder)
        return

    for p in sorted(png_files):
        try:
            out = convert_png_to_jpeg(p)
            print(f"Converted: {p.name} -> {out.name}")
        except Exception as e:
            print(f"Failed to convert {p.name}: {e}")

if __name__ == "__main__":
    main()