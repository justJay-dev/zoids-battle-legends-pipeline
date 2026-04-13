#!/usr/bin/env python3
"""Decode GC TPL texture files → PNG. Uses the same CMPR decoder as decode_cmpr.py."""
import struct, sys, os
from pathlib import Path

# Import CMPR decoder from sibling script
sys.path.insert(0, str(Path(__file__).parent))
from decode_cmpr import decode_cmpr, rgb565_to_rgb


def decode_tpl(filepath):
    """Parse TPL file, return list of (width, height, rgba_pixels) tuples."""
    with open(filepath, "rb") as f:
        data = f.read()

    magic = struct.unpack(">I", data[0:4])[0]
    if magic != 0x0020AF30:
        return []

    num_images = struct.unpack(">I", data[4:8])[0]
    table_off = struct.unpack(">I", data[8:12])[0]

    results = []
    for i in range(num_images):
        img_off = struct.unpack(">I", data[table_off + i*8:table_off + i*8 + 4])[0]
        if img_off == 0:
            continue
        h = struct.unpack(">H", data[img_off:img_off+2])[0]
        w = struct.unpack(">H", data[img_off+2:img_off+4])[0]
        fmt = struct.unpack(">I", data[img_off+4:img_off+8])[0]
        data_off = struct.unpack(">I", data[img_off+8:img_off+12])[0]

        if fmt == 14:  # CMPR
            pixels = decode_cmpr(data, data_off, w, h)
            results.append((w, h, pixels))
        else:
            print(f"  Skipping image {i}: format {fmt} not supported (only CMPR)")

    return results


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.tpl> [output_dir]")
        sys.exit(1)

    filepath = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(outdir, exist_ok=True)

    basename = Path(filepath).stem
    images = decode_tpl(filepath)

    if not images:
        print(f"No images in {filepath}")
        return

    try:
        from PIL import Image
    except ImportError:
        print("pip install Pillow for PNG output")
        return

    for i, (w, h, pixels) in enumerate(images):
        img = Image.new("RGBA", (w, h))
        img.putdata(pixels)
        suffix = f"_{i}" if len(images) > 1 else ""
        out_path = os.path.join(outdir, f"{basename}{suffix}.png")
        img.save(out_path)
        print(f"  {basename}{suffix}.png ({w}x{h})")


if __name__ == "__main__":
    main()
