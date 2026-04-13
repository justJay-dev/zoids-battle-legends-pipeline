#!/usr/bin/env python3
"""Decode CMPR (GX DXT1) textures from Zoids Battle Legends .dat files → PNG."""
import struct, sys, os
from pathlib import Path


def rgb565_to_rgb(val):
    r = ((val >> 11) & 0x1F) * 255 // 31
    g = ((val >> 5) & 0x3F) * 255 // 63
    b = (val & 0x1F) * 255 // 31
    return (r, g, b, 255)


def decode_dxt1_block(data, offset):
    """Decode one 4x4 DXT1 sub-block (8 bytes) → 16 RGBA tuples."""
    c0_raw = struct.unpack_from(">H", data, offset)[0]  # big-endian on GC
    c1_raw = struct.unpack_from(">H", data, offset + 2)[0]
    c0 = rgb565_to_rgb(c0_raw)
    c1 = rgb565_to_rgb(c1_raw)

    if c0_raw > c1_raw:
        c2 = tuple((2 * c0[i] + c1[i]) // 3 for i in range(3)) + (255,)
        c3 = tuple((c0[i] + 2 * c1[i]) // 3 for i in range(3)) + (255,)
    else:
        c2 = tuple((c0[i] + c1[i]) // 2 for i in range(3)) + (255,)
        c3 = (0, 0, 0, 0)  # transparent

    colors = [c0, c1, c2, c3]
    pixels = []
    for byte_idx in range(4):
        byte = data[offset + 4 + byte_idx]
        for bit_idx in range(4):
            idx = (byte >> (6 - bit_idx * 2)) & 0x3
            pixels.append(colors[idx])
    return pixels


def decode_cmpr(data, offset, width, height):
    """Decode CMPR texture → flat list of RGBA tuples, row by row."""
    img = [(0, 0, 0, 0)] * (width * height)
    pos = offset

    # Iterate 8x8 macro-blocks
    for macro_y in range(0, height, 8):
        for macro_x in range(0, width, 8):
            # Each macro-block = 4 DXT1 sub-blocks (4x4 each), arranged 2x2
            for sub_idx in range(4):
                sub_x = macro_x + (sub_idx % 2) * 4
                sub_y = macro_y + (sub_idx // 2) * 4
                pixels = decode_dxt1_block(data, pos)
                pos += 8
                # Place 4x4 pixels
                for py in range(4):
                    for px in range(4):
                        ix = sub_x + px
                        iy = sub_y + py
                        if ix < width and iy < height:
                            img[iy * width + ix] = pixels[py * 4 + px]
    return img


def parse_dat_textures(filepath):
    """Find all ImageObjDesc in a DAT file, return list of (width, height, type, data_offset)."""
    with open(filepath, "rb") as f:
        data = f.read()

    u32 = lambda off: struct.unpack_from(">I", data, off)[0]
    u16 = lambda off: struct.unpack_from(">H", data, off)[0]

    dataBlockSize = u32(0x04)
    relocCount = u32(0x08)
    rootCount = u32(0x0C)
    dataOffset = 0x20
    relocOffset = dataOffset + dataBlockSize
    rootOffset = relocOffset + relocCount * 4
    tableOffset = rootOffset + rootCount * 8

    # Collect root names + offsets
    roots = []
    for i in range(rootCount):
        off = rootOffset + i * 8
        roff = u32(off)
        soff = u32(off + 4)
        strStart = tableOffset + soff
        strEnd = (
            data.index(0, strStart)
            if 0 in data[strStart : strStart + 64]
            else strStart + 32
        )
        name = data[strStart:strEnd].decode("ascii", errors="replace")
        roots.append((roff, name))

    # Walk joint tree recursively, collect ImageObjDesc entries
    visited = set()
    textures = []  # (width, height, image_type, data_offset_in_file)
    seen_img_offsets = set()

    def follow_pointer(off):
        """Read a struct at data-relative offset, follow known pointer chains."""
        if off == 0 or off in visited:
            return
        visited.add(off)

    def walk_joint(off):
        if off == 0 or off in visited:
            return
        visited.add(off)
        base = off + dataOffset
        child = u32(base + 0x08)
        sibling = u32(base + 0x0C)
        display = u32(base + 0x10)
        walk_display(display)
        walk_joint(child)
        walk_joint(sibling)

    def walk_display(off):
        if off == 0 or off in visited:
            return
        visited.add(off)
        base = off + dataOffset
        sibling = u32(base + 0x04)
        material = u32(base + 0x08)
        polygon = u32(base + 0x0C)
        walk_material(material)
        walk_display(sibling)

    def walk_material(off):
        if off == 0 or off in visited:
            return
        visited.add(off)
        base = off + dataOffset
        texture = u32(base + 0x08)
        walk_texture(texture)

    def walk_texture(off):
        if off == 0 or off in visited:
            return
        visited.add(off)
        base = off + dataOffset
        sibling = u32(base + 0x04)
        image_header = u32(base + 0x4C)  # Image_Header_Pointer
        walk_image(image_header)
        walk_texture(sibling)

    def walk_image(off):
        if off == 0 or off in visited:
            return
        visited.add(off)
        base = off + dataOffset
        img_data_ptr = u32(base + 0x00)
        width = u16(base + 0x04)
        height = u16(base + 0x06)
        img_type = u32(base + 0x08)

        if img_data_ptr not in seen_img_offsets:
            seen_img_offsets.add(img_data_ptr)
            textures.append((width, height, img_type, img_data_ptr + dataOffset))

    # Walk from root table joints
    for roff, name in roots:
        if "scene_data" in name:
            continue
        walk_joint(roff)

    # Also walk from master tree via scene_data (finds weapon textures)
    for roff, name in roots:
        if "scene_data" not in name:
            continue
        p0 = u32(roff + dataOffset)
        if p0 == 0 or p0 >= dataBlockSize:
            continue
        p0_0 = u32(p0 + dataOffset)
        if p0_0 == 0 or p0_0 >= dataBlockSize:
            continue
        master = u32(p0_0 + dataOffset)
        if master and master < dataBlockSize:
            walk_joint(master)

    return data, textures


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.dat> [output_dir]")
        print(f"  Decodes CMPR textures from a Zoids .dat file to PNG")
        sys.exit(1)

    filepath = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(outdir, exist_ok=True)

    basename = Path(filepath).stem
    data, textures = parse_dat_textures(filepath)

    if not textures:
        print("No textures found.")
        return

    print(f"Found {len(textures)} unique texture(s)")

    # Try importing PIL for PNG output
    try:
        from PIL import Image
    except ImportError:
        print("pip install Pillow for PNG output")
        print("Dumping raw RGBA instead...")
        for i, (w, h, t, off) in enumerate(textures):
            print(f"  [{i}] {w}x{h} type={t} offset=0x{off:X}")
            if t == 14:  # CMPR
                pixels = decode_cmpr(data, off, w, h)
                raw_path = os.path.join(outdir, f"{basename}_tex{i}_{w}x{h}.rgba")
                with open(raw_path, "wb") as f:
                    for px in pixels:
                        f.write(bytes(px))
                print(f"    → {raw_path}")
        return

    for i, (w, h, t, off) in enumerate(textures):
        print(f"  [{i}] {w}x{h} type={t} offset=0x{off:X}")
        if t != 14:
            print(f"    Skipping non-CMPR type {t}")
            continue

        pixels = decode_cmpr(data, off, w, h)
        img = Image.new("RGBA", (w, h))
        img.putdata(pixels)
        out_path = os.path.join(outdir, f"{basename}_tex{i}_{w}x{h}.png")
        img.save(out_path)
        print(f"    → {out_path}")


if __name__ == "__main__":
    main()
