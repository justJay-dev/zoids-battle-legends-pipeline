#!/usr/bin/env python3
"""Export animation clips from _b.dat files to JSON for browser playback."""
import struct, sys, os, json
from pathlib import Path

TRACK_NAMES = {
    1: "ROTX",
    2: "ROTY",
    3: "ROTZ",
    4: "PATH",
    5: "TRAX",
    6: "TRAY",
    7: "TRAZ",
    8: "SCAX",
    9: "SCAY",
    10: "SCAZ",
}


class BufReader:
    def __init__(self, b):
        self.b = b
        self.pos = 0

    def read_packed(self):
        val = shift = 0
        while True:
            byte = self.b[self.pos]
            self.pos += 1
            val |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        return val

    def read_val(self, fmt, scale):
        import math

        if fmt == 0:
            v = struct.unpack(">f", self.b[self.pos : self.pos + 4])[0]
            self.pos += 4
        elif fmt == 1:
            v = struct.unpack(">h", self.b[self.pos : self.pos + 2])[0]
            self.pos += 2
            v = v / scale
        elif fmt == 2:
            v = struct.unpack(">H", self.b[self.pos : self.pos + 2])[0]
            self.pos += 2
            v = v / scale
        elif fmt == 3:
            v = struct.unpack(">b", self.b[self.pos : self.pos + 1])[0]
            self.pos += 1
            v = v / scale
        elif fmt == 4:
            v = self.b[self.pos]
            self.pos += 1
            v = v / scale
        else:
            v = 0
        if math.isnan(v) or math.isinf(v) or abs(v) > 100:
            return None
        return v


def decode_fobj(buf_bytes, vfmt, vscale, tfmt, tscale, start_frame):
    reader = BufReader(buf_bytes)
    keys = []
    clock = start_frame
    try:
        while reader.pos < len(reader.b):
            packed = reader.read_packed()
            interp = packed & 0x0F
            num_keys = (packed >> 4) + 1
            if interp == 0:
                break
            for _ in range(num_keys):
                if interp in [1, 2, 3]:  # CON, LIN, SPL0
                    value = reader.read_val(vfmt, vscale)
                    time = reader.read_packed()
                    if value is not None:
                        k = {"f": round(clock, 3), "v": round(value, 6)}
                        if interp == 1:
                            k["i"] = "C"
                        elif interp == 3:
                            k["i"] = "S0"
                        keys.append(k)
                    clock += time
                elif interp == 4:  # SPL
                    value = reader.read_val(vfmt, vscale)
                    tangent = reader.read_val(tfmt, tscale)
                    time = reader.read_packed()
                    if value is not None:
                        k = {
                            "f": round(clock, 3),
                            "v": round(value, 6),
                            "i": "S",
                        }
                        if tangent is not None:
                            k["t"] = round(tangent, 6)
                        keys.append(k)
                    clock += time
                elif interp == 5:  # SLP
                    tangent = reader.read_val(tfmt, tscale)
                    if tangent is not None:
                        keys.append(
                            {"f": round(clock, 3), "t": round(tangent, 6), "i": "T"}
                        )
                elif interp == 6:  # KEY
                    value = reader.read_val(vfmt, vscale)
                    if value is not None:
                        keys.append({"f": round(clock, 3), "v": round(value, 6), "i": "K"})
    except (IndexError, struct.error):
        pass
    return keys


def parse_b_dat(filepath):
    with open(filepath, "rb") as f:
        data = f.read()

    data_offset = 0x20
    u32 = lambda off: struct.unpack(
        ">I", data[data_offset + off : data_offset + off + 4]
    )[0]
    f32 = lambda off: struct.unpack(
        ">f", data[data_offset + off : data_offset + off + 4]
    )[0]

    file_size = struct.unpack(">I", data[0:4])[0]
    data_block_size = struct.unpack(">I", data[4:8])[0]
    reloc_count = struct.unpack(">I", data[8:12])[0]

    # Find clip table via pointer chain:
    # scene_data → ptr0 → [0] → +0x04 = clip table address
    root_offset_abs = data_offset + data_block_size + reloc_count * 4
    scene_data_off = struct.unpack(">I", data[root_offset_abs : root_offset_abs + 4])[0]

    ptr0 = u32(scene_data_off)  # first pointer in scene_data
    ptr0_target = u32(ptr0)  # where ptr0 points
    clip_table_addr = u32(ptr0_target + 4)  # +0x04 = clip table pointer

    clip_roots = []
    off = clip_table_addr
    while off < data_block_size:
        val = u32(off)
        if val == 0 or val >= data_block_size:
            break
        clip_roots.append(val)
        off += 4

    if not clip_roots:
        return None

    # Parse rest-pose skeleton (JointObjDesc at 0x40 stride, starting at data+0x00)
    # Output includes tree structure (children indices) for topology matching
    rest_joints = []
    rest_visited = set()
    rest_index_map = {}  # offset → index

    def parse_rest_joint(off, is_root=False):
        if off in rest_visited or off >= data_block_size:
            return -1
        if not is_root and off == 0:
            return -1
        rest_visited.add(off)
        idx = len(rest_joints)
        rest_index_map[off] = idx
        child_off = u32(off + 0x08)
        sib_off = u32(off + 0x0C)
        rx, ry, rz = f32(off + 0x14), f32(off + 0x18), f32(off + 0x1C)
        sx, sy, sz = f32(off + 0x20), f32(off + 0x24), f32(off + 0x28)
        tx, ty, tz = f32(off + 0x2C), f32(off + 0x30), f32(off + 0x34)
        joint = {
            "rotation": [round(rx, 6), round(ry, 6), round(rz, 6)],
            "scale": [round(sx, 6), round(sy, 6), round(sz, 6)],
            "translation": [round(tx, 6), round(ty, 6), round(tz, 6)],
            "children": [],
        }
        rest_joints.append(joint)
        # Parse children (child first, then siblings become parent's children)
        child_idx = parse_rest_joint(child_off)
        if child_idx >= 0:
            joint["children"].append(child_idx)
        # Siblings at same level
        sib_off_cur = sib_off
        while (
            sib_off_cur != 0
            and sib_off_cur not in rest_visited
            and sib_off_cur < data_block_size
        ):
            sib_idx = parse_rest_joint(sib_off_cur)
            if sib_idx >= 0:
                joint["children"].append(sib_idx)
            # advance to next sibling (already parsed inside parse_rest_joint)
            break  # parse_rest_joint handles recursion
        return idx

    parse_rest_joint(0, is_root=True)

    # Fix children: re-walk with proper child/sibling separation
    for j in rest_joints:
        j["children"] = []
    rest_visited2 = set()
    rest_roots = []

    def fix_rest_children(off, parent_idx, is_root=False):
        if off in rest_visited2 or off >= data_block_size:
            return
        if not is_root and off == 0:
            return
        rest_visited2.add(off)
        idx = rest_index_map.get(off)
        if idx is None:
            return
        if parent_idx >= 0:
            rest_joints[parent_idx]["children"].append(idx)
        else:
            rest_roots.append(idx)
        child_off = u32(off + 0x08)
        sib_off = u32(off + 0x0C)
        fix_rest_children(child_off, idx)
        fix_rest_children(sib_off, parent_idx)

    fix_rest_children(0, -1, is_root=True)

    # Parse clips
    clips = []
    for clip_idx, root in enumerate(clip_roots):
        joints = []

        def traverse(off, depth=0):
            if off == 0:
                return
            child = u32(off)
            nxt = u32(off + 4)
            aobj_ptr = u32(off + 8)
            joint = {"tracks": {}}
            if aobj_ptr != 0:
                flags = u32(aobj_ptr)
                end_frame = f32(aobj_ptr + 4)
                fd = u32(aobj_ptr + 8)
                joint["endFrame"] = round(end_frame, 3)
                joint["loop"] = bool(flags & (1 << 29))
                while fd != 0:
                    nxt_fd = u32(fd)
                    data_len = u32(fd + 4)
                    start_frame = f32(fd + 8)
                    track_type = data[data_offset + fd + 0x0C]
                    value_flag = data[data_offset + fd + 0x0D]
                    tan_flag = data[data_offset + fd + 0x0E]
                    buf_ptr = u32(fd + 0x10)
                    vfmt = (value_flag >> 5) & 7
                    vscale = 1 << (value_flag & 0x1F)
                    tfmt = (tan_flag >> 5) & 7
                    tscale = 1 << (tan_flag & 0x1F)
                    buf_bytes = data[
                        data_offset + buf_ptr : data_offset + buf_ptr + data_len
                    ]
                    keys = decode_fobj(
                        buf_bytes, vfmt, vscale, tfmt, tscale, start_frame
                    )
                    track_name = TRACK_NAMES.get(track_type, f"T{track_type}")
                    if keys:
                        joint["tracks"][track_name] = keys
                    fd = nxt_fd
            joints.append(joint)
            traverse(child, depth + 1)
            traverse(nxt, depth)

        traverse(root)

        # Determine frame count
        end_frames = [j["endFrame"] for j in joints if "endFrame" in j]
        frame_count = max(end_frames) if end_frames else 0

        clips.append(
            {
                "index": clip_idx,
                "frameCount": round(frame_count, 3),
                "joints": joints,
            }
        )

    # Fix TRAX mirroring for L/R joint pairs.
    # Some zoids have R-side TRAX values with wrong sign (encoding issue).
    # Detect by comparing L/R pairs and mirror L's delta to correct R.
    def fix_trax_mirror(clips, rest_joints):
        # Find L/R pairs by matching |X| with opposite signs
        pairs = []  # (l_idx, r_idx)
        for i in range(len(rest_joints)):
            xi = rest_joints[i]["translation"][0]
            if xi <= 0.5:
                continue
            # Find mirror partner (negative X, same Y/Z)
            for j in range(len(rest_joints)):
                xj = rest_joints[j]["translation"][0]
                if xj >= -0.5:
                    continue
                if abs(xi + xj) < 0.05:  # |X| matches
                    yi, zi = rest_joints[i]["translation"][1], rest_joints[i]["translation"][2]
                    yj, zj = rest_joints[j]["translation"][1], rest_joints[j]["translation"][2]
                    if abs(yi - yj) < 0.05 and abs(zi - zj) < 0.05:
                        pairs.append((i, j))
                        break

        if not pairs:
            return 0

        fixed = 0
        for clip in clips:
            for li, ri in pairs:
                if li >= len(clip["joints"]) or ri >= len(clip["joints"]):
                    continue
                l_trax = clip["joints"][li].get("tracks", {}).get("TRAX", [])
                r_trax = clip["joints"][ri].get("tracks", {}).get("TRAX", [])
                if not l_trax or not r_trax:
                    continue

                l_rest = rest_joints[li]["translation"][0]
                r_rest = rest_joints[ri]["translation"][0]

                # Check if R values have wrong sign by testing first key
                l_val = next((k["v"] for k in l_trax if "v" in k), None)
                r_val = next((k["v"] for k in r_trax if "v" in k), None)
                if l_val is None or r_val is None:
                    continue

                l_delta = l_val - l_rest
                expected_r = -l_delta + r_rest

                # If actual R is far from expected (wrong sign), fix ALL keys
                if abs(r_val - expected_r) > abs(r_rest) * 0.5:
                    for k in r_trax:
                        if "v" in k:
                            # Mirror: R_corrected = -(L_value - L_rest) + R_rest
                            # But L might vary per frame while R is constant...
                            # Simpler: negate the delta from R's rest
                            # Use the ratio: R_corrected = R_rest - (R_val - R_rest) = 2*R_rest - R_val
                            # Actually: compute from L counterpart at same frame
                            pass

                    # Better approach: find L key at same frame and mirror
                    l_by_frame = {k["f"]: k["v"] for k in l_trax if "v" in k}
                    for k in r_trax:
                        if "v" not in k:
                            continue
                        # Find nearest L key
                        l_v = l_by_frame.get(k["f"])
                        if l_v is None:
                            # Interpolate from L keys
                            l_v = l_val  # fallback to first
                        k["v"] = round(-(l_v - l_rest) + r_rest, 6)
                    fixed += 1

        return fixed

    n_fixed = fix_trax_mirror(clips, rest_joints)

    # Load action names
    action_list_path = Path(__file__).parent.parent / "data" / "action_list.json"
    action_names = {}
    if action_list_path.exists():
        with open(action_list_path) as f:
            action_names = json.load(f)

    # Add action names to clips
    for clip in clips:
        clip["action"] = action_names.get(str(clip["index"]), f"action_{clip['index']}")

    return {
        "restPose": rest_joints,
        "restRoots": rest_roots,
        "clipCount": len(clips),
        "clips": clips,
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file_b.dat> [output_dir]")
        print(f"  Exports animation clips to JSON")
        sys.exit(1)

    filepath = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(outdir, exist_ok=True)

    basename = Path(filepath).stem.replace("_b", "")
    result = parse_b_dat(filepath)

    if not result:
        print(f"No animations found in {filepath}")
        return

    # Filter to only multi-frame clips for the main export
    multi_frame = [c for c in result["clips"] if c["frameCount"] > 1]
    single_frame = [c for c in result["clips"] if c["frameCount"] <= 1]

    print(
        f"{basename}: {len(result['clips'])} total clips, "
        f"{len(multi_frame)} animated, {len(single_frame)} poses, "
        f"{len(result['restPose'])} rest joints"
    )

    # Export full data
    out_path = os.path.join(outdir, f"{basename}_anims.json")
    # Strip single-frame clips to save space — keep only multi-frame
    export = {
        "restPose": result["restPose"],
        "restRoots": result["restRoots"],
        "clips": multi_frame,
    }
    with open(out_path, "w") as f:
        json.dump(export, f, separators=(",", ":"))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  → {out_path} ({size_kb:.0f}KB, {len(multi_frame)} clips)")


if __name__ == "__main__":
    main()
