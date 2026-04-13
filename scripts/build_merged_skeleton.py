#!/usr/bin/env python3
"""
Build the merged skeleton by applying scene_data pointer patching.
Traces the full animation chain: AnimJoint → rest-pose → gap joints → model skeleton.

Usage: python3 scripts/build_merged_skeleton.py <zoid_id>
"""
import struct, sys, json, os
from pathlib import Path


def build_merged_skeleton(dat_path, b_dat_path):
    with open(dat_path, 'rb') as f: dat = f.read()
    with open(b_dat_path, 'rb') as f: bdat = f.read()

    ddo = 0x20; bdo = 0x20
    ddbs = struct.unpack('>I', dat[4:8])[0]
    bdbs = struct.unpack('>I', bdat[4:8])[0]
    du32 = lambda off: struct.unpack('>I', dat[ddo+off:ddo+off+4])[0]
    df32 = lambda off: struct.unpack('>f', dat[ddo+off:ddo+off+4])[0]
    bu32 = lambda off: struct.unpack('>I', bdat[bdo+off:bdo+off+4])[0]
    bf32 = lambda off: struct.unpack('>f', bdat[bdo+off:bdo+off+4])[0]

    # --- Model names ---
    drc = struct.unpack('>I', dat[8:12])[0]
    droots = struct.unpack('>I', dat[12:16])[0]
    dro = ddo + ddbs + drc * 4
    dto = dro + droots * 8
    name_map = {}
    for i in range(droots):
        roff = struct.unpack('>I', dat[dro+i*8:dro+i*8+4])[0]
        soff = struct.unpack('>I', dat[dro+i*8+4:dro+i*8+8])[0]
        ss = dto + soff; se = dat.index(0, ss) if 0 in dat[ss:ss+64] else ss+32
        name_map[roff] = dat[ss:se].decode('ascii','replace').rstrip('\x00')

    # --- Find scene_data in both files ---
    m_sd = None
    for i in range(droots):
        roff = struct.unpack('>I', dat[dro+i*8:dro+i*8+4])[0]
        soff = struct.unpack('>I', dat[dro+i*8+4:dro+i*8+8])[0]
        ss = dto + soff; se = dat.index(0, ss)
        if 'scene_data' in dat[ss:se].decode('ascii','replace'):
            m_sd = roff; break

    brc = struct.unpack('>I', bdat[8:12])[0]
    bro = bdo + bdbs + brc * 4
    b_sd = struct.unpack('>I', bdat[bro:bro+4])[0]

    # --- Find master tree ---
    p0 = du32(m_sd); p0_0 = du32(p0); master = du32(p0_0)

    # --- Build model master tree DFS index ---
    model_dfs = []  # list of offsets in DFS order
    model_visited = set()
    def walk_model(off):
        if off == 0 or off in model_visited or off >= ddbs: return
        model_visited.add(off)
        model_dfs.append(off)
        walk_model(du32(off + 0x08))
        walk_model(du32(off + 0x0C))
    walk_model(master)
    model_dfs_idx = {off: i for i, off in enumerate(model_dfs)}

    # --- Read scene_data mapping table ---
    CHILD_OFF = 0x08
    SIB_OFF = 0x0C

    # Collect all LINK entries: (rest_joint_idx, child_or_sib, model_target_offset)
    links = []
    i = 0
    while True:
        b_val = bu32(b_sd + (i+4)*4)
        m_val = du32(m_sd + (i+4)*4)
        if b_val == 0 and m_val == 0:
            # Check if we've hit the end
            zeros = sum(1 for j in range(5) if bu32(b_sd+(i+4+j)*4)==0 and du32(m_sd+(i+4+j)*4)==0)
            if zeros >= 4: break

        b_joint = b_val // 0x40
        b_field = b_val % 0x40

        if b_field in [CHILD_OFF, SIB_OFF]:
            # Follow indirection chain in model to find master tree joint
            target = m_val
            visited_chain = set()
            resolved = None
            for _ in range(10):  # max 10 levels of indirection
                if target in visited_chain: break
                visited_chain.add(target)
                if target >= ddbs: break
                val = du32(target)
                if val in model_dfs_idx:
                    resolved = val
                    break
                elif 0 < val < ddbs:
                    target = val  # follow indirection
                else:
                    break

            if resolved is not None:
                link_type = 'child' if b_field == CHILD_OFF else 'sibling'
                links.append({
                    'rest_joint': b_joint,
                    'link_type': link_type,
                    'model_offset': resolved,
                    'model_dfs_idx': model_dfs_idx[resolved],
                    'model_name': name_map.get(resolved, ''),
                })

        i += 1
        if i > 500: break

    # --- Build rest-pose tree DFS ---
    rest_dfs = []
    rest_visited = set()
    def walk_rest(off, is_root=False):
        if off in rest_visited or off >= bdbs: return
        if not is_root and off == 0: return
        rest_visited.add(off)
        rest_dfs.append(off)
        walk_rest(bu32(off + 0x08))
        walk_rest(bu32(off + 0x0C))
    walk_rest(0, True)

    # --- Build the mapping: anim joint index → model joint name ---
    # The AnimJoint clips have the same count as rest_dfs (25 for a01)
    # Each AnimJoint[i] drives rest_dfs[i]
    # Through the hierarchy, rest_dfs joints are parents of gap joints,
    # which connect to model skeleton joints via links.

    # For each link, trace WHICH rest-pose joint (0-24) is the ancestor
    # that drives this connection
    rest_joint_to_link = {}  # rest_joint_idx → list of linked model joints
    for link in links:
        rj = link['rest_joint']
        if rj not in rest_joint_to_link:
            rest_joint_to_link[rj] = []
        rest_joint_to_link[rj].append(link)

    return {
        'rest_pose_count': len(rest_dfs),
        'model_joint_count': len(model_dfs),
        'links': links,
        'rest_joint_to_model': rest_joint_to_link,
        'model_names': {model_dfs_idx[off]: name_map.get(off, '') for off in model_dfs if off in name_map},
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <zoid_id>")
        sys.exit(1)

    zoid_id = sys.argv[1]
    base = Path(__file__).parent.parent
    dat_path = base / "dat" / "vs3" / "zoid" / f"{zoid_id}.dat"
    b_dat_path = base / "iso_dump" / "files" / "zoid" / f"{zoid_id}_b.dat"

    if not dat_path.exists():
        print(f"Model not found: {dat_path}")
        sys.exit(1)
    if not b_dat_path.exists():
        print(f"Animation not found: {b_dat_path}")
        sys.exit(1)

    result = build_merged_skeleton(str(dat_path), str(b_dat_path))

    print(f"Rest-pose joints: {result['rest_pose_count']}")
    print(f"Model joints: {result['model_joint_count']}")
    print(f"Links found: {len(result['links'])}")
    print()

    print("Links (rest-pose gap joints → model skeleton):")
    for link in result['links']:
        name = link['model_name'] or f"model[{link['model_dfs_idx']}]"
        print(f"  rest[{link['rest_joint']:2d}].{link['link_type']:>7s} → {name}")

    print()
    print("Rest-pose joints with linked model joints:")
    for rj, links_list in sorted(result['rest_joint_to_model'].items()):
        names = [l['model_name'] or f"model[{l['model_dfs_idx']}]" for l in links_list]
        in_clip = rj < result['rest_pose_count']
        marker = '(ANIMATED)' if in_clip else '(gap)'
        print(f"  rest[{rj:2d}] {marker}: {', '.join(names)}")

    # Save to data/
    out_path = base / "data" / f"{zoid_id}_skeleton_map.json"
    with open(str(out_path), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
