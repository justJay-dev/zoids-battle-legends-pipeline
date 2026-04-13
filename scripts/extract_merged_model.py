#!/usr/bin/env python3
"""
Extract model with merged animation hierarchy.
Matches skeleton leaf joints to accessory joints by world-space position,
then reparents skeleton chains under the accessories they correspond to.

This creates the same merged hierarchy the game builds at runtime via
scene_data pointer patching, allowing animation on rest-pose joints (0-N)
to drive skeleton joints through transform inheritance.

Usage: python3 scripts/extract_merged_model.py <zoid_id> [output_dir]
"""
import struct, sys, os, json, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_model import extract_model


def compute_world_positions(joints, roots):
    """Compute world-space position of every joint."""
    world_pos = {}

    def trs_to_mat(t, r, s):
        cx, sx_ = math.cos(r[0]), math.sin(r[0])
        cy, sy_ = math.cos(r[1]), math.sin(r[1])
        cz, sz_ = math.cos(r[2]), math.sin(r[2])
        m = [0.0] * 16
        m[0] = cy*cz*s[0]; m[1] = cy*sz_*s[0]; m[2] = -sy_*s[0]
        m[4] = (sx_*sy_*cz - cx*sz_)*s[1]; m[5] = (sx_*sy_*sz_ + cx*cz)*s[1]; m[6] = sx_*cy*s[1]
        m[8] = (cx*sy_*cz + sx_*sz_)*s[2]; m[9] = (cx*sy_*sz_ - sx_*cz)*s[2]; m[10] = cx*cy*s[2]
        m[12] = t[0]; m[13] = t[1]; m[14] = t[2]; m[15] = 1.0
        return m

    def mat_mul(a, b):
        out = [0.0] * 16
        for col in range(4):
            for row in range(4):
                out[col*4+row] = sum(a[k*4+row]*b[col*4+k] for k in range(4))
        return out

    def walk(idx, parent_mat):
        j = joints[idx]
        local = trs_to_mat(j['translation'], j['rotation'], j['scale'])
        w = mat_mul(parent_mat, local)
        world_pos[idx] = (w[12], w[13], w[14])
        for c in j.get('children', []):
            walk(c, w)

    ident = [0.0]*16; ident[0] = ident[5] = ident[10] = ident[15] = 1.0
    for root in roots:
        walk(root, ident)
    return world_pos


def find_rest_pose_count(dat_path, b_dat_path):
    """Count rest-pose joints in _b.dat by walking its connected tree."""
    with open(b_dat_path, 'rb') as f:
        bdat = f.read()
    bdo = 0x20
    bdbs = struct.unpack('>I', bdat[4:8])[0]
    bu32 = lambda off: struct.unpack('>I', bdat[bdo+off:bdo+off+4])[0]

    visited = set()
    count = 0
    def walk(off, is_root=False):
        nonlocal count
        if off in visited or off >= bdbs: return
        if not is_root and off == 0: return
        visited.add(off)
        count += 1
        walk(bu32(off + 0x08))
        walk(bu32(off + 0x0C))
    walk(0, True)
    return count


def build_parent_map(joints):
    """Build child→parent index map."""
    parent = {}
    for i, j in enumerate(joints):
        for c in j.get('children', []):
            parent[c] = i
    return parent


def merge_hierarchy(joints, roots, rest_pose_count):
    """Reparent skeleton joints under matching accessory joints by position."""
    world_pos = compute_world_positions(joints, roots)
    parent_map = build_parent_map(joints)

    # Accessory joints = first rest_pose_count joints in DFS (indices 0..N-1)
    # Skeleton joints = everything else (indices N+)
    acc_indices = set(range(rest_pose_count))
    skel_indices = set(range(rest_pose_count, len(joints)))

    # Only match NAMED skeleton endpoint joints (FOOT, HAND, HEAD, TEAL)
    # to NAMED accessory joints. Ignore unnamed wrappers and body-core joints.
    ENDPOINT_KEYWORDS = ['FOOT', 'HAND', 'HEAD1']
    ACC_KEYWORDS = ['shoe', 'glove', 'collar', 'cap']

    skel_endpoints = {}
    for i in skel_indices:
        name = joints[i].get('name', '')
        if any(kw in name for kw in ENDPOINT_KEYWORDS):
            skel_endpoints[i] = name

    acc_targets = {}
    for i in acc_indices:
        name = joints[i].get('name', '')
        if any(kw in name for kw in ACC_KEYWORDS):
            acc_targets[i] = name

    # Match skeleton endpoints to accessories by world-space position
    MATCH_THRESHOLD = 0.15
    matches = []
    for si, sname in skel_endpoints.items():
        if si not in world_pos:
            continue
        sp = world_pos[si]
        # Skip joints near origin (ambiguous)
        if math.sqrt(sp[0]**2 + sp[1]**2 + sp[2]**2) < 0.5:
            continue
        best_ai = None
        best_dist = float('inf')
        for ai in acc_targets:
            if ai not in world_pos:
                continue
            ap = world_pos[ai]
            d = math.sqrt(sum((sp[k] - ap[k])**2 for k in range(3)))
            if d < best_dist:
                best_dist = d
                best_ai = ai
        if best_ai is not None and best_dist < MATCH_THRESHOLD:
            matches.append((si, best_ai, best_dist))

    if not matches:
        return joints, roots, []

    # For each match, find the highest skeleton ancestor that should be reparented.
    # Walk UP from matched skeleton joint until we hit a joint that has
    # siblings also in the skeleton (branching point — don't reparent above it).
    reparent_ops = []  # (subtree_root_idx, new_parent_acc_idx)
    already_reparented = set()

    for skel_idx, acc_idx, dist in matches:
        # Walk up from skel_idx to find the subtree root to reparent
        current = skel_idx
        while True:
            p = parent_map.get(current)
            if p is None or p in acc_indices:
                break
            # Check if parent has other children that are ALSO matched to a DIFFERENT accessory
            siblings = joints[p].get('children', [])
            other_matched = False
            for sib in siblings:
                if sib != current and sib in skel_indices:
                    # Check if this sibling subtree matches a DIFFERENT accessory
                    for si2, ai2, _ in matches:
                        if si2 != skel_idx and is_ancestor(joints, sib, si2):
                            if ai2 != acc_idx:
                                other_matched = True
                                break
                if other_matched:
                    break
            if other_matched:
                # Parent branches to different accessories — reparent at current level
                break
            current = p

        if current not in already_reparented:
            reparent_ops.append((current, acc_idx))
            already_reparented.add(current)

    # Apply reparenting
    reparented = set()
    for subtree_root, new_parent in reparent_ops:
        old_parent = parent_map.get(subtree_root)
        if old_parent is not None:
            children = joints[old_parent]['children']
            if subtree_root in children:
                children.remove(subtree_root)

        if subtree_root not in joints[new_parent]['children']:
            joints[new_parent]['children'].append(subtree_root)
            reparented.add(subtree_root)
            parent_map[subtree_root] = new_parent

    # Update roots — remove reparented joints from roots list
    new_roots = [r for r in roots if r not in reparented]

    return joints, new_roots, reparent_ops


def is_ancestor(joints, ancestor_idx, descendant_idx):
    """Check if ancestor_idx is an ancestor of descendant_idx."""
    visited = set()
    def search(idx):
        if idx == descendant_idx:
            return True
        if idx in visited:
            return False
        visited.add(idx)
        for c in joints[idx].get('children', []):
            if search(c):
                return True
        return False
    return search(ancestor_idx)


def recompute_inverse_bind_matrices(joints, roots):
    """Recompute world transforms and IBMs after hierarchy changes."""
    world = {}

    def trs_to_mat(t, r, s):
        cx, sx_ = math.cos(r[0]), math.sin(r[0])
        cy, sy_ = math.cos(r[1]), math.sin(r[1])
        cz, sz_ = math.cos(r[2]), math.sin(r[2])
        m = [0.0]*16
        m[0]=cy*cz*s[0]; m[1]=cy*sz_*s[0]; m[2]=-sy_*s[0]
        m[4]=(sx_*sy_*cz-cx*sz_)*s[1]; m[5]=(sx_*sy_*sz_+cx*cz)*s[1]; m[6]=sx_*cy*s[1]
        m[8]=(cx*sy_*cz+sx_*sz_)*s[2]; m[9]=(cx*sy_*sz_-sx_*cz)*s[2]; m[10]=cx*cy*s[2]
        m[12]=t[0]; m[13]=t[1]; m[14]=t[2]; m[15]=1.0
        return m

    def mat_mul(a, b):
        out = [0.0]*16
        for col in range(4):
            for row in range(4):
                out[col*4+row] = sum(a[k*4+row]*b[col*4+k] for k in range(4))
        return out

    def mat_invert(m):
        a,b,c,d,e,f,g,h,k = m[0],m[4],m[8],m[1],m[5],m[9],m[2],m[6],m[10]
        det = a*(e*k-f*h)-b*(d*k-f*g)+c*(d*h-e*g)
        if abs(det) < 1e-12:
            r = [0.0]*16; r[0]=r[5]=r[10]=r[15]=1.0; return r
        inv = 1.0/det
        r00=(e*k-f*h)*inv; r01=(c*h-b*k)*inv; r02=(b*f-c*e)*inv
        r10=(f*g-d*k)*inv; r11=(a*k-c*g)*inv; r12=(c*d-a*f)*inv
        r20=(d*h-e*g)*inv; r21=(b*g-a*h)*inv; r22=(a*e-b*d)*inv
        tx,ty,tz = m[12],m[13],m[14]
        out = [0.0]*16
        out[0]=r00;out[1]=r10;out[2]=r20
        out[4]=r01;out[5]=r11;out[6]=r21
        out[8]=r02;out[9]=r12;out[10]=r22
        out[12]=-(r00*tx+r01*ty+r02*tz)
        out[13]=-(r10*tx+r11*ty+r12*tz)
        out[14]=-(r20*tx+r21*ty+r22*tz)
        out[15]=1.0
        return out

    ident = [0.0]*16; ident[0]=ident[5]=ident[10]=ident[15]=1.0

    def walk(idx, parent_mat):
        j = joints[idx]
        local = trs_to_mat(j['translation'], j['rotation'], j['scale'])
        w = mat_mul(parent_mat, local)
        world[idx] = w
        for c in j.get('children', []):
            walk(c, w)

    for root in roots:
        walk(root, ident)

    # Build IBMs in joint index order
    ibms = []
    for i in range(len(joints)):
        w = world.get(i, ident)
        ibms.append([round(v, 8) for v in mat_invert(w)])
    return ibms


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <zoid_id> [output_dir]")
        sys.exit(1)

    zoid_id = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "models"
    os.makedirs(outdir, exist_ok=True)

    base = Path(__file__).parent.parent
    dat_path = base / "dat" / "vs3" / "zoid" / f"{zoid_id}.dat"
    b_dat_path = base / "iso_dump" / "files" / "zoid" / f"{zoid_id}_b.dat"

    if not dat_path.exists():
        print(f"Model not found: {dat_path}"); sys.exit(1)
    if not b_dat_path.exists():
        print(f"Animation not found: {b_dat_path}"); sys.exit(1)

    # Extract model
    result = extract_model(str(dat_path))
    joints = result['joints']
    roots = result['roots']

    # Get rest-pose count from _b.dat
    rest_pose_count = find_rest_pose_count(str(dat_path), str(b_dat_path))

    # Merge hierarchy (reparent skeleton limbs under accessories)
    joints, new_roots, reparent_ops = merge_hierarchy(joints, roots, rest_pose_count)
    result['roots'] = new_roots

    # Recompute inverse bind matrices with the new hierarchy
    if reparent_ops and 'skin' in result:
        result['skin']['inverseBindMatrices'] = recompute_inverse_bind_matrices(joints, new_roots)

    # Store rest-pose joint count for animation targeting
    result['restPoseCount'] = rest_pose_count

    # Report
    total_verts = sum(
        sum(len(m['positions'])//3 for m in j.get('meshes', []))
        for j in joints
    )
    joints_with_mesh = sum(1 for j in joints if 'meshes' in j)

    out_path = os.path.join(outdir, f"{zoid_id}_model.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, separators=(',', ':'))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"{zoid_id}: {len(joints)} joints ({joints_with_mesh} with geometry), "
          f"{total_verts} verts, {len(new_roots)} roots, "
          f"{len(reparent_ops)} reparent ops")
    for subtree, parent in reparent_ops:
        sname = joints[subtree].get('name', f'[{subtree}]')
        pname = joints[parent].get('name', f'[{parent}]')
        print(f"  {sname} → under {pname}")
    print(f"→ {out_path} ({size_kb:.0f}KB)")


if __name__ == "__main__":
    main()
