#!/usr/bin/env python3
"""
Extract model data from Zoids .dat files into JSON.
Includes per-joint meshes (legacy) and a single skinned mesh with envelope weights.

Output includes:
  joints[]       — hierarchy with per-joint meshes (legacy, joint-local coords)
  skin.mesh      — combined mesh in model space with JOINTS_0/WEIGHTS_0
  skin.inverseBindMatrices — 4x4 per joint for glTF skinning
"""
import struct, sys, os, json
from pathlib import Path
from collections import defaultdict

# GX enums
GX_VA_PNMTXIDX = 0
GX_VA_POS = 9
GX_VA_NRM = 10
GX_VA_CLR0 = 11
GX_VA_TEX0 = 13
GX_VA_NULL = 0xFF
GX_DIRECT = 1
GX_INDEX8 = 2
GX_INDEX16 = 3

PRIM_QUADS = 0x80
PRIM_TRIANGLES = 0x90
PRIM_TRIANGLE_STRIP = 0x98
PRIM_TRIANGLE_FAN = 0xA0

POBJ_ENVELOPE = 1 << 13  # bit 13 of polygon flags


class DatReader:
    def __init__(self, data, data_offset):
        self.data = data
        self.do = data_offset

    def u8(self, off):  return self.data[self.do + off]
    def s8(self, off):  return struct.unpack_from(">b", self.data, self.do + off)[0]
    def u16(self, off): return struct.unpack_from(">H", self.data, self.do + off)[0]
    def s16(self, off): return struct.unpack_from(">h", self.data, self.do + off)[0]
    def u32(self, off): return struct.unpack_from(">I", self.data, self.do + off)[0]
    def f32(self, off): return struct.unpack_from(">f", self.data, self.do + off)[0]


def parse_vertex_attributes(r, va_ptr):
    attrs = []
    off = va_ptr
    while True:
        attr_name = r.u32(off)
        if attr_name == GX_VA_NULL:
            break
        attrs.append({
            "name": attr_name,
            "type": r.u32(off + 0x04),
            "comp_count": r.u32(off + 0x08),
            "comp_type": r.u32(off + 0x0C),
            "scale": r.u8(off + 0x10),
            "stride": r.u16(off + 0x12),
            "data_ptr": r.u32(off + 0x14),
        })
        off += 0x18
    return attrs


def parse_display_list(r, dl_ptr, dl_length, attrs):
    """Parse display list → list of primitives with per-vertex pos/nrm/uv/pnmtxidx."""
    primitives = []
    off = dl_ptr

    for _ in range(dl_length):
        prim_type = r.u8(off); off += 1
        count = r.u16(off); off += 2
        if prim_type == 0:
            break

        verts = []
        for _ in range(count):
            v = {"pos": [0,0,0], "nrm": [0,0,0], "tx0": [0,0], "pnmtxidx": 0}
            for attr in attrs:
                scale = 2 ** attr["scale"]
                if attr["type"] == GX_DIRECT:
                    stride_len = r.u8(off); off += 1
                elif attr["type"] == GX_INDEX16:
                    stride_len = r.s16(off); off += 2
                elif attr["type"] == GX_INDEX8:
                    stride_len = r.u8(off); off += 1
                else:
                    continue

                data_off = attr["data_ptr"] + attr["stride"] * stride_len

                if attr["name"] == GX_VA_PNMTXIDX:
                    v["pnmtxidx"] = stride_len  # raw PNMTXIDX value
                elif attr["name"] == GX_VA_POS:
                    v["pos"] = [r.f32(data_off)/scale, r.f32(data_off+4)/scale, r.f32(data_off+8)/scale]
                elif attr["name"] == GX_VA_NRM:
                    v["nrm"] = [r.f32(data_off)/scale, r.f32(data_off+4)/scale, r.f32(data_off+8)/scale]
                elif attr["name"] == GX_VA_TEX0:
                    v["tx0"] = [r.f32(data_off), r.f32(data_off+4)]
                # CLR0, others: skip

            verts.append(v)
        primitives.append({"type": prim_type, "verts": verts})

    return primitives


def parse_envelopes(r, env_array_ptr, data_block_size):
    """Parse null-terminated array of envelope pointers.
    Each envelope = null-terminated list of (jobj_offset, weight) pairs.
    Returns list of envelopes: [[(jobj_offset, weight), ...], ...]"""
    envelopes = []
    off = env_array_ptr
    while off < data_block_size:
        ptr = r.u32(off)
        if ptr == 0:
            break
        off += 4
        # Parse envelope: (jobj_ptr, weight) pairs until null jobj
        pairs = []
        eoff = ptr
        while True:
            jobj = r.u32(eoff)
            if jobj == 0:
                break
            weight = r.f32(eoff + 4)
            pairs.append((jobj, weight))
            eoff += 8
        envelopes.append(pairs)
    return envelopes


def resolve_vertex_joint(pnmtxidx, envelopes, single_bound_jobj):
    """Resolve a vertex's PNMTXIDX to (joint_offset, weight) pairs."""
    if single_bound_jobj is not None:
        return [(single_bound_jobj, 1.0)]
    env_idx = pnmtxidx // 3
    if 0 <= env_idx < len(envelopes):
        return envelopes[env_idx]
    return []


def primitives_to_indexed_mesh(verts_with_joint):
    """Convert list of (pos, nrm, uv, joint_offset) to indexed mesh."""
    if not verts_with_joint:
        return None
    positions, normals, uvs, indices = [], [], [], []
    for pos, nrm, uv in verts_with_joint:
        positions.extend(pos)
        normals.extend(nrm)
        uvs.append(uv[0])
        uvs.append(1.0 - uv[1])
    indices = list(range(len(verts_with_joint)))  # will be re-indexed per primitive
    return {
        "positions": [round(v, 6) for v in positions],
        "normals": [round(v, 6) for v in normals],
        "uvs": [round(v, 6) for v in uvs],
    }


def extract_model(filepath, strip_root=False):
    with open(filepath, "rb") as f:
        data = f.read()

    data_block_size = struct.unpack(">I", data[4:8])[0]
    reloc_count = struct.unpack(">I", data[8:12])[0]
    root_count = struct.unpack(">I", data[12:16])[0]
    data_offset = 0x20
    reloc_offset = data_offset + data_block_size
    root_offset = reloc_offset + reloc_count * 4
    table_offset = root_offset + root_count * 8

    r = DatReader(data, data_offset)

    # Parse root node names
    name_map = {}
    root_nodes = []
    for i in range(root_count):
        off = root_offset + i * 8
        roff = struct.unpack(">I", data[off:off+4])[0]
        soff = struct.unpack(">I", data[off+4:off+8])[0]
        str_start = table_offset + soff
        str_end = data.index(0, str_start) if 0 in data[str_start:str_start+64] else str_start+32
        name = data[str_start:str_end].decode("ascii", errors="replace").rstrip("\x00")
        name_map[roff] = name
        root_nodes.append((roff, name))

    # Joint array + offset→index map
    joints = []
    joint_index_map = {}  # dat offset → index
    visited_joints = set()

    def ensure_joint(off):
        """Create joint entry if not exists, return index."""
        if off in joint_index_map:
            return joint_index_map[off]
        idx = len(joints)
        joint_index_map[off] = idx
        name = name_map.get(off, "")
        # When strip_root, zero the root joint's TRS so GLB node has identity
        if strip_root and off in set(true_roots):
            joint = {
                "name": name,
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "translation": [0, 0, 0],
                "children": [],
            }
        else:
            joint = {
                "name": name,
                "rotation": [round(r.f32(off+0x14),6), round(r.f32(off+0x18),6), round(r.f32(off+0x1C),6)],
                "scale": [round(r.f32(off+0x20),6), round(r.f32(off+0x24),6), round(r.f32(off+0x28),6)],
                "translation": [round(r.f32(off+0x2C),6), round(r.f32(off+0x30),6), round(r.f32(off+0x34),6)],
                "children": [],
            }
        joints.append(joint)
        return idx

    # Per-joint geometry (legacy, used for per-joint meshes)
    joint_geometry = defaultdict(lambda: {"positions": [], "normals": [], "uvs": [], "indices": []})
    # Skinned mesh accumulator: single combined mesh with per-vertex skin data
    skinned_mesh = {"positions": [], "normals": [], "uvs": [], "indices": [],
                    "joints": [], "weights": []}  # joints/weights: 4 per vertex

    def parse_joint_tree(off):
        """Walk joint tree, register all joints."""
        if off == 0 or off in visited_joints: return
        visited_joints.add(off)
        ensure_joint(off)
        parse_joint_tree(r.u32(off + 0x08))  # child
        parse_joint_tree(r.u32(off + 0x0C))  # sibling

    def parse_geometry(off):
        """Walk joint tree, parse display objects and assign geometry to joints via skinning."""
        if off == 0 or off in visited_geo: return
        visited_geo.add(off)
        dobj_ptr = r.u32(off + 0x10)
        parse_display_chain(dobj_ptr, off)
        parse_geometry(r.u32(off + 0x08))
        parse_geometry(r.u32(off + 0x0C))

    def parse_display_chain(dobj_ptr, owning_joint_off):
        seen = set()
        dptr = dobj_ptr
        while dptr != 0 and dptr not in seen:
            seen.add(dptr)
            poly_ptr = r.u32(dptr + 0x0C)
            parse_polygon_chain(poly_ptr, owning_joint_off)
            dptr = r.u32(dptr + 0x04)

    def parse_polygon_chain(pobj_ptr, owning_joint_off):
        seen = set()
        pptr = pobj_ptr
        while pptr != 0 and pptr not in seen:
            seen.add(pptr)
            va_ptr = r.u32(pptr + 0x08)
            pflags = r.u16(pptr + 0x0C)
            dl_length = r.u16(pptr + 0x0E)
            dl_ptr = r.u32(pptr + 0x10)
            skin_ptr = r.u32(pptr + 0x14)  # envelope array or single-bound JOBJ

            if va_ptr == 0 or dl_ptr == 0 or dl_length == 0:
                pptr = r.u32(pptr + 0x04)
                continue

            attrs = parse_vertex_attributes(r, va_ptr)
            is_envelope = bool(pflags & POBJ_ENVELOPE)

            envelopes = None
            single_bound_jobj = None
            if is_envelope and skin_ptr != 0:
                envelopes = parse_envelopes(r, skin_ptr, data_block_size)
            elif skin_ptr != 0:
                single_bound_jobj = skin_ptr  # pointer to a single JOBJ

            primitives = parse_display_list(r, dl_ptr, dl_length, attrs)

            for prim in primitives:
                verts = prim["verts"]
                n = len(verts)
                if n == 24 and len(primitives) == 1:
                    continue  # skip collision boxes

                # Build triangle list from primitive
                tri_indices = []
                if prim["type"] == PRIM_TRIANGLE_STRIP:
                    for i in range(n - 2):
                        if i % 2 == 1: tri_indices.append((i, i+1, i+2))
                        else: tri_indices.append((i, i+2, i+1))
                elif prim["type"] == PRIM_QUADS:
                    for i in range(0, n, 4):
                        tri_indices.append((i+2, i+1, i))
                        tri_indices.append((i, i+3, i+2))
                elif prim["type"] == PRIM_TRIANGLES:
                    for i in range(0, n, 3):
                        tri_indices.append((i, i+1, i+2))
                elif prim["type"] == PRIM_TRIANGLE_FAN:
                    for i in range(1, n - 1):
                        tri_indices.append((0, i, i+1))

                # For each triangle, collect vertices with skin data
                for i0, i1, i2 in tri_indices:
                    tri_verts = [verts[i0], verts[i1], verts[i2]]

                    base = len(skinned_mesh["positions"]) // 3
                    for v in tri_verts:
                        # Resolve per-vertex joint/weight pairs
                        if envelopes is not None:
                            pairs = resolve_vertex_joint(v.get("pnmtxidx", 0), envelopes, None)
                        elif single_bound_jobj is not None:
                            pairs = [(single_bound_jobj, 1.0)]
                        else:
                            pairs = [(owning_joint_off, 1.0)]
                        if not pairs:
                            pairs = [(owning_joint_off, 1.0)]

                        # Ensure all referenced joints exist, convert offsets to indices
                        joint_weights = []
                        for joff, w in pairs:
                            if joff not in joint_index_map:
                                ensure_joint(joff)
                            joint_weights.append((joint_index_map[joff], w))

                        # Sort by weight descending, take top 4, normalize
                        joint_weights.sort(key=lambda x: -x[1])
                        joint_weights = joint_weights[:4]
                        total_w = sum(w for _, w in joint_weights)
                        if total_w > 0:
                            joint_weights = [(j, w / total_w) for j, w in joint_weights]
                        # Pad to exactly 4
                        while len(joint_weights) < 4:
                            joint_weights.append((0, 0.0))

                        # Store model-space position and normal
                        skinned_mesh["positions"].extend(v["pos"])
                        skinned_mesh["normals"].extend(v["nrm"])
                        skinned_mesh["uvs"].append(round(v["tx0"][0], 6))
                        skinned_mesh["uvs"].append(round(v["tx0"][1], 6))
                        skinned_mesh["joints"].extend([j for j, _ in joint_weights])
                        skinned_mesh["weights"].extend([round(w, 6) for _, w in joint_weights])

                    skinned_mesh["indices"].extend([base, base+1, base+2])

                    # Also build per-joint geometry (legacy format)
                    v0 = tri_verts[0]
                    if envelopes is not None:
                        p0 = resolve_vertex_joint(v0.get("pnmtxidx", 0), envelopes, None)
                    elif single_bound_jobj is not None:
                        p0 = [(single_bound_jobj, 1.0)]
                    else:
                        p0 = [(owning_joint_off, 1.0)]
                    target_off = max(p0, key=lambda p: p[1])[0] if p0 else owning_joint_off
                    if target_off not in joint_index_map:
                        ensure_joint(target_off)
                    geo = joint_geometry[target_off]
                    gbase = len(geo["positions"]) // 3
                    inv = mat4_invert(world_transforms.get(target_off, mat4_identity()))
                    for v in tri_verts:
                        lx,ly,lz = mat4_transform_point(inv, v["pos"][0], v["pos"][1], v["pos"][2])
                        geo["positions"].extend([lx, ly, lz])
                        nx,ny,nz = mat4_transform_point(inv, v["nrm"][0]+v["pos"][0], v["nrm"][1]+v["pos"][1], v["nrm"][2]+v["pos"][2])
                        nx-=lx; ny-=ly; nz-=lz
                        geo["normals"].extend([nx, ny, nz])
                        geo["uvs"].append(round(v["tx0"][0], 6))
                        geo["uvs"].append(round(v["tx0"][1], 6))
                    geo["indices"].extend([gbase, gbase+1, gbase+2])

            pptr = r.u32(pptr + 0x04)

    # Find master joint tree via scene_data → ptr0 → [0] → master root
    # This works for both zoid models and weapon models
    master_root = None
    for roff, name in root_nodes:
        if "scene_data" not in name:
            continue
        sd = roff
        p0 = r.u32(sd)
        if p0 == 0 or p0 >= data_block_size:
            continue
        p0_0 = r.u32(p0)
        if p0_0 == 0 or p0_0 >= data_block_size:
            continue
        candidate = r.u32(p0_0)
        if 0 < candidate < data_block_size:
            master_root = candidate
            break

    if master_root is None:
        # Fallback: use root table joints directly
        all_offsets = set()
        referenced = set()
        def scan_refs(off, seen):
            if off == 0 or off in seen: return
            seen.add(off)
            all_offsets.add(off)
            child = r.u32(off + 0x08)
            sib = r.u32(off + 0x0C)
            if child != 0: referenced.add(child); scan_refs(child, seen)
            if sib != 0: referenced.add(sib); scan_refs(sib, seen)
        for roff, name in root_nodes:
            if "scene_data" in name: continue
            scan_refs(roff, set())
        true_roots = [off for off in all_offsets if off not in referenced]
    else:
        true_roots = [master_root]

    # Phase 1: register all joints
    for roff in true_roots:
        parse_joint_tree(roff)

    # Phase 2: compute world transforms for all joints (for inverse bind)
    import math
    world_transforms = {}  # joint_offset → 4x4 matrix (flat, column-major)

    def mat4_identity():
        m = [0.0]*16; m[0]=m[5]=m[10]=m[15]=1.0; return m

    def mat4_multiply(a, b):
        out = [0.0]*16
        for col in range(4):
            for row in range(4):
                out[col*4+row] = sum(a[k*4+row]*b[col*4+k] for k in range(4))
        return out

    def mat4_from_trs(tx,ty,tz,rx,ry,rz,sx,sy,sz):
        cx,sx_=math.cos(rx),math.sin(rx)
        cy,sy_=math.cos(ry),math.sin(ry)
        cz,sz_=math.cos(rz),math.sin(rz)
        r00=cy*cz; r01=sx_*sy_*cz-cx*sz_; r02=cx*sy_*cz+sx_*sz_
        r10=cy*sz_; r11=sx_*sy_*sz_+cx*cz; r12=cx*sy_*sz_-sx_*cz
        r20=-sy_; r21=sx_*cy; r22=cx*cy
        m=[0.0]*16
        m[0]=r00*sx;m[1]=r10*sx;m[2]=r20*sx
        m[4]=r01*sy;m[5]=r11*sy;m[6]=r21*sy
        m[8]=r02*sz;m[9]=r12*sz;m[10]=r22*sz
        m[12]=tx;m[13]=ty;m[14]=tz;m[15]=1.0
        return m

    def mat4_invert(m):
        # Invert a 4x4 affine matrix (rotation+scale+translation)
        # For affine: inv = [R^-1 | -R^-1 * t]
        # Compute 3x3 inverse via cofactors
        a=m[0];b=m[4];c=m[8];d=m[1];e=m[5];f=m[9];g=m[2];h=m[6];k=m[10]
        det=a*(e*k-f*h)-b*(d*k-f*g)+c*(d*h-e*g)
        if abs(det)<1e-12: return mat4_identity()
        inv_det=1.0/det
        r00=(e*k-f*h)*inv_det;r01=(c*h-b*k)*inv_det;r02=(b*f-c*e)*inv_det
        r10=(f*g-d*k)*inv_det;r11=(a*k-c*g)*inv_det;r12=(c*d-a*f)*inv_det
        r20=(d*h-e*g)*inv_det;r21=(b*g-a*h)*inv_det;r22=(a*e-b*d)*inv_det
        tx=m[12];ty=m[13];tz=m[14]
        out=[0.0]*16
        out[0]=r00;out[1]=r10;out[2]=r20
        out[4]=r01;out[5]=r11;out[6]=r21
        out[8]=r02;out[9]=r12;out[10]=r22
        out[12]=-(r00*tx+r01*ty+r02*tz)
        out[13]=-(r10*tx+r11*ty+r12*tz)
        out[14]=-(r20*tx+r21*ty+r22*tz)
        out[15]=1.0
        return out

    def mat4_transform_point(m, x, y, z):
        return (
            m[0]*x+m[4]*y+m[8]*z+m[12],
            m[1]*x+m[5]*y+m[9]*z+m[13],
            m[2]*x+m[6]*y+m[10]*z+m[14],
        )

    root_offsets = set(true_roots)

    def compute_world_transforms(off, parent_matrix):
        if off == 0 or off in world_transforms: return
        rx,ry,rz = r.f32(off+0x14),r.f32(off+0x18),r.f32(off+0x1C)
        sx,sy,sz = r.f32(off+0x20),r.f32(off+0x24),r.f32(off+0x28)
        tx,ty,tz = r.f32(off+0x2C),r.f32(off+0x30),r.f32(off+0x34)
        # When strip_root is True, use identity for root joints so geometry
        # ends up in mount-point-local space (for weapons)
        if strip_root and off in root_offsets:
            local = mat4_identity()
        else:
            local = mat4_from_trs(tx,ty,tz,rx,ry,rz,sx,sy,sz)
        world = mat4_multiply(parent_matrix, local)
        world_transforms[off] = world
        compute_world_transforms(r.u32(off+0x08), world)   # child
        compute_world_transforms(r.u32(off+0x0C), parent_matrix)  # sibling

    for roff in true_roots:
        compute_world_transforms(roff, mat4_identity())

    # Phase 3: parse geometry with skinning
    visited_geo = set()
    for roff in true_roots:
        parse_geometry(roff)

    # Phase 4: fix children (child/sibling → parent's children)
    for j in joints:
        j["children"] = []
    root_indices = []
    visited_fix = set()
    def fix_children(off, parent_idx):
        if off == 0 or off in visited_fix: return
        visited_fix.add(off)
        idx = joint_index_map.get(off)
        if idx is None: return
        if parent_idx >= 0:
            joints[parent_idx]["children"].append(idx)
        else:
            root_indices.append(idx)
        fix_children(r.u32(off + 0x08), idx)
        fix_children(r.u32(off + 0x0C), parent_idx)
    for roff in true_roots:
        fix_children(roff, -1)

    # Phase 5: attach geometry to joints
    for joff, geo in joint_geometry.items():
        idx = joint_index_map.get(joff)
        if idx is None: continue
        if not geo["positions"]: continue
        if "meshes" not in joints[idx]:
            joints[idx]["meshes"] = []
        joints[idx]["meshes"].append({
            "positions": [round(v, 6) for v in geo["positions"]],
            "normals": [round(v, 6) for v in geo["normals"]],
            "uvs": geo["uvs"],
            "indices": geo["indices"],
        })

    # Cleanup
    for j in joints:
        if "meshes" not in j or not j.get("meshes"):
            j.pop("meshes", None)
        if not j.get("name"):
            j.pop("name", None)

    # Build inverse bind matrices (one 4x4 per joint, flattened)
    inverse_bind_matrices = []
    for joff in [off for off, _ in sorted(joint_index_map.items(), key=lambda x: x[1])]:
        wt = world_transforms.get(joff, mat4_identity())
        ibm = mat4_invert(wt)
        inverse_bind_matrices.append([round(v, 8) for v in ibm])

    # Round skinned mesh positions/normals
    skinned_mesh["positions"] = [round(v, 6) for v in skinned_mesh["positions"]]
    skinned_mesh["normals"] = [round(v, 6) for v in skinned_mesh["normals"]]

    result = {"roots": root_indices, "joints": joints}
    if skinned_mesh["positions"]:
        result["skin"] = {
            "mesh": skinned_mesh,
            "inverseBindMatrices": inverse_bind_matrices,
        }
    return result


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.dat> [output_dir]")
        sys.exit(1)

    filepath = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(outdir, exist_ok=True)

    basename = Path(filepath).stem
    result = extract_model(filepath)

    total_verts = 0
    joints_with_mesh = 0
    for j in result["joints"]:
        if "meshes" in j:
            joints_with_mesh += 1
            for m in j["meshes"]:
                total_verts += len(m["positions"]) // 3

    out_path = os.path.join(outdir, f"{basename}_model.json")
    with open(out_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"{basename}: {len(result['joints'])} joints ({joints_with_mesh} with geometry), "
          f"{total_verts} verts, {len(result['roots'])} roots → {out_path} ({size_kb:.0f}KB)")


if __name__ == "__main__":
    main()
