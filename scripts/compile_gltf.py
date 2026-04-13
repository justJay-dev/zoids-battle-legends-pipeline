#!/usr/bin/env python3
"""
Compile model JSON + texture PNG → glTF binary (.glb)

Usage: python3 scripts/compile_gltf.py <id> [output_dir]
  Reads: models/{id}_model.json, textures/{id}_tex0_512x512.png
  Writes: {output_dir}/{id}.glb
"""
import struct, sys, os, json, math
from pathlib import Path

# glTF constants
GLTF_COMPONENT_FLOAT = 5126
GLTF_COMPONENT_UNSIGNED_INT = 5125
GLTF_COMPONENT_UNSIGNED_SHORT = 5123
GLTF_COMPONENT_UNSIGNED_BYTE = 5121
GLTF_ELEMENT_ARRAY_BUFFER = 34963
GLTF_ARRAY_BUFFER = 34962


def pad_to_4(data: bytes) -> bytes:
    """Pad bytes to 4-byte alignment."""
    r = len(data) % 4
    return data + b'\x00' * (4 - r) if r else data


def build_glb(model_path: str, texture_path: str | None, name: str = "") -> bytes:
    with open(model_path) as f:
        model = json.load(f)

    joints = model["joints"]
    roots = model["roots"]

    # --- Build binary buffer ---
    buffer_data = bytearray()
    buffer_views = []
    accessors = []

    def add_buffer_view(data: bytes, target: int | None = None) -> int:
        idx = len(buffer_views)
        offset = len(buffer_data)
        buffer_data.extend(data)
        # Pad to 4 bytes
        while len(buffer_data) % 4:
            buffer_data.append(0)
        bv: dict = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            bv["target"] = target
        buffer_views.append(bv)
        return idx

    def add_accessor(bv_idx: int, comp_type: int, type_str: str, count: int,
                     min_vals: list | None = None, max_vals: list | None = None) -> int:
        idx = len(accessors)
        acc: dict = {
            "bufferView": bv_idx,
            "componentType": comp_type,
            "count": count,
            "type": type_str,
        }
        if min_vals is not None:
            acc["min"] = min_vals
        if max_vals is not None:
            acc["max"] = max_vals
        accessors.append(acc)
        return idx

    # --- Build nodes (one per joint) ---
    nodes = []
    meshes = []
    skins = []
    skin_data = model.get("skin")
    use_skinned_mesh = skin_data is not None and skin_data.get("mesh", {}).get("positions")

    for i, j in enumerate(joints):
        node: dict = {"name": j.get("name", f"joint_{i}")}

        r = j["rotation"]
        s = j["scale"]
        t = j["translation"]

        quat = euler_to_quat(r[0], r[1], r[2])
        node["rotation"] = [round(quat[0], 7), round(quat[1], 7),
                            round(quat[2], 7), round(quat[3], 7)]
        node["scale"] = [round(s[0], 6), round(s[1], 6), round(s[2], 6)]
        node["translation"] = [round(t[0], 6), round(t[1], 6), round(t[2], 6)]

        if j.get("children"):
            node["children"] = list(j["children"])

        # Per-joint meshes (legacy, used when no skin data)
        if not use_skinned_mesh and "meshes" in j:
            primitives = []
            for md in j["meshes"]:
                verts = len(md["positions"]) // 3
                if verts == 24:
                    continue
                pos_bytes = struct.pack(f"<{verts * 3}f", *md["positions"])
                pos_bv = add_buffer_view(pos_bytes, GLTF_ARRAY_BUFFER)
                pos_min = [min(md["positions"][k::3]) for k in range(3)]
                pos_max = [max(md["positions"][k::3]) for k in range(3)]
                pos_acc = add_accessor(pos_bv, GLTF_COMPONENT_FLOAT, "VEC3", verts,
                                       [round(v, 6) for v in pos_min],
                                       [round(v, 6) for v in pos_max])
                attrs: dict = {"POSITION": pos_acc}
                if md.get("normals") and len(md["normals"]) == verts * 3:
                    nrm_bytes = struct.pack(f"<{verts * 3}f", *md["normals"])
                    nrm_bv = add_buffer_view(nrm_bytes, GLTF_ARRAY_BUFFER)
                    attrs["NORMAL"] = add_accessor(nrm_bv, GLTF_COMPONENT_FLOAT, "VEC3", verts)
                if md.get("uvs") and len(md["uvs"]) == verts * 2:
                    uv_bytes = struct.pack(f"<{verts * 2}f", *md["uvs"])
                    uv_bv = add_buffer_view(uv_bytes, GLTF_ARRAY_BUFFER)
                    attrs["TEXCOORD_0"] = add_accessor(uv_bv, GLTF_COMPONENT_FLOAT, "VEC2", verts)
                idx_count = len(md["indices"])
                if verts > 65535:
                    idx_bytes = struct.pack(f"<{idx_count}I", *md["indices"])
                    idx_bv = add_buffer_view(idx_bytes, GLTF_ELEMENT_ARRAY_BUFFER)
                    idx_acc = add_accessor(idx_bv, GLTF_COMPONENT_UNSIGNED_INT, "SCALAR", idx_count)
                else:
                    idx_bytes = struct.pack(f"<{idx_count}H", *md["indices"])
                    idx_bv = add_buffer_view(idx_bytes, GLTF_ELEMENT_ARRAY_BUFFER)
                    idx_acc = add_accessor(idx_bv, GLTF_COMPONENT_UNSIGNED_SHORT, "SCALAR", idx_count)
                prim: dict = {"attributes": attrs, "indices": idx_acc}
                if texture_path:
                    prim["material"] = 0
                primitives.append(prim)
            if primitives:
                mesh_idx = len(meshes)
                meshes.append({"name": j.get("name", f"mesh_{i}"), "primitives": primitives})
                node["mesh"] = mesh_idx

        nodes.append(node)

    # --- Skinned mesh (when skin data available) ---
    if use_skinned_mesh:
        sm = skin_data["mesh"]
        ibms = skin_data["inverseBindMatrices"]
        verts = len(sm["positions"]) // 3

        # Positions (model space)
        pos_bytes = struct.pack(f"<{verts * 3}f", *sm["positions"])
        pos_bv = add_buffer_view(pos_bytes, GLTF_ARRAY_BUFFER)
        pos_min = [min(sm["positions"][k::3]) for k in range(3)]
        pos_max = [max(sm["positions"][k::3]) for k in range(3)]
        pos_acc = add_accessor(pos_bv, GLTF_COMPONENT_FLOAT, "VEC3", verts,
                               [round(v, 6) for v in pos_min],
                               [round(v, 6) for v in pos_max])
        attrs = {"POSITION": pos_acc}

        # Normals
        if sm.get("normals") and len(sm["normals"]) == verts * 3:
            nrm_bytes = struct.pack(f"<{verts * 3}f", *sm["normals"])
            nrm_bv = add_buffer_view(nrm_bytes, GLTF_ARRAY_BUFFER)
            attrs["NORMAL"] = add_accessor(nrm_bv, GLTF_COMPONENT_FLOAT, "VEC3", verts)

        # UVs
        if sm.get("uvs") and len(sm["uvs"]) == verts * 2:
            uv_bytes = struct.pack(f"<{verts * 2}f", *sm["uvs"])
            uv_bv = add_buffer_view(uv_bytes, GLTF_ARRAY_BUFFER)
            attrs["TEXCOORD_0"] = add_accessor(uv_bv, GLTF_COMPONENT_FLOAT, "VEC2", verts)

        # JOINTS_0 (4 joint indices per vertex, unsigned byte if < 256 joints)
        max_joint = max(sm["joints"]) if sm["joints"] else 0
        if max_joint < 256:
            j_bytes = struct.pack(f"<{verts * 4}B", *sm["joints"])
            j_bv = add_buffer_view(j_bytes, GLTF_ARRAY_BUFFER)
            attrs["JOINTS_0"] = add_accessor(j_bv, GLTF_COMPONENT_UNSIGNED_BYTE, "VEC4", verts)
        else:
            j_bytes = struct.pack(f"<{verts * 4}H", *sm["joints"])
            j_bv = add_buffer_view(j_bytes, GLTF_ARRAY_BUFFER)
            attrs["JOINTS_0"] = add_accessor(j_bv, GLTF_COMPONENT_UNSIGNED_SHORT, "VEC4", verts)

        # WEIGHTS_0 (4 weights per vertex, float)
        w_bytes = struct.pack(f"<{verts * 4}f", *sm["weights"])
        w_bv = add_buffer_view(w_bytes, GLTF_ARRAY_BUFFER)
        attrs["WEIGHTS_0"] = add_accessor(w_bv, GLTF_COMPONENT_FLOAT, "VEC4", verts)

        # Indices
        idx_count = len(sm["indices"])
        if verts > 65535:
            idx_bytes = struct.pack(f"<{idx_count}I", *sm["indices"])
            idx_bv = add_buffer_view(idx_bytes, GLTF_ELEMENT_ARRAY_BUFFER)
            idx_acc = add_accessor(idx_bv, GLTF_COMPONENT_UNSIGNED_INT, "SCALAR", idx_count)
        else:
            idx_bytes = struct.pack(f"<{idx_count}H", *sm["indices"])
            idx_bv = add_buffer_view(idx_bytes, GLTF_ELEMENT_ARRAY_BUFFER)
            idx_acc = add_accessor(idx_bv, GLTF_COMPONENT_UNSIGNED_SHORT, "SCALAR", idx_count)

        prim = {"attributes": attrs, "indices": idx_acc}
        if texture_path:
            prim["material"] = 0
        mesh_idx = len(meshes)
        meshes.append({"name": name or "skinned_mesh", "primitives": [prim]})

        # Inverse bind matrices
        ibm_flat = []
        for m in ibms:
            ibm_flat.extend(m)
        ibm_bytes = struct.pack(f"<{len(ibm_flat)}f", *ibm_flat)
        ibm_bv = add_buffer_view(ibm_bytes)
        ibm_acc = add_accessor(ibm_bv, GLTF_COMPONENT_FLOAT, "MAT4", len(ibms))

        # Skin object
        joint_indices = list(range(len(joints)))
        skin_obj = {
            "joints": joint_indices,
            "inverseBindMatrices": ibm_acc,
            "skeleton": roots[0] if roots else 0,
        }
        skins.append(skin_obj)

        # Add mesh node at SCENE level (not under any joint — avoids double transform)
        mesh_node_idx = len(nodes)
        nodes.append({"name": "skinned_mesh_node", "mesh": mesh_idx, "skin": 0})

    # --- Texture / Material ---
    images = []
    textures_gltf = []
    materials = []
    samplers = []

    if texture_path and os.path.exists(texture_path):
        # Embed PNG as buffer view
        with open(texture_path, "rb") as f:
            png_data = f.read()
        png_bv = add_buffer_view(png_data)

        images.append({"bufferView": png_bv, "mimeType": "image/png"})
        samplers.append({"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497})
        textures_gltf.append({"source": 0, "sampler": 0})
        materials.append({
            "name": "zoid_material",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            },
            "doubleSided": True,
        })
    else:
        materials.append({
            "name": "default",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.6, 0.6, 0.6, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            },
            "doubleSided": True,
        })

    # --- Scene ---
    # Include joint hierarchy roots + skinned mesh node (if present)
    scene_nodes = list(roots)
    if use_skinned_mesh:
        scene_nodes.append(mesh_node_idx)
    scene = {"nodes": scene_nodes}

    # --- Assemble glTF JSON ---
    gltf: dict = {
        "asset": {"version": "2.0", "generator": "zoids-pipeline", "extras": {"name": name}},
        "scene": 0,
        "scenes": [scene],
        "nodes": nodes,
        "buffers": [{"byteLength": len(buffer_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "materials": materials,
    }
    if meshes:
        gltf["meshes"] = meshes
    if skins:
        gltf["skins"] = skins
    if images:
        gltf["images"] = images
    if textures_gltf:
        gltf["textures"] = textures_gltf
    if samplers:
        gltf["samplers"] = samplers

    # --- Pack as GLB ---
    json_str = json.dumps(gltf, separators=(",", ":"))
    json_raw = json_str.encode("utf-8")
    # glTF spec: JSON chunk must be padded with spaces (0x20), not nulls
    r = len(json_raw) % 4
    json_bytes = json_raw + b' ' * (4 - r) if r else json_raw
    bin_bytes = pad_to_4(bytes(buffer_data))

    # GLB header: magic + version + length
    # JSON chunk: length + type + data
    # BIN chunk: length + type + data
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)

    glb = bytearray()
    glb.extend(struct.pack("<I", 0x46546C67))  # magic: glTF
    glb.extend(struct.pack("<I", 2))  # version
    glb.extend(struct.pack("<I", total_length))
    # JSON chunk
    glb.extend(struct.pack("<I", len(json_bytes)))
    glb.extend(struct.pack("<I", 0x4E4F534A))  # JSON
    glb.extend(json_bytes)
    # BIN chunk
    glb.extend(struct.pack("<I", len(bin_bytes)))
    glb.extend(struct.pack("<I", 0x004E4942))  # BIN
    glb.extend(bin_bytes)

    return bytes(glb)


def euler_to_quat(rx: float, ry: float, rz: float) -> list:
    """Convert XYZ intrinsic Euler (radians) to quaternion [x, y, z, w]."""
    # HSD uses intrinsic XYZ = extrinsic ZYX
    # Rotation order: Rz * Ry * Rx
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)

    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz

    return [qx, qy, qz, qw]


def load_file_map() -> dict:
    """Load the file_map.py name lookup."""
    fmap_path = Path(__file__).parent.parent / "file_map.py"
    if not fmap_path.exists():
        return {}
    ns: dict = {}
    exec(fmap_path.read_text(), ns)
    return ns.get("FILE_MAP", {})


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <id> [output_dir]")
        print(f"  Reads models/{{id}}_model.json and textures/{{id}}_tex0_512x512.png")
        sys.exit(1)

    zoid_id = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(outdir, exist_ok=True)

    file_map = load_file_map()
    friendly_name = file_map.get(zoid_id, zoid_id)

    base = Path(__file__).parent.parent
    model_path = base / "models" / f"{zoid_id}_model.json"
    texture_path = base / "textures" / f"{zoid_id}_tex0_512x512.png"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    tex_path_str = str(texture_path) if texture_path.exists() else None
    glb = build_glb(str(model_path), tex_path_str, friendly_name)

    out_path = os.path.join(outdir, f"{zoid_id}_{friendly_name}.glb")
    with open(out_path, "wb") as f:
        f.write(glb)

    size_kb = len(glb) / 1024
    print(f"{zoid_id}_{friendly_name}.glb — {size_kb:.0f}KB")


if __name__ == "__main__":
    main()
