#!/usr/bin/env python3
"""
Build organized zoid packages: each zoid gets a directory with its model GLB
and weapon GLBs baked with the zoid's texture.

Output structure:
  output/{zoid_name}/
    model.glb                    ← zoid model with texture
    weapons/
      {weapon_id}.glb            ← weapon with zoid's texture baked in
"""
import os, sys, json, math
from pathlib import Path

# Add scripts dir to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir.parent))

from compile_gltf import build_glb
from extract_model import extract_model
from decode_cmpr import parse_dat_textures, decode_cmpr
from file_map import FILE_MAP


def mat4_identity():
    m = [0.0]*16; m[0]=m[5]=m[10]=m[15]=1.0; return m

def mat4_multiply(a, b):
    out = [0.0]*16
    for col in range(4):
        for row in range(4):
            out[col*4+row] = sum(a[k*4+row]*b[col*4+k] for k in range(4))
    return out

def mat4_from_trs(tx,ty,tz,rx,ry,rz,sx,sy,sz):
    cx,sx_=math.cos(rx),math.sin(rx); cy,sy_=math.cos(ry),math.sin(ry); cz,sz_=math.cos(rz),math.sin(rz)
    r00=cy*cz;r01=sx_*sy_*cz-cx*sz_;r02=cx*sy_*cz+sx_*sz_
    r10=cy*sz_;r11=sx_*sy_*sz_+cx*cz;r12=cx*sy_*sz_-sx_*cz
    r20=-sy_;r21=sx_*cy;r22=cx*cy
    m=[0.0]*16
    m[0]=r00*sx;m[1]=r10*sx;m[2]=r20*sx;m[4]=r01*sy;m[5]=r11*sy;m[6]=r21*sy
    m[8]=r02*sz;m[9]=r12*sz;m[10]=r22*sz;m[12]=tx;m[13]=ty;m[14]=tz;m[15]=1.0
    return m


def find_texture_path(zoid_id, base_dir):
    """Find the best texture file for a zoid."""
    tex_dir = base_dir / "textures"
    # Try standard naming
    for suffix in ["_tex0_512x512.png", "_tex0_256x256.png", "_tex0_128x128.png"]:
        p = tex_dir / f"{zoid_id}{suffix}"
        if p.exists():
            return str(p)
    return None


def get_weapon_files(zoid_id, weapon_dir):
    """Get weapon .dat files for a zoid based on naming convention."""
    # a## → wa##XX, b## → wb##XX
    faction = "wa" if zoid_id.startswith("a") else "wb"
    num = zoid_id[1:]  # "01", "07", etc.
    prefix = f"{faction}{num}"

    weapons = []
    if not weapon_dir.exists():
        return weapons
    for f in sorted(weapon_dir.iterdir()):
        if f.name.startswith(prefix) and f.name.endswith(".dat"):
            weapons.append(f)
    return weapons


def extract_weapon_texture(weapon_dat_path, output_dir):
    """Extract texture from weapon .dat file, return path to PNG or None."""
    try:
        data, textures = parse_dat_textures(str(weapon_dat_path))
        if not textures:
            return None
        # Use first texture found
        w, h, img_type, data_offset = textures[0]
        if img_type != 14:  # only CMPR supported
            return None
        pixels = decode_cmpr(data, data_offset, w, h)
        from PIL import Image
        img = Image.new("RGBA", (w, h))
        img.putdata(pixels)
        out_path = output_dir / f"{weapon_dat_path.stem}_tex.png"
        img.save(str(out_path))
        return str(out_path)
    except Exception:
        return None


def main():
    base = Path(__file__).parent.parent
    output_dir = base / (sys.argv[1] if len(sys.argv) > 1 else "output")
    dat_dir = base / "dat" / "vs3" / "zoid"
    weapon_dir = base / "iso_dump" / "files" / "weapon"
    weapon_tex_dir = base / "weapons" / "textures"
    models_dir = base / "models"
    textures_dir = base / "textures"

    # Load shared data files once
    weapon_descriptions = {}
    weapon_desc_path = base / "data" / "weapons.json"
    if weapon_desc_path.exists():
        with open(weapon_desc_path) as f:
            weapon_descriptions = json.load(f)

    all_zoid_weapons = {}
    zoid_weapons_path = base / "data" / "zoid_weapons.json"
    if zoid_weapons_path.exists():
        with open(zoid_weapons_path) as f:
            all_zoid_weapons = json.load(f)

    all_stats = {}
    zoid_stats_path = base / "data" / "zoid_stats.json"
    if zoid_stats_path.exists():
        with open(zoid_stats_path) as f:
            all_stats = json.load(f)

    weapon_details = {}
    weapon_details_path = base / "data" / "weapon_details.json"
    if weapon_details_path.exists():
        with open(weapon_details_path) as f:
            weapon_details = json.load(f)

    # wp→wa mapping (from game executable table)
    wp_to_wa = {}
    wp_to_wa_path = base / "data" / "wp_to_wa.json"
    if wp_to_wa_path.exists():
        with open(wp_to_wa_path) as f:
            wp_to_wa = json.load(f)
    # Build reverse: wa_file → wp_id + description
    wa_to_wp = {}
    for wp_id, info in wp_to_wa.items():
        if info.get("file"):
            wa_to_wp[info["file"]] = {"wp_id": wp_id, "description": info.get("description", "")}

    # Process each zoid
    for zoid_id, friendly_name in sorted(FILE_MAP.items()):
        dat_path = dat_dir / f"{zoid_id}.dat"
        if not dat_path.exists():
            continue

        model_json = models_dir / f"{zoid_id}_model.json"
        if not model_json.exists():
            print(f"  {zoid_id}: no model JSON, skipping")
            continue

        tex_path = find_texture_path(zoid_id, base)
        zoid_dir = output_dir / friendly_name
        zoid_dir.mkdir(parents=True, exist_ok=True)

        zoid_weapon_list = all_zoid_weapons.get(zoid_id, [])
        zoid_stats = all_stats.get(zoid_id)

        # Build zoid GLB
        glb = build_glb(str(model_json), tex_path, friendly_name)
        glb_path = zoid_dir / "model.glb"
        glb_path.write_bytes(glb)
        print(f"{friendly_name}/model.glb — {len(glb)//1024}KB")

        # Compute mount point world positions from model hierarchy
        mount_point_info = {}
        try:
            with open(str(model_json)) as mf:
                model_data = json.load(mf)
            parent_of = {}
            for ji, jj in enumerate(model_data['joints']):
                for ci in jj['children']:
                    parent_of[ci] = ji

            world_xforms = [None] * len(model_data['joints'])
            def compute_world(idx):
                if world_xforms[idx] is not None: return
                p = parent_of.get(idx)
                if p is not None: compute_world(p)
                j = model_data['joints'][idx]
                r, s, t = j['rotation'], j['scale'], j['translation']
                local = mat4_from_trs(t[0],t[1],t[2],r[0],r[1],r[2],s[0],s[1],s[2])
                if p is not None and world_xforms[p] is not None:
                    world_xforms[idx] = mat4_multiply(world_xforms[p], local)
                else:
                    world_xforms[idx] = local
            for ji in range(len(model_data['joints'])): compute_world(ji)

            for ji, jj in enumerate(model_data['joints']):
                jname = jj.get('name', '')
                if 'mtp_' not in jname: continue
                raw_num = jname.split('mtp_')[1].split('_')[0]
                mtp_id = 'mtp_' + raw_num
                pi = parent_of.get(ji)
                body_part = None
                while pi is not None:
                    pn = model_data['joints'][pi].get('name', '')
                    for part in ['HEAD','NECK','BUST','HIP','ARM','HAND','LEG','FOOT','TEAL']:
                        if part in pn.upper():
                            body_part = part.lower()
                            break
                    if body_part: break
                    pi = parent_of.get(pi)
                w = world_xforms[ji]
                pos = [round(w[12],3), round(w[13],3), round(w[14],3)] if w else [0,0,0]
                mount_point_info[mtp_id] = {"body_region": body_part or "unknown", "position": pos}
        except Exception:
            pass

        # Build weapon GLBs
        weapon_files = get_weapon_files(zoid_id, weapon_dir)
        if not weapon_files:
            continue

        weapons_dir = zoid_dir / "weapons"
        weapons_dir.mkdir(exist_ok=True)

        for wf in weapon_files:
            weapon_id = wf.stem  # wa0100, wb0704, etc.
            weapon_model = base / "weapons" / "models" / f"{weapon_id}_model.json"

            if not weapon_model.exists():
                try:
                    result = extract_model(str(wf))
                    has_geo = any("meshes" in j for j in result["joints"])
                    if not has_geo:
                        continue
                    weapon_model_tmp = str(weapons_dir / f"{weapon_id}_model.json")
                    with open(weapon_model_tmp, "w") as f:
                        json.dump(result, f, separators=(",", ":"))
                    weapon_model = Path(weapon_model_tmp)
                except Exception:
                    continue

            # Extract texture directly from weapon .dat file
            weapon_tex = extract_weapon_texture(wf, weapons_dir)
            final_tex = weapon_tex

            try:
                # Read weapon model to get root transform for mount alignment
                with open(str(weapon_model)) as wmf:
                    weapon_model_data = json.load(wmf)
                root_idx = weapon_model_data['roots'][0]
                root_joint = weapon_model_data['joints'][root_idx]
                weapon_root_transform = {
                    "rotation": root_joint['rotation'][:],
                    "translation": root_joint['translation'][:],
                    "scale": root_joint['scale'][:],
                }

                weapon_glb = build_glb(str(weapon_model), final_tex, weapon_id)
                if len(weapon_glb) < 100:
                    continue  # empty model
                out_path = weapons_dir / f"{weapon_id}.glb"
                out_path.write_bytes(weapon_glb)
                tex_src = "own" if weapon_tex else "none"
                print(f"  {friendly_name}/weapons/{weapon_id}.glb — {len(weapon_glb)//1024}KB (tex: {tex_src})")
                # Clean up extracted texture PNG from weapons dir
                if weapon_tex:
                    try: os.remove(weapon_tex)
                    except: pass
            except Exception as e:
                print(f"  {weapon_id}: error — {e}")

        # Build manifest.json
        manifest_weapons = []
        if (weapons_dir).exists():
            for wglb in sorted(weapons_dir.glob("*.glb")):
                wid = wglb.stem
                entry = {
                    "id": wid,
                    "file": f"weapons/{wid}.glb",
                }
                # Read this weapon's root transform for mount matching
                wm_path = base / "weapons" / "models" / f"{wid}_model.json"
                this_root_transform = {"rotation":[0,0,0],"translation":[0,0,0],"scale":[1,1,1]}
                if wm_path.exists():
                    try:
                        with open(str(wm_path)) as wrf:
                            wm_data = json.load(wrf)
                        wr = wm_data['joints'][wm_data['roots'][0]]
                        this_root_transform = {"rotation":wr['rotation'],"translation":wr['translation'],"scale":wr['scale']}
                    except: pass
                # Match weapon to mount point by root position (not pa_## name)
                wp_pos = this_root_transform["translation"]
                best_mtp = None
                best_dist = 999
                for mtp_id, mtp_info in mount_point_info.items():
                    if isinstance(mtp_info, dict) and "position" in mtp_info:
                        mp = mtp_info["position"]
                        dist = sum((a-b)**2 for a, b in zip(wp_pos, mp)) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_mtp = mtp_id
                            best_info = mtp_info

                if best_mtp and best_dist < 0.1:
                    entry["mount_point"] = {
                        "id": best_mtp,
                        "body_region": best_info.get("body_region", "unknown"),
                        "position": best_info.get("position", [0,0,0]),
                    }

                wd = weapon_details.get(wid, {})
                if wd.get("has_target"):
                    entry["has_target"] = True
                # Add root transform for mount alignment
                entry["root_transform"] = this_root_transform
                # Add wp description from executable table mapping
                wp_info = wa_to_wp.get(wid)
                if wp_info:
                    entry["wp_id"] = wp_info["wp_id"]
                    desc = wp_info["description"].strip()
                    if desc and not desc.startswith('\ufffd'):
                        entry["description"] = desc
                manifest_weapons.append(entry)

        manifest = {
            "id": zoid_id,
            "name": friendly_name,
            "model": "model.glb",
            "weapons": manifest_weapons,
        }
        if mount_point_info:
            manifest["mount_points"] = dict(sorted(mount_point_info.items()))
        if zoid_stats:
            manifest["stats"] = zoid_stats.get("stats", {})

        manifest_path = zoid_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
