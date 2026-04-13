# Zoids Battle Legends — Asset Extraction Pipeline

Reverse engineering toolkit for extracting 3D models, textures, weapons, and game data from the GameCube game **Zoids Battle Legends** (Zoids VS III).

## What This Does

Extracts game assets from `.dat` binary files (HSD/HAL Laboratory format) into standard formats:

- **3D Models** → glTF (.glb) with joint hierarchy and textures
- **Textures** → PNG (decoded from GC CMPR/DXT1)
- **Weapons** → glTF with mount point data
- **Game Data** → JSON (stats, weapon descriptions, pilot info)

## Quick Start

```bash
bun install
make pipeline    # extract zoid models + textures → glTF
make weapons     # extract weapon models + textures → glTF  
make packages    # organized output with manifests
bun run dev      # start viewer at http://localhost:8080
```

## Output Structure

```
output/{zoid_name}/
  model.glb              — zoid model with embedded texture
  manifest.json          — metadata, stats, weapon list, mount points
  weapons/
    {weapon_id}.glb      — weapon models
```

## Key References

Format understanding was informed by these open-source projects:
- [HSDRaw](https://github.com/Ploaj/HSDLib) (MIT) — HSD struct layouts and animation format
- [DAT Texture Wizard](https://github.com/DRGN-DRC/DAT-Texture-Wizard) (Apache 2.0) — HSD struct documentation
- [Smash Forge](https://github.com/jam1garner/Smash-Forge) (MIT) — display list parsing approach

## Disclaimer

Game data files (`.dat`, `.prm`, ISO contents) are copyright their original developers and TOMY. They are not included in this repository.

## License

MIT
