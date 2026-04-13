DAT_DIR    = dat/vs3/zoid
ISO_ZOID   = iso_dump/files/zoid
ISO_WEAPON = iso_dump/files/weapon
SCRIPTS    = scripts

dev:
	bun run server.ts

kill:
	-lsof -ti tcp:8080 | xargs kill -9

# Full pipeline
pipeline: textures models gltf

# Zoid textures from DAT files
textures:
	@mkdir -p textures
	@for f in $(DAT_DIR)/*.dat; do \
		case "$$f" in *_b.dat) continue;; esac; \
		python3 $(SCRIPTS)/decode_cmpr.py "$$f" textures/; \
	done

# Zoid models from DAT files
models:
	@mkdir -p models
	@for f in $(DAT_DIR)/*.dat; do \
		case "$$f" in *_b.dat) continue;; esac; \
		python3 $(SCRIPTS)/extract_model.py "$$f" models/; \
	done

# Compile zoid glTF from models + textures
gltf: models textures
	@mkdir -p gltf
	@for f in models/*_model.json; do \
		id=$$(basename "$$f" _model.json); \
		python3 $(SCRIPTS)/compile_gltf.py "$$id" gltf/; \
	done

# Weapon pipeline
weapons: weapon-models weapon-textures weapon-gltf

weapon-models:
	@mkdir -p weapons/models
	@for f in $(ISO_WEAPON)/wa*.dat $(ISO_WEAPON)/wb*.dat; do \
		[ -f "$$f" ] || continue; \
		python3 $(SCRIPTS)/extract_model.py "$$f" weapons/models/; \
	done

weapon-textures:
	@mkdir -p weapons/textures
	@for f in $(ISO_WEAPON)/all_tex/*.tpl; do \
		[ -f "$$f" ] || continue; \
		python3 $(SCRIPTS)/decode_tpl.py "$$f" weapons/textures/; \
	done

weapon-gltf: weapon-models weapon-textures
	@mkdir -p weapons/gltf
	@for f in weapons/models/*_model.json; do \
		[ -f "$$f" ] || continue; \
		id=$$(basename "$$f" _model.json); \
		tex=$$(ls weapons/textures/$${id}_A.png 2>/dev/null || echo ""); \
		if [ -n "$$tex" ]; then \
			python3 -c "import sys; sys.path.insert(0,'$(SCRIPTS)'); from compile_gltf import build_glb; glb=build_glb('$$f','$$tex','$$id'); open('weapons/gltf/$${id}.glb','wb').write(glb); print(f'$${id}.glb — {len(glb)//1024}KB')"; \
		else \
			python3 -c "import sys; sys.path.insert(0,'$(SCRIPTS)'); from compile_gltf import build_glb; glb=build_glb('$$f',None,'$$id'); open('weapons/gltf/$${id}.glb','wb').write(glb); print(f'$${id}.glb — {len(glb)//1024}KB (no texture)')"; \
		fi; \
	done

# Accessory animations from _b.dat files
animations:
	@mkdir -p animations
	@for f in $(ISO_ZOID)/*_b.dat; do \
		python3 $(SCRIPTS)/export_animations.py "$$f" animations/; \
	done

# Organized packages: zoid + weapons with baked textures
packages: models textures weapon-models weapon-textures
	python3 $(SCRIPTS)/build_packages.py output

# Everything
all: pipeline weapons animations packages

clean:
	rm -rf textures/ models/ animations/ gltf/ weapons/

.PHONY: dev kill pipeline textures models gltf weapons weapon-models weapon-textures weapon-gltf animations all clean
