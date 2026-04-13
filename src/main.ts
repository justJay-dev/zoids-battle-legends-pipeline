import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";

// --- Model selection ---
let selection = (document.getElementById("selModel") as HTMLSelectElement).value;
if (window.location.hash) {
    selection = window.location.hash.substring(1);
} else {
    selection = "b07";
}
document.getElementById("selModel")!.onchange = () => {
    window.location.hash = (document.getElementById("selModel") as HTMLSelectElement).value;
    window.location.reload();
};
(document.getElementById("selModel") as HTMLSelectElement).value = selection;

// --- Load pipeline outputs ---
interface MeshData { positions:number[]; normals:number[]; uvs:number[]; indices:number[]; }
interface JointData {
    name?:string;
    rotation:[number,number,number];
    scale:[number,number,number];
    translation:[number,number,number];
    children:number[];
    meshes?:MeshData[];
}
interface ModelData { roots:number[]; joints:JointData[]; }

interface AnimKey { f:number; v?:number; t?:number; i?:string; }
interface AnimClip { index:number; frameCount:number; joints:Array<{endFrame?:number;loop?:boolean;tracks:Record<string,AnimKey[]>}>; }
interface AnimData { restPose:any[]; clips:AnimClip[]; }

const [modelData, animData, texture] = await Promise.all([
    fetch(`./models/${selection}_model.json`).then(r=>r.ok?r.json():null) as Promise<ModelData|null>,
    fetch(`./animations/${selection}_anims.json`).then(r=>r.ok?r.json():null).catch(()=>null) as Promise<AnimData|null>,
    new Promise<THREE.Texture|null>(resolve=>{
        new THREE.TextureLoader().load(`./textures/${selection}_tex0_512x512.png`,tex=>{
            tex.flipY=false;
            tex.colorSpace=THREE.SRGBColorSpace;
            resolve(tex);
        },undefined,()=>resolve(null));
    }),
]);

if (!modelData) { document.body.textContent = `No model data for ${selection}`; throw new Error("no model"); }

console.log(`Model: ${modelData.joints.length} joints, roots=[${modelData.roots}]`);
if (animData) console.log(`Animations: ${animData.clips.length} clips, ${animData.restPose.length} rest joints`);

// --- Material ---
const material = new THREE.MeshStandardMaterial({
    side: THREE.DoubleSide,
    ...(texture ? { map: texture } : { color: 0x888888 }),
});

// --- Build THREE.js joint hierarchy from model JSON ---
const jointObj3Ds:THREE.Object3D[] = new Array(modelData.joints.length);

for (let i = 0; i < modelData.joints.length; i++) {
    const jd = modelData.joints[i];
    const obj3d = new THREE.Object3D();
    obj3d.name = jd.name || `joint_${i}`;
    obj3d.rotation.order = "XYZ";
    obj3d.rotation.set(jd.rotation[0], jd.rotation[1], jd.rotation[2]);
    obj3d.scale.set(jd.scale[0], jd.scale[1], jd.scale[2]);
    obj3d.position.set(jd.translation[0], jd.translation[1], jd.translation[2]);

    // Attach geometry meshes (skip 24-vert collision boxes)
    if (jd.meshes) {
        for (const md of jd.meshes) {
            const vertCount = md.positions.length / 3;
            if (vertCount === 24) continue;
            const geo = new THREE.BufferGeometry();
            geo.setAttribute("position", new THREE.Float32BufferAttribute(md.positions, 3));
            if (md.normals.length > 0) {
                geo.setAttribute("normal", new THREE.Float32BufferAttribute(md.normals, 3));
            }
            if (md.uvs.length > 0) {
                geo.setAttribute("uv", new THREE.Float32BufferAttribute(md.uvs, 2));
            }
            geo.setIndex(md.indices);
            if (md.normals.length === 0) geo.computeVertexNormals();
            obj3d.add(new THREE.Mesh(geo, material));
        }
    }
    jointObj3Ds[i] = obj3d;
}

// Wire parent-child relationships
const modelRoot = new THREE.Group();
for (const rootIdx of modelData.roots) {
    modelRoot.add(jointObj3Ds[rootIdx]);
}
for (let i = 0; i < modelData.joints.length; i++) {
    for (const childIdx of modelData.joints[i].children) {
        jointObj3Ds[i].add(jointObj3Ds[childIdx]);
    }
}

// Compute model extents from all mesh positions (world space via manual TRS walk)
let xMin=1e9,yMin=1e9,zMin=1e9,xMax=-1e9,yMax=-1e9,zMax=-1e9;
{
    // Quick extent calculation: add modelRoot to a temp scene, compute world matrices, get box
    const tmpScene = new THREE.Scene();
    tmpScene.add(modelRoot);
    modelRoot.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(modelRoot);
    tmpScene.remove(modelRoot);

    if (box.isEmpty()) {
        // Fallback: scan vertex positions from JSON
        for (const j of modelData.joints) {
            if (!j.meshes) continue;
            for (const m of j.meshes) {
                for (let i=0; i<m.positions.length; i+=3) {
                    const x=m.positions[i],y=m.positions[i+1],z=m.positions[i+2];
                    if(x<xMin)xMin=x;if(x>xMax)xMax=x;
                    if(y<yMin)yMin=y;if(y>yMax)yMax=y;
                    if(z<zMin)zMin=z;if(z>zMax)zMax=z;
                }
            }
        }
    } else {
        const min=box.min, max=box.max;
        xMin=min.x;yMin=min.y;zMin=min.z;
        xMax=max.x;yMax=max.y;zMax=max.z;
    }
}
const size = new THREE.Vector3(xMax-xMin, yMax-yMin, zMax-zMin);
const center = new THREE.Vector3((xMin+xMax)/2,(yMin+yMax)/2,(zMin+zMax)/2);
const modelHeight = size.y, modelDepth = Math.max(size.z, size.x);

modelRoot.position.sub(center);
modelRoot.rotateY(Math.PI + Math.PI / 4);

console.log(`Scene: ${modelData.joints.filter(j=>j.meshes).length} joints with geometry, extents=${size.x.toFixed(1)}x${size.y.toFixed(1)}x${size.z.toFixed(1)}`);

// --- Scene ---
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100000);
camera.position.set(modelDepth*1.5, modelHeight*1.2, modelDepth*2);
camera.lookAt(0,0,0);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

scene.add(new THREE.AmbientLight(0xffffff, 1.5));
scene.add(new THREE.HemisphereLight(0xffffff, 0x444466, 1.0));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
keyLight.position.set(modelDepth*2, modelHeight*4, modelDepth*2);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xddeeff, 1.2);
fillLight.position.set(-modelDepth*2, modelHeight*2, modelDepth*2);
scene.add(fillLight);
const rimLight = new THREE.DirectionalLight(0x8899ff, 0.8);
rimLight.position.set(0, modelHeight, -modelDepth*3);
scene.add(rimLight);
const grid = new THREE.GridHelper(modelDepth*6, 24, 0x444466, 0x333344);
grid.position.y = -size.y/2;
scene.add(grid);
scene.add(modelRoot);

// --- Animation ---
let currentClip = 0;
let animTime = 0;
const animSpeed = 30;
let animPlaying = false;
let animJointMap:number[] = [];

if (animData) {
    // Map anim joints → model joints by recursive tree topology matching.
    // Walk both trees (anim rest-pose + model) and match children by TRS proximity.
    const eps = 0.05;
    const restPose = animData.restPose;
    const restRoots:number[] = (animData as any).restRoots || [0];

    function matchTRS(animIdx:number, modelIdx:number):boolean {
        const rp = restPose[animIdx];
        const mj = modelData!.joints[modelIdx];
        const dt = Math.abs(mj.translation[0]-rp.translation[0]) +
                   Math.abs(mj.translation[1]-rp.translation[1]) +
                   Math.abs(mj.translation[2]-rp.translation[2]);
        const ds = Math.abs(mj.scale[0]-rp.scale[0]) +
                   Math.abs(mj.scale[1]-rp.scale[1]) +
                   Math.abs(mj.scale[2]-rp.scale[2]);
        return dt + ds < eps;
    }

    // Build full subtree list for each model joint (all descendants)
    function getSubtree(mi:number):number[] {
        const result = [mi];
        for (const ci of modelData!.joints[mi].children) {
            result.push(...getSubtree(ci));
        }
        return result;
    }

    // Recursive: match anim joint to model joint by TRS, searching candidates and their subtrees
    function matchTree(animIdx:number, modelCandidates:number[]) {
        const rp = restPose[animIdx];
        // Score each candidate — prefer joints with matching child count
        const animChildCount = (rp.children || []).length;

        for (const mi of modelCandidates) {
            if (animJointMap.includes(mi)) continue;
            if (matchTRS(animIdx, mi)) {
                animJointMap[animIdx] = mi;
                const animChildren = rp.children || [];
                // For children, search this model joint's full subtree + all model roots' subtrees
                const childCandidates:number[] = [];
                // Direct children first
                childCandidates.push(...modelData!.joints[mi].children);
                // Then all descendants
                for (const ci of modelData!.joints[mi].children) {
                    childCandidates.push(...getSubtree(ci));
                }
                // Also search from all roots (animation tree may span multiple model root chains)
                for (const ri of modelData!.roots) {
                    childCandidates.push(...getSubtree(ri));
                }
                const uniqueCandidates = [...new Set(childCandidates)];
                for (const ac of animChildren) {
                    matchTree(ac, uniqueCandidates);
                }
                return;
            }
        }
        // No match at this level — try going deeper into each candidate's subtree
        for (const mi of modelCandidates) {
            if (animJointMap.includes(mi)) continue;
            const subtree = getSubtree(mi);
            for (const si of subtree) {
                if (animJointMap.includes(si)) continue;
                if (matchTRS(animIdx, si)) {
                    matchTree(animIdx, [si]); // re-enter with the match
                    return;
                }
            }
        }
    }

    animJointMap = new Array(restPose.length).fill(-1);
    // Start from all model joints as candidates (not just roots)
    const allCandidates:number[] = [];
    for (const ri of modelData!.roots) allCandidates.push(...getSubtree(ri));
    for (const ar of restRoots) {
        matchTree(ar, allCandidates);
    }
    const mapped = animJointMap.filter(i=>i>=0).length;
    console.log(`Mapped ${mapped}/${restPose.length} anim joints`);
    // Log which skeleton joints got mapped
    for (let i=0;i<animJointMap.length;i++) {
        const mi = animJointMap[i];
        if (mi >= 0) {
            const n = modelData!.joints[mi].name || `joint_${mi}`;
            const hasMesh = !!modelData!.joints[mi].meshes;
            const hasChildMesh = modelData!.joints[mi].children.some(ci=>!!modelData!.joints[ci].meshes);
            if (hasMesh || hasChildMesh) console.log(`  anim[${i}]→model[${mi}] ${n} mesh=${hasMesh}`);
        }
    }

    // UI
    const clipSelect = document.createElement("select");
    clipSelect.style.cssText = "margin-left:8px";
    for (let i=0; i<animData.clips.length; i++) {
        const c = animData.clips[i];
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.text = `Clip ${c.index} (${c.frameCount.toFixed(0)}f)`;
        clipSelect.appendChild(opt);
    }
    clipSelect.onchange = () => { currentClip = +clipSelect.value; animTime = 0; };
    const playBtn = document.createElement("button");
    playBtn.textContent = "Play";
    playBtn.style.cssText = "margin-left:8px";
    playBtn.onclick = () => { animPlaying=!animPlaying; playBtn.textContent=animPlaying?"Pause":"Play"; };
    document.getElementById("selModel")!.after(clipSelect, playBtn);
}

function evalTrack(keys:AnimKey[], frame:number):number|undefined {
    if (!keys || keys.length===0) return undefined;
    if (keys.length===1) return keys[0].v;
    let prev=keys[0], next=keys[keys.length-1];
    for (let i=0; i<keys.length-1; i++) {
        if (keys[i+1].f >= frame) { prev=keys[i]; next=keys[i+1]; break; }
    }
    if (frame<=prev.f) return prev.v;
    if (frame>=next.f) return next.v;
    if (prev.v===undefined) return next.v;
    if (next.v===undefined) return prev.v;
    if (Math.abs(prev.v)>1000 || Math.abs(next.v)>1000) return undefined;
    const t = (frame-prev.f)/(next.f-prev.f);
    return prev.v + (next.v-prev.v) * t;
}

function applyAnimation(clip:AnimClip, frame:number) {
    const f = clip.frameCount > 0 ? frame % clip.frameCount : 0;
    for (let i=0; i<clip.joints.length; i++) {
        const si = animJointMap[i];
        if (si<0 || si>=jointObj3Ds.length) continue;
        const aj = clip.joints[i];
        const o = jointObj3Ds[si];
        if (!aj.tracks || Object.keys(aj.tracks).length===0) continue;
        const rx=evalTrack(aj.tracks.ROTX,f); if(rx!==undefined) o.rotation.x=rx;
        const ry=evalTrack(aj.tracks.ROTY,f); if(ry!==undefined) o.rotation.y=ry;
        const rz=evalTrack(aj.tracks.ROTZ,f); if(rz!==undefined) o.rotation.z=rz;
        const tx=evalTrack(aj.tracks.TRAX,f); if(tx!==undefined) o.position.x=tx;
        const ty=evalTrack(aj.tracks.TRAY,f); if(ty!==undefined) o.position.y=ty;
        const tz=evalTrack(aj.tracks.TRAZ,f); if(tz!==undefined) o.position.z=tz;
        const sx=evalTrack(aj.tracks.SCAX,f); if(sx!==undefined) o.scale.x=sx;
        const sy=evalTrack(aj.tracks.SCAY,f); if(sy!==undefined) o.scale.y=sy;
        const sz=evalTrack(aj.tracks.SCAZ,f); if(sz!==undefined) o.scale.z=sz;
    }
}

// --- Renderer ---
const renderer = new THREE.WebGLRenderer();
const container = document.getElementById("container")!;
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);
window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth/container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});
const controls = new TrackballControls(camera, renderer.domElement);

// --- Input ---
const keys = new Set<string>();
window.addEventListener("keydown", e => {
    keys.add(e.code);
    if (e.code==="Backquote") material.wireframe = !material.wireframe;
    if (e.code==="Space") { animPlaying=!animPlaying; const b=document.querySelector("button"); if(b) b.textContent=animPlaying?"Pause":"Play"; }
});
window.addEventListener("keyup", e => keys.delete(e.code));
const panSpeed=modelDepth*0.02, rotSpeed=0.03;
const _p=new THREE.Vector3(),_r=new THREE.Vector3(),_f=new THREE.Vector3(),_o=new THREE.Vector3(),_q=new THREE.Quaternion();

let lastTime = performance.now();
const animate = () => {
    requestAnimationFrame(animate);
    const now=performance.now(), dt=(now-lastTime)/1000;
    lastTime=now;

    if (animPlaying && animData && animData.clips.length>0) {
        animTime += dt * animSpeed;
        applyAnimation(animData.clips[currentClip], animTime);
    }

    if (keys.size>0) {
        camera.getWorldDirection(_f); _f.y=0; _f.normalize();
        _r.crossVectors(_f, camera.up).normalize();
        _p.set(0,0,0);
        if (keys.has("KeyW")||keys.has("ArrowUp")) _p.addScaledVector(_f, panSpeed);
        if (keys.has("KeyS")||keys.has("ArrowDown")) _p.addScaledVector(_f, -panSpeed);
        if (keys.has("KeyA")||keys.has("ArrowLeft")) _p.addScaledVector(_r, -panSpeed);
        if (keys.has("KeyD")||keys.has("ArrowRight")) _p.addScaledVector(_r, panSpeed);
        camera.position.add(_p); controls.target.add(_p);
        if (keys.has("KeyQ")||keys.has("KeyE")) {
            const a = keys.has("KeyQ") ? rotSpeed : -rotSpeed;
            _q.setFromAxisAngle(camera.up, a);
            _o.subVectors(camera.position, controls.target).applyQuaternion(_q);
            camera.position.copy(controls.target).add(_o); camera.lookAt(controls.target);
        }
    }

    controls.update();
    renderer.render(scene, camera);
};
animate();
