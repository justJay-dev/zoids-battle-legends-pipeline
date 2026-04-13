/**
 * Joint Mapper Tool — visual 3D + tree panels
 * Left: 3D model with highlighted joints (spheres)
 * Right: two tree panels (anim + model)
 * Click model joint sphere in 3D or in the tree to select.
 */
import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";

const selection = window.location.hash?.substring(1) || "b02";

const [modelData, animData, texture] = await Promise.all([
    fetch(`./models/${selection}_model.json`).then(r => r.json()),
    fetch(`./animations/${selection}_anims.json`).then(r => r.json()),
    new Promise<THREE.Texture|null>(resolve => {
        new THREE.TextureLoader().load(`./textures/${selection}_tex0_512x512.png`, tex => {
            tex.flipY = false; tex.colorSpace = THREE.SRGBColorSpace; resolve(tex);
        }, undefined, () => resolve(null));
    }),
]);

document.title = `Joint Mapper — ${selection}`;

// --- Layout ---
const container = document.getElementById("container")!;
container.innerHTML = `
<style>
    #layout { display:flex; height:100%; }
    #view3d { flex:2; position:relative; }
    #panels { flex:1; display:flex; flex-direction:column; overflow:hidden; min-width:350px; }
    .tree-panel { flex:1; overflow:auto; border:1px solid #333; padding:6px; background:#1a1a2e; color:#ccc; font:12px monospace; }
    .tree-panel h3 { margin:0 0 6px; color:#fff; font-size:13px; }
    .joint { padding:1px 4px; cursor:pointer; white-space:nowrap; }
    .joint:hover { background:#333; }
    .joint.selected { background:#446; outline:1px solid #88f; }
    .joint.mapped { color:#4f4; }
    .joint.has-mesh { font-weight:bold; }
    .joint .idx { color:#666; }
    .joint .t { color:#555; font-size:11px; }
    #toolbar { padding:6px 10px; background:#222; color:#ccc; font:12px monospace; border-bottom:1px solid #333; }
    #toolbar button { margin-right:6px; }
    #status { color:#ff0; margin-left:8px; }
    #mapping-output { width:100%; height:80px; font:11px monospace; background:#111; color:#0f0; border:1px solid #333; resize:vertical; }
    #joint-info { position:absolute; bottom:10px; left:10px; background:rgba(0,0,0,0.8); color:#fff; padding:8px; font:12px monospace; pointer-events:none; border-radius:4px; }
</style>
<div id="toolbar">
    <b>${selection}</b> |
    Step 1: Click anim joint (left tree) | Step 2: Click model joint (right tree or 3D sphere)
    <button id="btnClear">Clear</button>
    <button id="btnExport">Copy JSON</button>
    <span id="status"></span>
    <br><textarea id="mapping-output" readonly></textarea>
</div>
<div id="layout">
    <div id="view3d">
        <div id="joint-info" style="display:none"></div>
    </div>
    <div id="panels">
        <div class="tree-panel" id="anim-tree"><h3>Anim (${animData.restPose.length})</h3></div>
        <div class="tree-panel" id="model-tree"><h3>Model (${modelData.joints.length})</h3></div>
    </div>
</div>
`;

// --- State ---
interface Mapping { animIdx:number; modelIdx:number; }
const mappings:Mapping[] = [];
let selectedAnimIdx:number|null = null;
let hoveredModelIdx:number|null = null;

// --- Build 3D scene ---
const viewEl = document.getElementById("view3d")!;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const material = new THREE.MeshStandardMaterial({
    side: THREE.DoubleSide,
    ...(texture ? { map: texture } : { color: 0x888888 }),
    transparent: true, opacity: 0.6,
});

// Build joint Object3Ds
const jointObj3Ds:THREE.Object3D[] = [];
const jointSpheres:THREE.Mesh[] = []; // clickable spheres per joint
const sphereGeo = new THREE.SphereGeometry(0.3, 8, 8);
const sphereMatDefault = new THREE.MeshBasicMaterial({ color: 0x4488ff, transparent: true, opacity: 0.7 });
const sphereMatHover = new THREE.MeshBasicMaterial({ color: 0xffff00 });
const sphereMatSelected = new THREE.MeshBasicMaterial({ color: 0xff4444 });
const sphereMatMapped = new THREE.MeshBasicMaterial({ color: 0x44ff44 });

for (let i = 0; i < modelData.joints.length; i++) {
    const jd = modelData.joints[i];
    const obj3d = new THREE.Object3D();
    obj3d.name = jd.name || `joint_${i}`;
    obj3d.rotation.order = "XYZ";
    obj3d.rotation.set(jd.rotation[0], jd.rotation[1], jd.rotation[2]);
    obj3d.scale.set(jd.scale[0], jd.scale[1], jd.scale[2]);
    obj3d.position.set(jd.translation[0], jd.translation[1], jd.translation[2]);

    if (jd.meshes) {
        for (const md of jd.meshes) {
            if (md.positions.length / 3 === 24) continue;
            const geo = new THREE.BufferGeometry();
            geo.setAttribute("position", new THREE.Float32BufferAttribute(md.positions, 3));
            if (md.uvs.length > 0) geo.setAttribute("uv", new THREE.Float32BufferAttribute(md.uvs, 2));
            geo.setIndex(md.indices);
            geo.computeVertexNormals();
            obj3d.add(new THREE.Mesh(geo, material));
        }
    }

    // Add joint sphere marker — added to a separate group so it doesn't inherit joint scale
    const sphere = new THREE.Mesh(sphereGeo, sphereMatDefault.clone());
    sphere.userData.jointIdx = i;
    sphere.userData.parentObj = obj3d;
    jointSpheres.push(sphere);

    jointObj3Ds.push(obj3d);
}

// Wire hierarchy
const modelRoot = new THREE.Group();
for (const ri of modelData.roots) modelRoot.add(jointObj3Ds[ri]);
for (let i = 0; i < modelData.joints.length; i++) {
    for (const ci of modelData.joints[i].children) {
        jointObj3Ds[i].add(jointObj3Ds[ci]);
    }
}

// Center model
modelRoot.updateMatrixWorld(true);
const box = new THREE.Box3().setFromObject(modelRoot);
if (box.isEmpty()) {
    // Fallback extents
    box.set(new THREE.Vector3(-10,-10,-10), new THREE.Vector3(10,10,10));
}
const size = box.getSize(new THREE.Vector3());
const center = box.getCenter(new THREE.Vector3());
modelRoot.position.sub(center);
scene.add(modelRoot);

// Find skeleton subtree (the root whose subtree contains the most geometry)
function getSubtree(idx:number):Set<number> {
    const result = new Set<number>([idx]);
    for (const ci of modelData.joints[idx].children) {
        for (const i of getSubtree(ci)) result.add(i);
    }
    return result;
}
let skeletonRoot = modelData.roots[0];
let maxGeo = 0;
for (const ri of modelData.roots) {
    const sub = getSubtree(ri);
    const geoCount = [...sub].filter(i => modelData.joints[i].meshes).length;
    if (geoCount > maxGeo) { maxGeo = geoCount; skeletonRoot = ri; }
}
const skeletonJoints = getSubtree(skeletonRoot);
console.log(`Skeleton root: ${skeletonRoot}, ${skeletonJoints.size} joints, ${maxGeo} with geometry`);
console.log(`Sphere group will have: ${[...skeletonJoints].length} spheres`);

// Only show spheres for joints with geometry + their direct parent chain
// This filters out mount points, collision boxes, and other structural joints
const geoJoints = new Set<number>();
for (let i = 0; i < modelData.joints.length; i++) {
    if (modelData.joints[i].meshes) {
        const verts = modelData.joints[i].meshes.reduce((s:number,m:any) => s + m.positions.length/3, 0);
        if (verts > 24) geoJoints.add(i); // skip collision boxes
    }
}
// Add parent chain for each geo joint (so structural joints between body parts show)
const parentOf:Record<number,number> = {};
for (let i = 0; i < modelData.joints.length; i++) {
    for (const c of modelData.joints[i].children) parentOf[c] = i;
}
const showJoints = new Set(geoJoints);
for (const gi of geoJoints) {
    let idx:number|undefined = parentOf[gi];
    while (idx !== undefined) {
        showJoints.add(idx);
        idx = parentOf[idx];
    }
}
console.log(`Showing ${showJoints.size} joints (${geoJoints.size} with geometry + parent chains)`);

const sphereGroup = new THREE.Group();
scene.add(sphereGroup);
const sphereScale = Math.max(size.x, size.y, size.z) * 0.025;
for (let i = 0; i < jointSpheres.length; i++) {
    const s = jointSpheres[i];
    if (!showJoints.has(i)) continue;
    s.scale.setScalar(sphereScale);
    const hasMesh = geoJoints.has(i);
    (s.material as THREE.MeshBasicMaterial).color.setHex(hasMesh ? 0xffaa00 : 0x4488ff);
    sphereGroup.add(s);
}
// Update sphere positions from joint world positions
const _worldPos = new THREE.Vector3();
function updateSpherePositions() {
    modelRoot.updateMatrixWorld(true);
    for (let i = 0; i < jointSpheres.length; i++) {
        const parentObj = jointSpheres[i].userData.parentObj as THREE.Object3D;
        parentObj.getWorldPosition(_worldPos);
        // Apply modelRoot's transform (centering offset)
        jointSpheres[i].position.copy(_worldPos);
    }
}

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 1.5));
scene.add(new THREE.HemisphereLight(0xffffff, 0x444466, 1.0));
const kl = new THREE.DirectionalLight(0xffffff, 2);
kl.position.set(size.x*2, size.y*3, size.z*2);
scene.add(kl);

// Camera
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 10000);
camera.position.set(size.z*1.5, size.y*1.2, size.z*2);
camera.lookAt(0,0,0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
viewEl.appendChild(renderer.domElement);
const controls = new TrackballControls(camera, renderer.domElement);

function resizeRenderer() {
    const w = viewEl.clientWidth, h = viewEl.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}
resizeRenderer();
window.addEventListener("resize", resizeRenderer);

// --- Raycasting for sphere hover/click ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function getHoveredJoint(event:MouseEvent):number|null {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(sphereGroup.children, false);
    return hits.length > 0 ? hits[0].object.userData.jointIdx : null;
}

renderer.domElement.addEventListener("mousemove", (e) => {
    hoveredModelIdx = getHoveredJoint(e);
    const info = document.getElementById("joint-info")!;
    if (hoveredModelIdx !== null) {
        const j = modelData.joints[hoveredModelIdx];
        info.style.display = "block";
        info.textContent = `[${hoveredModelIdx}] ${j.name||'(unnamed)'} ${j.meshes?'⬢':''}`;
    } else {
        info.style.display = "none";
    }
    updateSphereColors();
});

renderer.domElement.addEventListener("click", (e) => {
    const mi = getHoveredJoint(e);
    if (mi !== null) selectModelJoint(mi);
});

function selectModelJoint(mi:number) {
    if (selectedAnimIdx === null) {
        document.getElementById("status")!.textContent = "Select an anim joint first (left panel)";
        return;
    }
    const existing = mappings.findIndex(m => m.animIdx === selectedAnimIdx);
    if (existing >= 0) mappings.splice(existing, 1);
    mappings.push({ animIdx: selectedAnimIdx!, modelIdx: mi });
    const name = modelData.joints[mi].name || `joint_${mi}`;
    document.getElementById("status")!.textContent = `Mapped anim[${selectedAnimIdx}] → [${mi}] ${name}`;
    selectedAnimIdx = null;
    updateAll();
}

function updateSphereColors() {
    const mappedModelIndices = new Set(mappings.map(m => m.modelIdx));
    for (let i = 0; i < jointSpheres.length; i++) {
        const s = jointSpheres[i];
        if (mappedModelIndices.has(i)) (s.material as THREE.MeshBasicMaterial).color.setHex(0x44ff44);
        else if (i === hoveredModelIdx) (s.material as THREE.MeshBasicMaterial).color.setHex(0xffff00);
        else (s.material as THREE.MeshBasicMaterial).color.setHex(0x4488ff);
    }
}

// --- Tree rendering ---
function renderAnimTree() {
    const panel = document.getElementById("anim-tree")!;
    let html = `<h3>Anim (${animData.restPose.length})</h3>`;
    function renderJoint(idx:number, depth:number) {
        const j = animData.restPose[idx];
        const t = j.translation;
        const m = mappings.find(m => m.animIdx === idx);
        const cls = `joint ${m?'mapped':''} ${selectedAnimIdx===idx?'selected':''}`;
        const mapStr = m ? ` → [${m.modelIdx}] ${modelData.joints[m.modelIdx].name||''}` : '';
        html += `<div class="${cls}" data-anim="${idx}" style="padding-left:${depth*12+4}px">`;
        html += `<span class="idx">[${idx}]</span> `;
        html += `<span class="t">[${t[0].toFixed(1)},${t[1].toFixed(1)},${t[2].toFixed(1)}]</span>`;
        html += `<span style="color:#4f4">${mapStr}</span></div>`;
        for (const ci of (j.children || [])) renderJoint(ci, depth+1);
    }
    for (const r of (animData.restRoots || [0])) renderJoint(r, 0);
    panel.innerHTML = html;
    panel.querySelectorAll('.joint').forEach(el => {
        el.addEventListener('click', () => {
            selectedAnimIdx = parseInt(el.getAttribute('data-anim')!);
            document.getElementById("status")!.textContent = `Selected anim[${selectedAnimIdx}] — click model joint in 3D or right panel`;
            updateAll();
        });
    });
}

function renderModelTree() {
    const panel = document.getElementById("model-tree")!;
    let html = `<h3>Model (${modelData.joints.length})</h3>`;
    function renderJoint(idx:number, depth:number) {
        const j = modelData.joints[idx];
        const t = j.translation;
        const name = j.name || '';
        const isMapped = mappings.some(m => m.modelIdx === idx);
        const hasMesh = !!j.meshes;
        const cls = `joint ${isMapped?'mapped':''} ${hasMesh?'has-mesh':''}`;
        html += `<div class="${cls}" data-model="${idx}" style="padding-left:${depth*12+4}px">`;
        html += `<span class="idx">[${idx}]</span> ${name} `;
        html += `<span class="t">[${t[0].toFixed(1)},${t[1].toFixed(1)},${t[2].toFixed(1)}]</span>`;
        if (hasMesh) html += ` <span style="color:#ff0">⬢</span>`;
        html += `</div>`;
        for (const ci of j.children) renderJoint(ci, depth+1);
    }
    for (const r of modelData.roots) renderJoint(r, 0);
    panel.innerHTML = html;
    panel.querySelectorAll('.joint').forEach(el => {
        el.addEventListener('click', () => {
            selectModelJoint(parseInt(el.getAttribute('data-model')!));
        });
    });
}

function updateOutput() {
    const out = document.getElementById("mapping-output") as HTMLTextAreaElement;
    const map:Record<number,number> = {};
    for (const m of mappings) map[m.animIdx] = m.modelIdx;
    out.value = JSON.stringify(map, null, 2);
}

function updateAll() {
    renderAnimTree();
    renderModelTree();
    updateOutput();
    updateSphereColors();
}

// --- Controls ---
document.getElementById("btnClear")!.onclick = () => {
    mappings.length = 0; selectedAnimIdx = null; updateAll();
};
document.getElementById("btnExport")!.onclick = () => {
    const out = document.getElementById("mapping-output") as HTMLTextAreaElement;
    navigator.clipboard.writeText(out.value);
    document.getElementById("status")!.textContent = "Copied!";
};

// --- Animate ---
const animate = () => {
    requestAnimationFrame(animate);
    updateSpherePositions();
    controls.update();
    renderer.render(scene, camera);
};
animate();

updateAll();
