// HSD .dat binary viewer — browser-side parser + tree renderer

// ── Types ────────────────────────────────────────────────────────────────────

interface DatHeader {
    fileSize: number;
    dataBlockSize: number;
    relocCount: number;
    rootCount: number;
    secondaryRootCount: number;
}

interface RootNode {
    offset: number;
    name: string;
}

interface JointNode {
    offset: number;
    name: string;
    flags: number;
    rotation: [number, number, number];
    scale: [number, number, number];
    translation: [number, number, number];
    childOffset: number;
    siblingOffset: number;
    dobjOffset: number; // display object ptr
    hasGeometry: boolean;
    hasTexture: boolean;
    textureWidth: number;
    textureHeight: number;
    textureFormat: number;
    children: JointNode[];
}

interface ParsedDat {
    filename: string;
    header: DatHeader;
    relocOffsets: number[];
    roots: RootNode[];
    jointTrees: JointNode[]; // one per root that looks like a joint root
}

// ── Binary helpers ────────────────────────────────────────────────────────────

class Reader {
    private dv: DataView;
    readonly dataOffset: number;

    constructor(buf: ArrayBuffer, dataOffset: number) {
        this.dv = new DataView(buf);
        this.dataOffset = dataOffset;
    }

    u8(off: number) {
        return this.dv.getUint8(this.dataOffset + off);
    }
    u16(off: number) {
        return this.dv.getUint16(this.dataOffset + off, false);
    }
    u32(off: number) {
        return this.dv.getUint32(this.dataOffset + off, false);
    }
    s16(off: number) {
        return this.dv.getInt16(this.dataOffset + off, false);
    }
    f32(off: number) {
        return this.dv.getFloat32(this.dataOffset + off, false);
    }

    str(absOffset: number): string {
        const dv = this.dv;
        let s = "";
        for (let i = absOffset; i < dv.byteLength; i++) {
            const c = dv.getUint8(i);
            if (c === 0) break;
            s += String.fromCharCode(c);
        }
        return s;
    }
}

// ── Known texture format names ────────────────────────────────────────────────
const TEX_FORMATS: Record<number, string> = {
    0: "I4",
    1: "I8",
    2: "IA4",
    3: "IA8",
    4: "RGB565",
    5: "RGB5A3",
    6: "RGBA8",
    8: "CI4",
    9: "CI8",
    10: "CI14x2",
    14: "CMPR",
};

// ── Parse ─────────────────────────────────────────────────────────────────────

function parseDat(buf: ArrayBuffer, filename: string): ParsedDat {
    const dv = new DataView(buf);

    const fileSize = dv.getUint32(0x00, false);
    const dataBlockSize = dv.getUint32(0x04, false);
    const relocCount = dv.getUint32(0x08, false);
    const rootCount = dv.getUint32(0x0c, false);
    const secondaryRootCount = dv.getUint32(0x10, false);

    const dataOffset = 0x20;
    const relocOffset = dataOffset + dataBlockSize;
    const rootOffset = relocOffset + relocCount * 4;
    const strTableOffset = rootOffset + rootCount * 8;

    const r = new Reader(buf, dataOffset);

    // Relocation table
    const relocOffsets: number[] = [];
    for (let i = 0; i < relocCount; i++) {
        relocOffsets.push(dv.getUint32(relocOffset + i * 4, false));
    }

    // Root nodes
    const roots: RootNode[] = [];
    for (let i = 0; i < rootCount; i++) {
        const off = rootOffset + i * 8;
        const dataOff = dv.getUint32(off, false);
        const nameOff = dv.getUint32(off + 4, false);
        const name = r.str(strTableOffset + nameOff);
        roots.push({ offset: dataOff, name });
    }

    // Walk joint trees from each root
    const visited = new Set<number>();

    function parseJoint(
        off: number,
        nameMap: Map<number, string>,
    ): JointNode | null {
        if (off === 0 || off >= dataBlockSize) return null;

        const flags = r.u32(off + 0x00);
        const childOff = r.u32(off + 0x08);
        const siblingOff = r.u32(off + 0x0c);
        const dobjOff = r.u32(off + 0x10);
        const rx = r.f32(off + 0x14);
        const ry = r.f32(off + 0x18);
        const rz = r.f32(off + 0x1c);
        const sx = r.f32(off + 0x20);
        const sy = r.f32(off + 0x24);
        const sz = r.f32(off + 0x28);
        const tx = r.f32(off + 0x2c);
        const ty = r.f32(off + 0x30);
        const tz = r.f32(off + 0x34);

        let hasGeometry = false;
        let hasTexture = false;
        let texW = 0,
            texH = 0,
            texFmt = 0;

        // Walk dobj chain → mobj → tobj for texture info
        if (dobjOff !== 0 && dobjOff < dataBlockSize) {
            hasGeometry = true;
            let dptr = dobjOff;
            while (dptr !== 0 && dptr < dataBlockSize) {
                const mobjPtr = r.u32(dptr + 0x08);
                if (mobjPtr !== 0 && mobjPtr < dataBlockSize) {
                    const tobjPtr = r.u32(mobjPtr + 0x08);
                    if (tobjPtr !== 0 && tobjPtr < dataBlockSize) {
                        hasTexture = true;
                        const imgPtr = r.u32(tobjPtr + 0x3c);
                        if (imgPtr !== 0 && imgPtr < dataBlockSize) {
                            texW = r.u16(imgPtr + 0x00);
                            texH = r.u16(imgPtr + 0x02);
                            texFmt = r.u32(imgPtr + 0x04);
                        }
                    }
                }
                const next = r.u32(dptr + 0x00);
                if (next === dptr) break;
                dptr = next;
            }
        }

        visited.add(off);

        // Recurse children (depth-first, sibling last so tree renders correctly)
        const children: JointNode[] = [];
        if (childOff !== 0 && !visited.has(childOff)) {
            const child = parseJointTree(childOff, nameMap);
            if (child) children.push(child);
        }

        return {
            offset: off,
            name: nameMap.get(off) ?? "",
            flags,
            rotation: [rx, ry, rz],
            scale: [sx, sy, sz],
            translation: [tx, ty, tz],
            childOffset: childOff,
            siblingOffset: siblingOff,
            dobjOffset: dobjOff,
            hasGeometry,
            hasTexture,
            textureWidth: texW,
            textureHeight: texH,
            textureFormat: texFmt,
            children,
        };
    }

    // Iterative sibling-chained tree builder
    function parseJointTree(
        startOff: number,
        nameMap: Map<number, string>,
    ): JointNode | null {
        if (startOff === 0 || visited.has(startOff)) return null;
        const node = parseJoint(startOff, nameMap);
        if (!node) return null;

        // Chain siblings as children of the same parent (per HSD convention)
        let sibOff = node.siblingOffset;
        let current = node;
        while (sibOff !== 0 && !visited.has(sibOff) && sibOff < dataBlockSize) {
            visited.add(sibOff);
            const sib = parseJoint(sibOff, nameMap);
            if (!sib) break;
            // siblings are returned as a flat list; we attach them as the node's siblings
            current._sibling = sib;
            // Also recurse the sibling's sibling
            sibOff = sib.siblingOffset;
            current = sib;
        }
        return node;
    }

    // Build name map from all roots
    const nameMap = new Map<number, string>();
    for (const root of roots) nameMap.set(root.offset, root.name);

    // Build joint trees from roots that look like joints (stride 0x40, valid floats)
    const jointTrees: JointNode[] = [];
    for (const root of roots) {
        if (!root.name.endsWith("_joint") && !root.name.endsWith("_Joint"))
            continue;
        if (visited.has(root.offset)) continue;
        const tree = parseJointTree(root.offset, nameMap);
        if (tree) jointTrees.push(tree);
    }

    return {
        filename,
        header: {
            fileSize,
            dataBlockSize,
            relocCount,
            rootCount,
            secondaryRootCount,
        },
        relocOffsets,
        roots,
        jointTrees,
    };
}

// Attach sibling as a property for tree flattening
declare module "./parser" {}
// @ts-ignore — extend our interface inline
interface JointNode {
    _sibling?: JointNode;
}

// ── Flatten sibling chains into arrays ───────────────────────────────────────

function flattenSiblings(node: JointNode): JointNode[] {
    const result: JointNode[] = [node];
    // @ts-ignore
    if (node._sibling) result.push(...flattenSiblings(node._sibling));
    return result;
}

// ── Render ────────────────────────────────────────────────────────────────────

const sidebar = document.getElementById("sidebar")!;
const detail = document.getElementById("detail")!;

function deg(r: number) {
    return ((r * 180) / Math.PI).toFixed(2) + "°";
}
function fmt3(v: [number, number, number]) {
    return v.map((n) => n.toFixed(4)).join(", ");
}
function hex(n: number) {
    return "0x" + n.toString(16).toUpperCase().padStart(6, "0");
}

function renderDetail(node: JointNode) {
    const texName =
        TEX_FORMATS[node.textureFormat] ?? `fmt${node.textureFormat}`;
    const badges = [
        node.hasGeometry && !node.hasTexture
            ? `<span class="badge badge-geo">geo</span>`
            : "",
        node.hasTexture
            ? `<span class="badge badge-tex">tex ${node.textureWidth}×${node.textureHeight} ${texName}</span>`
            : "",
        !node.hasGeometry ? `<span class="badge badge-mount">mount</span>` : "",
    ].join("");

    detail.innerHTML = `
    <h2>${node.name || "<unnamed>"} ${badges}</h2>

    <div class="section">
      <div class="section-title">Address</div>
      <table><tr><td>offset</td><td class="hex">${hex(node.offset)}</td></tr></table>
    </div>

    <div class="section">
      <div class="section-title">Transform</div>
      <table>
        <tr><td>rotation</td><td>${deg(node.rotation[0])}, ${deg(node.rotation[1])}, ${deg(node.rotation[2])}</td></tr>
        <tr><td>scale</td>   <td>${fmt3(node.scale)}</td></tr>
        <tr><td>translation</td><td>${fmt3(node.translation)}</td></tr>
      </table>
    </div>

    <div class="section">
      <div class="section-title">Pointers</div>
      <table>
        <tr><td>child</td>  <td class="hex">${hex(node.childOffset)}</td></tr>
        <tr><td>sibling</td><td class="hex">${hex(node.siblingOffset)}</td></tr>
        <tr><td>dobj</td>   <td class="hex">${hex(node.dobjOffset)}</td></tr>
      </table>
    </div>

    <div class="section">
      <div class="section-title">Flags</div>
      <table><tr><td>flags</td><td class="hex">${hex(node.flags)}</td></tr></table>
    </div>
  `;
}

function renderHeaderDetail(dat: ParsedDat) {
    const h = dat.header;
    detail.innerHTML = `
    <h2>${dat.filename}</h2>

    <div class="section">
      <div class="section-title">Header</div>
      <table>
        <tr><td>file size</td>        <td>${h.fileSize.toLocaleString()} bytes</td></tr>
        <tr><td>data block</td>       <td>${h.dataBlockSize.toLocaleString()} bytes</td></tr>
        <tr><td>reloc entries</td>    <td>${h.relocCount}</td></tr>
        <tr><td>root nodes</td>       <td>${h.rootCount}</td></tr>
        <tr><td>secondary roots</td>  <td>${h.secondaryRootCount}</td></tr>
      </table>
    </div>

    <div class="section">
      <div class="section-title">Root Nodes (${dat.roots.length})</div>
      <table>
        ${dat.roots.map((r) => `<tr><td class="hex">${hex(r.offset)}</td><td>${r.name || "<unnamed>"}</td></tr>`).join("")}
      </table>
    </div>
  `;
}

function buildTreeEl(
    nodes: JointNode[],
    onSelect: (n: JointNode) => void,
): HTMLElement {
    const container = document.createElement("div");

    for (const node of nodes) {
        const siblings = flattenSiblings(node);
        for (const n of siblings) {
            const wrap = document.createElement("div");
            wrap.className = "tree-root";

            const allChildren: JointNode[] = [];
            for (const child of n.children)
                allChildren.push(...flattenSiblings(child));

            const hasKids = allChildren.length > 0;
            let expanded = false;

            const row = document.createElement("div");
            row.className = "tree-node";
            row.innerHTML = `
        <span class="tree-toggle">${hasKids ? "▶" : " "}</span>
        <span class="tree-label ${n.name ? "" : "unnamed"}">${n.name || "[" + hex(n.offset) + "]"}</span>
        <span class="tree-offset">${hex(n.offset)}</span>
      `;

            const childContainer = document.createElement("div");
            childContainer.className = "tree-children";
            childContainer.style.display = "none";

            if (hasKids) {
                row.addEventListener("click", (e) => {
                    e.stopPropagation();
                    expanded = !expanded;
                    childContainer.style.display = expanded ? "" : "none";
                    row.querySelector(".tree-toggle")!.textContent = expanded
                        ? "▼"
                        : "▶";
                    row.classList.toggle("selected", true);
                    onSelect(n);
                });
                const subtree = buildTreeEl(allChildren, onSelect);
                childContainer.appendChild(subtree);
            } else {
                row.addEventListener("click", (e) => {
                    e.stopPropagation();
                    document
                        .querySelectorAll(".tree-node.selected")
                        .forEach((el) => el.classList.remove("selected"));
                    row.classList.add("selected");
                    onSelect(n);
                });
            }

            wrap.appendChild(row);
            if (hasKids) wrap.appendChild(childContainer);
            container.appendChild(wrap);
        }
    }

    return container;
}

function render(dat: ParsedDat) {
    sidebar.innerHTML = "";

    // File summary row
    const summary = document.createElement("div");
    summary.style.cssText =
        "padding:6px 4px 10px;border-bottom:1px solid #21262d;margin-bottom:6px;cursor:pointer;";
    summary.innerHTML = `<span style="color:#3fb950">▸ ${dat.filename}</span> <span style="color:#484f58">${dat.header.fileSize.toLocaleString()}b · ${dat.header.rootCount} roots</span>`;
    summary.addEventListener("click", () => renderHeaderDetail(dat));
    sidebar.appendChild(summary);

    if (dat.jointTrees.length > 0) {
        const treeEl = buildTreeEl(dat.jointTrees, renderDetail);
        sidebar.appendChild(treeEl);
    } else {
        // No joint roots — just list all roots
        const list = document.createElement("div");
        list.innerHTML = dat.roots
            .map(
                (r) =>
                    `<div class="tree-node"><span class="tree-label unnamed">${r.name || "<unnamed>"}</span><span class="tree-offset">${hex(r.offset)}</span></div>`,
            )
            .join("");
        sidebar.appendChild(list);
    }

    renderHeaderDetail(dat);
}

// ── File loading ──────────────────────────────────────────────────────────────

async function loadFile(file: File) {
    const buf = await file.arrayBuffer();
    const dat = parseDat(buf, file.name);
    render(dat);
}

document.getElementById("fileInput")!.addEventListener("change", (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (f) loadFile(f);
});

document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files[0];
    if (f?.name.endsWith(".dat")) loadFile(f);
});
