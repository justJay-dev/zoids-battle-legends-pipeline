/// <reference types="bun-types" />

const PORT = 8080;
const ROOT = import.meta.dir;

// Transpiler for TypeScript → JavaScript (no build step, no tsc)
const transpiler = new Bun.Transpiler({ loader: "ts" });

const MIME: Record<string, string> = {
    ".mjs": "text/javascript",
    ".mts": "text/javascript",
    ".js": "text/javascript",
    ".dat": "application/octet-stream",
    ".json": "application/json",
};

function extOf(p: string) {
    return p.slice(p.lastIndexOf("."));
}

const server = Bun.serve({
    port: PORT,

    async fetch(req) {
        const url = new URL(req.url);
        let pathname = url.pathname;

        if (pathname === "/" || pathname.endsWith("/"))
            pathname = pathname + "index.html";
        if (pathname === "//index.html") pathname = "/index.html";

        const filePath = `${ROOT}${pathname}`;
        const file = Bun.file(filePath);

        if (!(await file.exists())) {
            return new Response("Not found", { status: 404 });
        }

        // Serve HTML as-is
        if (pathname.endsWith(".html")) {
            return new Response(file, {
                headers: { "Content-Type": "text/html; charset=utf-8" },
            });
        }

        // Transpile TypeScript on the fly
        if (pathname.endsWith(".mts") || pathname.endsWith(".ts")) {
            const source = await file.text();
            const js = transpiler.transformSync(source);
            return new Response(js, {
                headers: { "Content-Type": "text/javascript; charset=utf-8" },
            });
        }

        // Serve node_modules JS files with correct MIME type
        if (pathname.startsWith("/node_modules/") && pathname.endsWith(".js")) {
            return new Response(file, {
                headers: { "Content-Type": "text/javascript; charset=utf-8" },
            });
        }

        const ext = extOf(pathname);
        const contentType = MIME[ext] ?? undefined;
        return new Response(file, {
            headers: contentType ? { "Content-Type": contentType } : {},
        });
    },
});

console.log(`[zoids] serving → http://localhost:${PORT}`);
