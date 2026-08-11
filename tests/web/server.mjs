import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const host = "127.0.0.1";
const port = Number(process.env.GLYPH_FORGE_WEB_PORT || 4173);
const prefix = "/glyph_forge/";
const root = resolve(fileURLToPath(new URL("../../src/glyph_forge/ui/web/", import.meta.url)));
const mediaTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".png", "image/png"],
  [".svg", "image/svg+xml"],
  [".webmanifest", "application/manifest+json"],
]);

function secureHeaders(response) {
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; style-src 'self'; script-src 'self'; worker-src 'self'; manifest-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'");
  response.setHeader("X-Content-Type-Options", "nosniff");
}

const server = createServer(async (request, response) => {
  secureHeaders(response);
  const url = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);
  if (url.pathname === "/" || url.pathname === "/glyph_forge") {
    response.writeHead(302, { Location: prefix });
    response.end();
    return;
  }
  if (url.pathname === `${prefix}api/config`) {
    response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.end(JSON.stringify({
      share_links: false,
      public_base_url: null,
      default_ttl_seconds: null,
      max_upload_bytes: 0,
      csrf_token: null,
    }));
    return;
  }
  if (!url.pathname.startsWith(prefix)) {
    response.writeHead(404);
    response.end("Not found\n");
    return;
  }
  const relative = decodeURIComponent(url.pathname.slice(prefix.length)) || "index.html";
  const path = resolve(root, relative);
  if (path !== root && !path.startsWith(`${root}${sep}`)) {
    response.writeHead(404);
    response.end("Not found\n");
    return;
  }
  try {
    const info = await stat(path);
    if (!info.isFile()) throw new Error("Not a file");
    response.setHeader("Content-Type", mediaTypes.get(extname(path)) || "application/octet-stream");
    response.setHeader("Content-Length", info.size);
    response.writeHead(200);
    if (request.method === "HEAD") response.end();
    else createReadStream(path).pipe(response);
  } catch {
    response.writeHead(404);
    response.end("Not found\n");
  }
});

server.listen(port, host, () => {
  process.stdout.write(`Glyph Forge web test server: http://localhost:${port}${prefix}\n`);
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}
