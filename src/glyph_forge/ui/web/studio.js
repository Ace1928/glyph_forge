"use strict";

const CHARSETS = Object.freeze({
  detailed: " .,:;irsXA253hMHGS#9B&@",
  general: " .,:;i1tfLCG08@",
  blocks: " ░▒▓█",
  minimal: " .-=+*#%@",
  matrix: " 01+*%#@アイウエオカキクケコ",
  stipple: " ·•○◎●◉",
  binary: " 01",
  cosmic: " ·˚。✧⋆✦✩★",
});

const $ = (id) => document.getElementById(id);
const elements = {
  canvas: $("outputCanvas"),
  sampler: $("sampleCanvas"),
  video: $("videoSource"),
  fileInput: $("fileInput"),
  dropZone: $("dropZone"),
  dropOverlay: $("dropOverlay"),
  emptyState: $("emptyState"),
  emptyOpen: $("emptyOpenButton"),
  webcam: $("webcamButton"),
  screen: $("screenButton"),
  stop: $("stopButton"),
  charset: $("charsetSelect"),
  customField: $("customCharsetField"),
  customCharset: $("customCharset"),
  font: $("fontSelect"),
  columns: $("columnsRange"),
  columnsValue: $("columnsValue"),
  resolution: $("resolutionSelect"),
  colorMode: $("colorMode"),
  foreground: $("foregroundColor"),
  background: $("backgroundColor"),
  brightness: $("brightnessRange"),
  brightnessValue: $("brightnessValue"),
  contrast: $("contrastRange"),
  contrastValue: $("contrastValue"),
  invert: $("invertToggle"),
  png: $("pngButton"),
  svg: $("svgButton"),
  text: $("textButton"),
  share: $("shareButton"),
  publish: $("publishButton"),
  link: $("linkButton"),
  sourceName: $("sourceName"),
  sourceMeta: $("sourceMeta"),
  fps: $("fpsMetric"),
  grid: $("gridMetric"),
  engine: $("engineMetric"),
  status: $("statusMessage"),
};

const state = {
  source: null,
  sourceKind: null,
  sourceName: "",
  objectUrl: null,
  generation: 0,
  frameTimes: [],
  renderer: null,
  dimensions: { width: 1280, height: 720, rows: 72 },
  shareConfig: {
    enabled: false,
    csrfToken: null,
    maxUploadBytes: 0,
    ttlSeconds: 0,
  },
};

function hexToRgb(hex) {
  const value = Number.parseInt(hex.slice(1), 16);
  return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
}

function escapeXml(value) {
  return value.replace(/[<>&'\"]/g, (character) => ({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "'": "&apos;",
    '"': "&quot;",
  })[character]);
}

function activeCharset() {
  const selected = elements.charset.value;
  const raw = selected === "custom" ? elements.customCharset.value : CHARSETS[selected];
  const glyphs = Array.from(raw || " .#").slice(0, 128);
  return glyphs.length ? glyphs.join("") : " .#";
}

function sourceSize() {
  if (!state.source) return { width: 1280, height: 720 };
  if (state.sourceKind === "image") {
    return { width: state.source.naturalWidth, height: state.source.naturalHeight };
  }
  return { width: state.source.videoWidth || 1280, height: state.source.videoHeight || 720 };
}

function adaptiveHeight() {
  const memory = navigator.deviceMemory || 4;
  const cores = navigator.hardwareConcurrency || 4;
  if (memory >= 12 && cores >= 12) return 1080;
  if (memory <= 3 || cores <= 2) return 540;
  return 720;
}

function targetDimensions() {
  const source = sourceSize();
  const aspect = Math.max(0.1, source.width / Math.max(1, source.height));
  let height;
  if (elements.resolution.value === "source") {
    height = source.height;
  } else if (elements.resolution.value === "auto") {
    height = Math.min(source.height || 720, adaptiveHeight());
  } else {
    height = Number(elements.resolution.value);
  }
  height = Math.max(180, Math.min(2160, Math.round(height / 2) * 2));
  let width = Math.max(320, Math.round((height * aspect) / 2) * 2);
  if (width > 4096) {
    width = 4096;
    height = Math.max(180, Math.round((width / aspect) / 2) * 2);
  }
  const columns = Number(elements.columns.value);
  const rows = Math.max(1, Math.round(columns / aspect));
  state.dimensions = { width, height, rows };
  elements.grid.textContent = `${columns}×${rows}`;
  return state.dimensions;
}

function currentOptions() {
  const dimensions = targetDimensions();
  return {
    ...dimensions,
    columns: Number(elements.columns.value),
    charset: activeCharset(),
    font: elements.font.value,
    sourceColor: elements.colorMode.value === "source",
    foreground: hexToRgb(elements.foreground.value),
    background: hexToRgb(elements.background.value),
    brightness: Number(elements.brightness.value),
    contrast: Number(elements.contrast.value),
    invert: elements.invert.checked,
  };
}

class WebGLGlyphRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    if (!this.gl) throw new Error("WebGL2 is unavailable");
    this.program = this.createProgram();
    this.uniforms = {};
    for (const name of [
      "u_source", "u_atlas", "u_grid", "u_glyph_count", "u_background",
      "u_foreground", "u_source_color", "u_brightness", "u_contrast", "u_invert",
    ]) {
      this.uniforms[name] = this.gl.getUniformLocation(this.program, name);
    }
    this.createGeometry();
    this.sourceTexture = this.createTexture();
    this.atlasTexture = this.createTexture();
    this.atlasKey = "";
  }

  compile(type, source) {
    const shader = this.gl.createShader(type);
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      throw new Error(this.gl.getShaderInfoLog(shader) || "Shader compilation failed");
    }
    return shader;
  }

  createProgram() {
    const vertex = this.compile(this.gl.VERTEX_SHADER, `#version 300 es
      in vec2 a_position;
      out vec2 v_uv;
      void main() {
        v_uv = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `);
    const fragment = this.compile(this.gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float;
      in vec2 v_uv;
      out vec4 out_color;
      uniform sampler2D u_source;
      uniform sampler2D u_atlas;
      uniform vec2 u_grid;
      uniform float u_glyph_count;
      uniform vec3 u_background;
      uniform vec3 u_foreground;
      uniform bool u_source_color;
      uniform float u_brightness;
      uniform float u_contrast;
      uniform bool u_invert;
      void main() {
        vec2 cell = floor(v_uv * u_grid);
        vec2 sample_uv = (cell + 0.5) / u_grid;
        vec3 source_color = texture(u_source, sample_uv).rgb;
        source_color = clamp((source_color - 0.5) * u_contrast + 0.5, 0.0, 1.0);
        source_color = clamp(source_color * u_brightness, 0.0, 1.0);
        float luma = dot(source_color, vec3(0.299, 0.587, 0.114));
        if (u_invert) luma = 1.0 - luma;
        float glyph = floor(clamp(luma, 0.0, 0.99999) * u_glyph_count);
        vec2 local = fract(v_uv * u_grid);
        vec2 atlas_uv = vec2((glyph + local.x) / u_glyph_count, local.y);
        float alpha = texture(u_atlas, atlas_uv).r;
        vec3 ink = u_source_color ? source_color : u_foreground;
        out_color = vec4(mix(u_background, ink, alpha), 1.0);
      }
    `);
    const program = this.gl.createProgram();
    this.gl.attachShader(program, vertex);
    this.gl.attachShader(program, fragment);
    this.gl.linkProgram(program);
    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      throw new Error(this.gl.getProgramInfoLog(program) || "Shader linking failed");
    }
    this.gl.deleteShader(vertex);
    this.gl.deleteShader(fragment);
    return program;
  }

  createGeometry() {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const location = gl.getAttribLocation(this.program, "a_position");
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
    this.vao = vao;
  }

  createTexture() {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  updateAtlas(options) {
    const key = `${options.charset}|${options.font}`;
    if (key === this.atlasKey) return;
    const glyphs = Array.from(options.charset);
    const cell = 48;
    const atlas = document.createElement("canvas");
    atlas.width = cell * glyphs.length;
    atlas.height = cell;
    const context = atlas.getContext("2d", { alpha: false });
    context.fillStyle = "black";
    context.fillRect(0, 0, atlas.width, atlas.height);
    context.fillStyle = "white";
    context.font = `700 ${Math.round(cell * 0.82)}px ${options.font}`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    glyphs.forEach((glyph, index) => context.fillText(glyph, index * cell + cell / 2, cell * 0.52));
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.atlasTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, atlas);
    this.atlasKey = key;
  }

  draw(source, options) {
    const gl = this.gl;
    if (this.canvas.width !== options.width || this.canvas.height !== options.height) {
      this.canvas.width = options.width;
      this.canvas.height = options.height;
    }
    this.updateAtlas(options);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sourceTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.uniform1i(this.uniforms.u_source, 0);
    gl.uniform1i(this.uniforms.u_atlas, 1);
    gl.uniform2f(this.uniforms.u_grid, options.columns, options.rows);
    gl.uniform1f(this.uniforms.u_glyph_count, Array.from(options.charset).length);
    gl.uniform3fv(this.uniforms.u_background, options.background);
    gl.uniform3fv(this.uniforms.u_foreground, options.foreground);
    gl.uniform1i(this.uniforms.u_source_color, options.sourceColor ? 1 : 0);
    gl.uniform1f(this.uniforms.u_brightness, options.brightness);
    gl.uniform1f(this.uniforms.u_contrast, options.contrast);
    gl.uniform1i(this.uniforms.u_invert, options.invert ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

class CanvasGlyphRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d", { alpha: false });
    this.sample = document.createElement("canvas");
    this.sampleContext = this.sample.getContext("2d", { willReadFrequently: true });
  }

  draw(source, options) {
    if (this.canvas.width !== options.width || this.canvas.height !== options.height) {
      this.canvas.width = options.width;
      this.canvas.height = options.height;
    }
    this.sample.width = options.columns;
    this.sample.height = options.rows;
    this.sampleContext.drawImage(source, 0, 0, options.columns, options.rows);
    const pixels = this.sampleContext.getImageData(0, 0, options.columns, options.rows).data;
    const glyphs = Array.from(options.charset);
    const cellWidth = options.width / options.columns;
    const cellHeight = options.height / options.rows;
    this.context.fillStyle = elements.background.value;
    this.context.fillRect(0, 0, options.width, options.height);
    this.context.font = `700 ${Math.ceil(cellHeight * 0.9)}px ${options.font}`;
    this.context.textAlign = "center";
    this.context.textBaseline = "middle";
    for (let row = 0; row < options.rows; row += 1) {
      for (let column = 0; column < options.columns; column += 1) {
        const offset = (row * options.columns + column) * 4;
        let red = pixels[offset] / 255;
        let green = pixels[offset + 1] / 255;
        let blue = pixels[offset + 2] / 255;
        red = Math.max(0, Math.min(1, ((red - 0.5) * options.contrast + 0.5) * options.brightness));
        green = Math.max(0, Math.min(1, ((green - 0.5) * options.contrast + 0.5) * options.brightness));
        blue = Math.max(0, Math.min(1, ((blue - 0.5) * options.contrast + 0.5) * options.brightness));
        let luma = red * 0.299 + green * 0.587 + blue * 0.114;
        if (options.invert) luma = 1 - luma;
        const glyph = glyphs[Math.min(glyphs.length - 1, Math.floor(luma * glyphs.length))];
        this.context.fillStyle = options.sourceColor
          ? `rgb(${red * 255} ${green * 255} ${blue * 255})`
          : elements.foreground.value;
        this.context.fillText(glyph, (column + 0.5) * cellWidth, (row + 0.52) * cellHeight);
      }
    }
  }
}

function initializeRenderer() {
  try {
    state.renderer = new WebGLGlyphRenderer(elements.canvas);
    elements.engine.textContent = "WebGL2 GPU";
  } catch (error) {
    console.warn("WebGL2 renderer unavailable; using Canvas 2D", error);
    state.renderer = new CanvasGlyphRenderer(elements.canvas);
    elements.engine.textContent = "Canvas 2D";
  }
}

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.style.color = isError ? "var(--danger)" : "";
}

function enableExports(enabled) {
  for (const button of [elements.png, elements.svg, elements.text, elements.share]) {
    button.disabled = !enabled;
  }
  elements.publish.disabled = !enabled || !state.shareConfig.enabled;
  elements.stop.disabled = !enabled;
}

function updateMetrics(timestamp) {
  state.frameTimes.push(timestamp);
  const cutoff = timestamp - 1000;
  while (state.frameTimes.length && state.frameTimes[0] < cutoff) state.frameTimes.shift();
  elements.fps.textContent = String(Math.max(0, state.frameTimes.length - 1));
}

function renderCurrent(timestamp = performance.now()) {
  if (!state.source || !state.renderer) return;
  try {
    state.renderer.draw(state.source, currentOptions());
    updateMetrics(timestamp);
  } catch (error) {
    console.error(error);
    setStatus(`Render failed: ${error.message}`, true);
  }
}

function beginRenderLoop() {
  state.generation += 1;
  const generation = state.generation;
  state.frameTimes = [];
  const frame = (timestamp) => {
    if (generation !== state.generation || !state.source) return;
    renderCurrent(timestamp);
    if (state.sourceKind !== "image") requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function describeSource(name) {
  const size = sourceSize();
  elements.sourceName.textContent = name;
  elements.sourceMeta.textContent = `${size.width}×${size.height} · ${state.sourceKind}`;
  elements.emptyState.classList.add("hidden");
  enableExports(true);
  setStatus("Rendering locally. No media has been uploaded.");
}

function stopSource({ reset = true } = {}) {
  state.generation += 1;
  if (elements.video.srcObject) {
    for (const track of elements.video.srcObject.getTracks()) track.stop();
    elements.video.srcObject = null;
  }
  elements.video.pause();
  elements.video.removeAttribute("src");
  elements.video.load();
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
  state.source = null;
  state.sourceKind = null;
  state.frameTimes = [];
  elements.fps.textContent = "0";
  if (reset) {
    elements.emptyState.classList.remove("hidden");
    elements.sourceName.textContent = "No source selected";
    elements.sourceMeta.textContent = "Drop a file or start a live source";
    enableExports(false);
    setStatus("Ready. Nothing is uploaded.");
  }
}

function waitForVideo(video) {
  return new Promise((resolve, reject) => {
    const loaded = () => {
      cleanup();
      resolve();
    };
    const failed = () => {
      cleanup();
      reject(new Error("The browser could not decode this video"));
    };
    const cleanup = () => {
      video.removeEventListener("loadedmetadata", loaded);
      video.removeEventListener("error", failed);
    };
    video.addEventListener("loadedmetadata", loaded, { once: true });
    video.addEventListener("error", failed, { once: true });
  });
}

async function openFile(file) {
  if (!file) return;
  stopSource({ reset: false });
  try {
    state.objectUrl = URL.createObjectURL(file);
    if (file.type.startsWith("video/")) {
      state.sourceKind = "video";
      elements.video.src = state.objectUrl;
      elements.video.loop = true;
      await waitForVideo(elements.video);
      await elements.video.play();
      state.source = elements.video;
    } else if (file.type.startsWith("image/")) {
      state.sourceKind = "image";
      const image = new Image();
      image.src = state.objectUrl;
      await image.decode();
      state.source = image;
    } else {
      throw new Error("Choose an image or video file");
    }
    state.sourceName = file.name;
    describeSource(file.name);
    beginRenderLoop();
  } catch (error) {
    stopSource();
    setStatus(error.message, true);
  }
}

async function openLive(kind) {
  stopSource({ reset: false });
  try {
    let stream;
    if (kind === "screen") {
      if (!navigator.mediaDevices?.getDisplayMedia) throw new Error("Screen capture is unavailable in this browser");
      stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: { ideal: 30, max: 60 } }, audio: false });
    } else {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Webcam capture is unavailable in this browser");
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30, max: 60 } },
        audio: false,
      });
    }
    state.sourceKind = kind;
    elements.video.srcObject = stream;
    await waitForVideo(elements.video);
    await elements.video.play();
    state.source = elements.video;
    state.sourceName = kind === "screen" ? "Shared screen" : "Webcam";
    const [track] = stream.getVideoTracks();
    if (track) track.addEventListener("ended", () => stopSource(), { once: true });
    describeSource(state.sourceName);
    beginRenderLoop();
  } catch (error) {
    stopSource();
    setStatus(`${kind === "screen" ? "Screen" : "Webcam"} access failed: ${error.message}`, true);
  }
}

function sampleTextFrame() {
  if (!state.source) return { lines: [], colors: [], options: currentOptions() };
  const options = currentOptions();
  elements.sampler.width = options.columns;
  elements.sampler.height = options.rows;
  const context = elements.sampler.getContext("2d", { willReadFrequently: true });
  context.drawImage(state.source, 0, 0, options.columns, options.rows);
  const pixels = context.getImageData(0, 0, options.columns, options.rows).data;
  const glyphs = Array.from(options.charset);
  const lines = [];
  const colors = [];
  for (let row = 0; row < options.rows; row += 1) {
    let line = "";
    const rowColors = [];
    for (let column = 0; column < options.columns; column += 1) {
      const offset = (row * options.columns + column) * 4;
      const red = Math.max(0, Math.min(255, ((pixels[offset] / 255 - 0.5) * options.contrast + 0.5) * options.brightness * 255));
      const green = Math.max(0, Math.min(255, ((pixels[offset + 1] / 255 - 0.5) * options.contrast + 0.5) * options.brightness * 255));
      const blue = Math.max(0, Math.min(255, ((pixels[offset + 2] / 255 - 0.5) * options.contrast + 0.5) * options.brightness * 255));
      let luma = (red * 0.299 + green * 0.587 + blue * 0.114) / 255;
      if (options.invert) luma = 1 - luma;
      line += glyphs[Math.min(glyphs.length - 1, Math.floor(luma * glyphs.length))];
      rowColors.push([Math.round(red), Math.round(green), Math.round(blue)]);
    }
    lines.push(line);
    colors.push(rowColors);
  }
  return { lines, colors, options };
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function outputName(extension) {
  const base = (state.sourceName || "glyph-forge").replace(/\.[^.]+$/, "").replace(/[^a-z0-9_-]+/gi, "-");
  return `${base || "glyph-forge"}.glyph.${extension}`;
}

function canvasBlob(type = "image/png") {
  return new Promise((resolve, reject) => {
    renderCurrent();
    elements.canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Canvas export failed")), type);
  });
}

async function savePng() {
  try {
    download(await canvasBlob(), outputName("png"));
    setStatus("PNG saved.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function saveText() {
  const { lines } = sampleTextFrame();
  download(new Blob([`${lines.join("\n")}\n`], { type: "text/plain;charset=utf-8" }), outputName("txt"));
  setStatus("Text glyph frame saved.");
}

function saveSvg() {
  const { lines, options } = sampleTextFrame();
  const fontSize = 12;
  const lineHeight = fontSize * 1.05;
  const width = options.columns * fontSize * 0.62;
  const height = options.rows * lineHeight;
  const family = escapeXml(options.font);
  const texts = lines.map((line, index) => `<text x="0" y="${((index + 1) * lineHeight).toFixed(2)}">${escapeXml(line)}</text>`).join("\n");
  const svg = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${width.toFixed(2)}" height="${height.toFixed(2)}" viewBox="0 0 ${width.toFixed(2)} ${height.toFixed(2)}" role="img" aria-label="Glyph Forge render">\n<rect width="100%" height="100%" fill="${elements.background.value}"/>\n<g fill="${elements.foreground.value}" font-family="${family}" font-size="${fontSize}" xml:space="preserve">\n${texts}\n</g>\n</svg>\n`;
  download(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }), outputName("svg"));
  setStatus("Scalable SVG saved with real text glyphs.");
}

function settingsHash() {
  const values = new URLSearchParams({
    style: elements.charset.value,
    glyphs: elements.charset.value === "custom" ? elements.customCharset.value : "",
    columns: elements.columns.value,
    resolution: elements.resolution.value,
    color: elements.colorMode.value,
    ink: elements.foreground.value.slice(1),
    canvas: elements.background.value.slice(1),
    brightness: elements.brightness.value,
    contrast: elements.contrast.value,
    invert: elements.invert.checked ? "1" : "0",
    font: elements.font.selectedIndex.toString(),
  });
  return values.toString();
}

async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const area = document.createElement("textarea");
  area.value = value;
  area.className = "visually-hidden";
  document.body.append(area);
  area.select();
  document.execCommand("copy");
  area.remove();
}

async function copyStyleLink() {
  const url = new URL(window.location.href);
  url.hash = settingsHash();
  history.replaceState(null, "", url);
  try {
    await copyText(url.toString());
    setStatus("Style link copied. Media remains private and is not embedded.");
  } catch (error) {
    setStatus(`Could not copy link: ${error.message}`, true);
  }
}

async function shareOutput() {
  try {
    const blob = await canvasBlob();
    const file = new File([blob], outputName("png"), { type: "image/png" });
    if (navigator.share && (!navigator.canShare || navigator.canShare({ files: [file] }))) {
      await navigator.share({ title: "Glyph Forge", text: "Forged locally with Glyph Forge", files: [file] });
      setStatus("Shared.");
    } else {
      download(blob, file.name);
      setStatus("Web Share is unavailable, so the PNG was saved instead.");
    }
  } catch (error) {
    if (error.name !== "AbortError") setStatus(`Share failed: ${error.message}`, true);
  }
}

async function loadStudioConfig() {
  try {
    const response = await fetch("/api/config", {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error(`server returned ${response.status}`);
    const config = await response.json();
    state.shareConfig = {
      enabled: config.share_links === true,
      csrfToken: config.csrf_token || null,
      maxUploadBytes: Number(config.max_upload_bytes) || 0,
      ttlSeconds: Number(config.default_ttl_seconds) || 0,
    };
    elements.publish.classList.toggle("hidden", !state.shareConfig.enabled);
    elements.publish.disabled = !state.source || !state.shareConfig.enabled;
    if (state.shareConfig.enabled) {
      elements.publish.title = `Link expires after ${state.shareConfig.ttlSeconds} seconds or when Glyph Forge stops`;
    }
  } catch (error) {
    console.warn("Studio sharing configuration is unavailable", error);
    elements.publish.classList.add("hidden");
    state.shareConfig.enabled = false;
  }
}

async function publishLink() {
  if (!state.shareConfig.enabled || !state.shareConfig.csrfToken) {
    setStatus("Temporary link sharing is not enabled. Restart Studio with --lan or --share-links.", true);
    return;
  }
  elements.publish.disabled = true;
  try {
    const blob = await canvasBlob();
    if (blob.size > state.shareConfig.maxUploadBytes) {
      const limit = (state.shareConfig.maxUploadBytes / (1024 * 1024)).toFixed(1);
      throw new Error(`PNG is larger than the ${limit} MiB link limit`);
    }
    const filename = outputName("png");
    const response = await fetch(`/api/share?name=${encodeURIComponent(filename)}`, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "image/png",
        "X-Glyph-Forge-Token": state.shareConfig.csrfToken,
      },
      body: blob,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `server returned ${response.status}`);
    await copyText(result.url);
    setStatus(`Temporary link copied. It expires in ${state.shareConfig.ttlSeconds} seconds or when Glyph Forge stops.`);
  } catch (error) {
    setStatus(`Could not publish link: ${error.message}`, true);
  } finally {
    elements.publish.disabled = !state.source || !state.shareConfig.enabled;
  }
}

function restoreSettings() {
  if (!window.location.hash) return;
  const values = new URLSearchParams(window.location.hash.slice(1));
  const assign = (element, key, validate = () => true) => {
    const value = values.get(key);
    if (value !== null && validate(value)) element.value = value;
  };
  assign(elements.charset, "style", (value) => [...elements.charset.options].some((option) => option.value === value));
  assign(elements.customCharset, "glyphs");
  assign(elements.columns, "columns", (value) => Number(value) >= 32 && Number(value) <= 240);
  assign(elements.resolution, "resolution", (value) => [...elements.resolution.options].some((option) => option.value === value));
  assign(elements.colorMode, "color");
  const ink = values.get("ink");
  const canvas = values.get("canvas");
  if (/^[0-9a-f]{6}$/i.test(ink || "")) elements.foreground.value = `#${ink}`;
  if (/^[0-9a-f]{6}$/i.test(canvas || "")) elements.background.value = `#${canvas}`;
  assign(elements.brightness, "brightness");
  assign(elements.contrast, "contrast");
  elements.invert.checked = values.get("invert") === "1";
  const fontIndex = Number(values.get("font"));
  if (Number.isInteger(fontIndex) && elements.font.options[fontIndex]) elements.font.selectedIndex = fontIndex;
}

function syncControlLabels() {
  elements.columnsValue.textContent = elements.columns.value;
  elements.brightnessValue.textContent = Number(elements.brightness.value).toFixed(2);
  elements.contrastValue.textContent = Number(elements.contrast.value).toFixed(2);
  elements.customField.classList.toggle("hidden", elements.charset.value !== "custom");
}

function rerender() {
  syncControlLabels();
  if (state.sourceKind === "image") renderCurrent();
}

function bindEvents() {
  const openPicker = () => elements.fileInput.click();
  elements.dropZone.addEventListener("click", openPicker);
  elements.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") openPicker();
  });
  elements.emptyOpen.addEventListener("click", openPicker);
  elements.fileInput.addEventListener("change", () => openFile(elements.fileInput.files[0]));
  elements.webcam.addEventListener("click", () => openLive("webcam"));
  elements.screen.addEventListener("click", () => openLive("screen"));
  elements.stop.addEventListener("click", () => stopSource());
  elements.png.addEventListener("click", savePng);
  elements.svg.addEventListener("click", saveSvg);
  elements.text.addEventListener("click", saveText);
  elements.share.addEventListener("click", shareOutput);
  elements.publish.addEventListener("click", publishLink);
  elements.link.addEventListener("click", copyStyleLink);

  for (const control of [
    elements.charset, elements.customCharset, elements.font, elements.columns,
    elements.resolution, elements.colorMode, elements.foreground, elements.background,
    elements.brightness, elements.contrast, elements.invert,
  ]) {
    control.addEventListener("input", rerender);
    control.addEventListener("change", rerender);
  }

  let dragDepth = 0;
  window.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth += 1;
    elements.dropOverlay.classList.add("visible");
  });
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("dragleave", (event) => {
    event.preventDefault();
    dragDepth -= 1;
    if (dragDepth <= 0) elements.dropOverlay.classList.remove("visible");
  });
  window.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    elements.dropOverlay.classList.remove("visible");
    openFile(event.dataTransfer.files[0]);
  });

  window.addEventListener("beforeunload", () => stopSource({ reset: false }));
}

restoreSettings();
syncControlLabels();
initializeRenderer();
bindEvents();
loadStudioConfig();
setStatus("Ready. Nothing is uploaded.");
