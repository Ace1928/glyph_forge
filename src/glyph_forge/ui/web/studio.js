import {
  BRAILLE_GLYPHS,
  CanvasGlyphRenderer,
  EDGE_GLYPHS,
  QUADRANT_GLYPHS,
  WebGLGlyphRenderer,
  clamp,
  sampleGlyphFrame,
} from "./studio-renderers.js";

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
  canvasShell: $("canvasShell"),
  install: $("installButton"),
  webcam: $("webcamButton"),
  screen: $("screenButton"),
  stop: $("stopButton"),
  textSource: $("textSourceInput"),
  useText: $("textSourceButton"),
  glyphCode: $("glyphCodeInput"),
  glyphCodeButton: $("glyphCodeButton"),
  audio: $("audioToggle"),
  mode: $("modeSelect"),
  charsetField: $("charsetField"),
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
  record: $("recordButton"),
  fullscreen: $("fullscreenButton"),
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
  lastMetricPaint: 0,
  lastRenderAt: 0,
  renderer: null,
  rendererEventsBound: false,
  options: null,
  optionsDirty: true,
  installPrompt: null,
  dimensions: { width: 1280, height: 720, rows: 72 },
  recording: null,
  audioBridge: null,
  gifFrames: null,
  gifDurations: null,
  gifIndex: 0,
  gifElapsedBase: undefined,
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

function densityCharset() {
  const selected = elements.charset.value;
  const raw = selected === "custom" ? elements.customCharset.value : CHARSETS[selected];
  const glyphs = Array.from(raw || " .#").slice(0, 128);
  return glyphs.length ? glyphs.join("") : " .#";
}

function activeCharset(mode = elements.mode.value) {
  if (mode === "braille") return BRAILLE_GLYPHS;
  if (mode === "quadrant") return QUADRANT_GLYPHS;
  if (mode === "half-block") return "▀";
  const density = densityCharset();
  return mode === "edge" ? `${density}${EDGE_GLYPHS}` : density;
}

function isDynamicSource() {
  return ["video", "webcam", "screen"].includes(state.sourceKind) || Boolean(state.gifFrames);
}

function performanceTier() {
  const memory = Number(navigator.deviceMemory) || 4;
  const cores = Number(navigator.hardwareConcurrency) || 4;
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  if (memory >= 12 || cores >= 12) return "workstation";
  if (memory <= 3 || cores <= 2 || (coarse && cores <= 4)) return "modest";
  return "balanced";
}

function sourceSize() {
  if (!state.source) return { width: 1280, height: 720 };
  if (state.sourceKind === "image") {
    return { width: state.source.naturalWidth, height: state.source.naturalHeight };
  }
  if (state.sourceKind === "text") {
    return { width: state.source.width, height: state.source.height };
  }
  return { width: state.source.videoWidth || 1280, height: state.source.videoHeight || 720 };
}

function adaptiveHeight() {
  return { workstation: 1080, balanced: 720, modest: 540 }[performanceTier()];
}

function adaptiveColumns() {
  return { workstation: 240, balanced: 160, modest: 96 }[performanceTier()];
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
  const rows = Math.max(1, Math.round((columns * 0.5) / aspect));
  state.dimensions = { width, height, rows };
  elements.grid.textContent = `${columns}×${rows}`;
  return state.dimensions;
}

function currentOptions() {
  if (!state.optionsDirty && state.options) return state.options;
  const dimensions = targetDimensions();
  const source = sourceSize();
  const baseCharset = densityCharset();
  state.options = {
    ...dimensions,
    sourceWidth: source.width,
    sourceHeight: source.height,
    columns: Number(elements.columns.value),
    mode: elements.mode.value,
    charset: activeCharset(),
    baseCharset,
    baseGlyphCount: Array.from(baseCharset).length,
    font: elements.font.value,
    sourceColor: elements.colorMode.value === "source",
    foreground: hexToRgb(elements.foreground.value),
    background: hexToRgb(elements.background.value),
    foregroundCss: elements.foreground.value,
    backgroundCss: elements.background.value,
    brightness: Number(elements.brightness.value),
    contrast: Number(elements.contrast.value),
    invert: elements.invert.checked,
  };
  state.optionsDirty = false;
  return state.options;
}

function invalidateOptions() {
  state.options = null;
  state.optionsDirty = true;
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
  if (!state.rendererEventsBound) {
    elements.canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      state.renderer = null;
      elements.engine.textContent = "GPU recovering";
      setStatus("The GPU context paused. Glyph Forge will resume automatically.");
    });
    elements.canvas.addEventListener("webglcontextrestored", () => {
      initializeRenderer();
      if (state.source) renderCurrent();
      setStatus("GPU rendering restored.");
    });
    state.rendererEventsBound = true;
  }
}

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.style.color = isError ? "var(--danger)" : "";
}

function recordingSupported() {
  return Boolean(window.MediaRecorder && window.MediaStream && elements.canvas.captureStream);
}

function fullscreenSupported() {
  return Boolean(elements.canvasShell.requestFullscreen || elements.canvasShell.webkitRequestFullscreen);
}

function enableExports(enabled) {
  for (const button of [elements.png, elements.svg, elements.text, elements.share]) {
    button.disabled = !enabled;
  }
  elements.publish.disabled = !enabled || !state.shareConfig.enabled;
  elements.stop.disabled = !enabled;
  elements.fullscreen.disabled = !enabled || !fullscreenSupported();
  elements.record.disabled = !enabled || !isDynamicSource() || !recordingSupported();
}

function syncAudioControl() {
  const lockedLiveSource = ["webcam", "screen"].includes(state.sourceKind);
  elements.audio.disabled = Boolean(state.recording) || lockedLiveSource;
  elements.audio.title = lockedLiveSource ? "Stop this live source to change its audio permission" : "";
}

function updateMetrics(timestamp) {
  state.frameTimes.push(timestamp);
  const cutoff = timestamp - 1000;
  while (state.frameTimes.length && state.frameTimes[0] < cutoff) state.frameTimes.shift();
  if (timestamp - state.lastMetricPaint >= 250) {
    elements.fps.textContent = String(Math.max(0, state.frameTimes.length - 1));
    state.lastMetricPaint = timestamp;
  }
}

function gifAdvance(timestamp) {
  if (!state.gifDurations || !state.gifFrames || state.gifDurations.length < 2) return;
  if (state.gifElapsedBase === undefined) state.gifElapsedBase = timestamp;
  let elapsed = timestamp - state.gifElapsedBase;
  let index = state.gifIndex || 0;
  while (elapsed > 0) {
    const duration = state.gifDurations[index];
    if (elapsed < duration) break;
    elapsed -= duration;
    index = (index + 1) % state.gifDurations.length;
  }
  if (index !== state.gifIndex) {
    state.gifIndex = index;
    state.source.src = state.gifFrames[index];
  }
}

function renderCurrent(timestamp = performance.now()) {
  if (!state.source || !state.renderer) return;
  try {
    gifAdvance(timestamp);
    state.renderer.draw(state.source, currentOptions());
    state.lastRenderAt = timestamp;
    updateMetrics(timestamp);
  } catch (error) {
    console.error(error);
    setStatus(`Render failed: ${error.message}`, true);
  }
}

function shouldRender(timestamp) {
  if (document.hidden && !state.recording) return false;
  const fallbackBudget = elements.engine.textContent.startsWith("Canvas")
    ? (performanceTier() === "modest" ? 50 : 1000 / 30)
    : 0;
  return !fallbackBudget || timestamp - state.lastRenderAt >= fallbackBudget;
}

function beginRenderLoop() {
  state.generation += 1;
  const generation = state.generation;
  state.frameTimes = [];
  state.lastMetricPaint = 0;
  state.lastRenderAt = 0;
  const frame = (timestamp) => {
    if (generation !== state.generation || !state.source) return;
    if (shouldRender(timestamp)) renderCurrent(timestamp);
    if (isDynamicSource()) schedule();
  };
  const schedule = () => {
    if (typeof elements.video.requestVideoFrameCallback === "function") {
      elements.video.requestVideoFrameCallback((timestamp) => frame(timestamp));
    } else {
      requestAnimationFrame(frame);
    }
  };
  requestAnimationFrame(frame);
}

function describeSource(name) {
  invalidateOptions();
  const size = sourceSize();
  elements.sourceName.textContent = name;
  elements.sourceMeta.textContent = `${size.width}×${size.height} · ${state.sourceKind}`;
  elements.emptyState.classList.add("hidden");
  elements.record.textContent = state.sourceKind === "video"
    ? "Render full video"
    : (isDynamicSource() ? "Record live video" : "Record video");
  enableExports(true);
  syncAudioControl();
  setStatus("Rendering locally. No media has been uploaded.");
}

async function stopSource({ reset = true, saveRecording = true } = {}) {
  state.generation += 1;
  await stopRecording({ save: saveRecording, resumePreview: false });
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
  state.lastMetricPaint = 0;
  state.lastRenderAt = 0;
  invalidateOptions();
  state.gifFrames = null;
  state.gifDurations = null;
  state.gifIndex = 0;
  state.gifElapsedBase = undefined;
  elements.record.textContent = "Record video";
  syncAudioControl();
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

function seekVideo(video, time) {
  if (Math.abs(video.currentTime - time) < 0.01) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const complete = () => {
      cleanup();
      resolve();
    };
    const failed = () => {
      cleanup();
      reject(new Error("The browser could not seek this video"));
    };
    const cleanup = () => {
      video.removeEventListener("seeked", complete);
      video.removeEventListener("error", failed);
    };
    video.addEventListener("seeked", complete, { once: true });
    video.addEventListener("error", failed, { once: true });
    video.currentTime = time;
  });
}

async function openFile(file) {
  if (!file) return;
  await stopSource({ reset: false });
  try {
    state.objectUrl = URL.createObjectURL(file);
    const mediaType = file.type.toLowerCase();
    const filename = file.name.toLowerCase();
    const videoFile = mediaType.startsWith("video/")
      || /\.(mp4|m4v|mov|webm|ogv|avi|mkv)$/u.test(filename);
    const imageFile = mediaType.startsWith("image/")
      || /\.(avif|bmp|gif|heic|heif|jpe?g|png|svg|webp)$/u.test(filename);
    if (videoFile) {
      state.sourceKind = "video";
      elements.video.src = state.objectUrl;
      elements.video.loop = true;
      await waitForVideo(elements.video);
      await elements.video.play();
      state.source = elements.video;
    } else if (imageFile) {
      state.sourceKind = "image";
      const image = new Image();
      image.src = state.objectUrl;
      if (typeof image.decode === "function") {
        await image.decode();
      } else {
        await new Promise((resolve, reject) => {
          image.addEventListener("load", resolve, { once: true });
          image.addEventListener("error", () => reject(new Error("The browser could not decode this image")), { once: true });
        });
      }
      state.source = image;
    } else {
      throw new Error("Choose an image or video file");
    }
    state.sourceName = file.name;
    describeSource(file.name);
    beginRenderLoop();
  } catch (error) {
    await stopSource();
    setStatus(error.message, true);
  }
}

async function openLive(kind) {
  await stopSource({ reset: false });
  try {
    let stream;
    if (kind === "screen") {
      if (!navigator.mediaDevices?.getDisplayMedia) throw new Error("Screen capture is unavailable in this browser");
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: { ideal: 30, max: 60 } },
        audio: elements.audio.checked,
      });
    } else {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Webcam capture is unavailable in this browser");
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30, max: 60 } },
        audio: elements.audio.checked,
      });
    }
    state.sourceKind = kind;
    elements.video.srcObject = stream;
    await waitForVideo(elements.video);
    await elements.video.play();
    state.source = elements.video;
    state.sourceName = kind === "screen" ? "Shared screen" : "Webcam";
    const [track] = stream.getVideoTracks();
    if (track) track.addEventListener("ended", () => void stopSource(), { once: true });
    describeSource(state.sourceName);
    beginRenderLoop();
  } catch (error) {
    await stopSource();
    setStatus(`${kind === "screen" ? "Screen" : "Webcam"} access failed: ${error.message}`, true);
  }
}

function wrappedText(context, value, maximumWidth) {
  const words = value.trim().split(/\s+/u);
  const lines = [];
  let line = "";
  for (const word of words) {
    if (context.measureText(word).width > maximumWidth) {
      if (line) lines.push(line);
      let fragment = "";
      for (const character of Array.from(word)) {
        if (fragment && context.measureText(`${fragment}${character}`).width > maximumWidth) {
          lines.push(fragment);
          fragment = character;
        } else {
          fragment += character;
        }
      }
      line = fragment;
      continue;
    }
    const candidate = line ? `${line} ${word}` : word;
    if (line && context.measureText(candidate).width > maximumWidth) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
  }
  if (line) lines.push(line);
  return lines;
}

async function openTextSource() {
  const value = elements.textSource.value.trim();
  if (!value) {
    elements.textSource.focus();
    setStatus("Type something to forge first.", true);
    return;
  }
  await stopSource({ reset: false });
  const canvas = document.createElement("canvas");
  canvas.width = 1600;
  canvas.height = 900;
  const context = canvas.getContext("2d", { alpha: false });
  context.fillStyle = elements.background.value;
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = elements.foreground.value;
  context.textAlign = "center";
  context.textBaseline = "middle";
  let fontSize = 280;
  let lines = [];
  for (; fontSize > 42; fontSize -= 6) {
    context.font = `800 ${fontSize}px ${elements.font.value}`;
    lines = wrappedText(context, value, canvas.width * 0.84);
    const tooWide = lines.some((line) => context.measureText(line).width > canvas.width * 0.84);
    const tooTall = lines.length * fontSize * 1.12 > canvas.height * 0.76;
    if (!tooWide && !tooTall && lines.length <= 5) break;
  }
  const lineHeight = fontSize * 1.12;
  const start = canvas.height / 2 - ((lines.length - 1) * lineHeight) / 2;
  context.font = `800 ${fontSize}px ${elements.font.value}`;
  lines.forEach((line, index) => context.fillText(line, canvas.width / 2, start + index * lineHeight));
  state.source = canvas;
  state.sourceKind = "text";
  state.sourceName = value.length > 36 ? `${value.slice(0, 33)}…` : value;
  describeSource(`Text · ${state.sourceName}`);
  beginRenderLoop();
}

async function openGlyphCode(raw) {
  const value = String(raw || "").trim() || elements.glyphCode.value.trim();
  if (!value) {
    setStatus("Paste a glyph code first.", true);
    return;
  }
  if (!value.startsWith("glyph:v1:")) {
    setStatus("That does not look like a glyph:v1:… code.", true);
    return;
  }
  const body = value.slice("glyph:v1:".length);
  const kind = body.slice(0, body.indexOf(":"));
  const payload = body.slice(kind.length + 1);
  try {
    if (kind === "img") {
      const blob = await (await fetch(`data:image/png;base64,${payload}`)).blob();
      await openFile(new File([blob], "glyph-code.png", { type: "image/png" }));
      setStatus("Image regenerated from the glyph code.");
    } else if (kind === "banner") {
      const spec = JSON.parse(atob(payload));
      elements.textSource.value = String(spec.text || "GLYPH");
      await openTextSource();
      setStatus(`Banner text restored. Font “${spec.font || "small"}” and style “${spec.style || "minimal"}” apply exactly in the terminal.`);
    } else if (kind === "gif") {
      const tilde = payload.indexOf("~");
      if (tilde < 0) throw new Error("GIF code is missing its frames");
      const info = JSON.parse(atob(payload.slice(0, tilde)));
      const frames = payload.slice(tilde + 1).split("~");
      if (!frames.length) throw new Error("GIF code has no frames");
      const durations = Array.isArray(info.durations) && info.durations.length === frames.length
        ? info.durations.map((value) => Math.min(10000, Math.max(10, Number(value) || 80)))
        : frames.map(() => 80);
      await stopSource({ reset: false });
      const image = new Image();
      image.src = `data:image/png;base64,${frames[0]}`;
      await image.decode();
      state.source = image;
      state.sourceKind = "image";
      state.gifFrames = frames.map((frame) => `data:image/png;base64,${frame}`);
      state.gifDurations = durations;
      state.gifIndex = 0;
      state.gifElapsedBase = undefined;
      describeSource(`GIF · ${frames.length} frames from glyph code`);
      beginRenderLoop();
      setStatus(`Animated GIF regenerated locally — ${frames.length} frames at ${(1000 / (durations.reduce((a, b) => a + b, 0) / durations.length)).toFixed(1)} fps.`);
    } else {
      throw new Error(`unknown kind “${kind}” (try img, banner, or gif)`);
    }
  } catch (error) {
    await stopSource();
    setStatus(`Could not regenerate glyph code: ${error.message}`, true);
  }
}

function sampleTextFrame() {
  const options = currentOptions();
  if (!state.source) return { lines: [], colors: [], options };
  const context = elements.sampler.getContext("2d", { willReadFrequently: true });
  return sampleGlyphFrame(state.source, options, elements.sampler, context);
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.hidden = true;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function outputName(extension) {
  const base = (state.sourceName || "glyph-forge").replace(/\.[^.]+$/, "").replace(/[^a-z0-9_-]+/gi, "-");
  return `${base || "glyph-forge"}.glyph.${extension}`;
}

function humanBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

function elapsedLabel(milliseconds) {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  return `${minutes}:${String(totalSeconds % 60).padStart(2, "0")}`;
}

function recordingFormat() {
  const candidates = [
    ["video/webm;codecs=vp9,opus", "webm"],
    ["video/webm;codecs=vp8,opus", "webm"],
    ["video/webm", "webm"],
    ["video/mp4;codecs=avc1,mp4a.40.2", "mp4"],
    ["video/mp4", "mp4"],
  ];
  for (const [mimeType, extension] of candidates) {
    if (!MediaRecorder.isTypeSupported || MediaRecorder.isTypeSupported(mimeType)) {
      return { mimeType, extension };
    }
  }
  return { mimeType: "", extension: "webm" };
}

function recordingFrameRate() {
  const sourceTrack = elements.video.srcObject?.getVideoTracks()[0];
  const sourceRate = Number(sourceTrack?.getSettings?.().frameRate) || 30;
  return clamp(Math.round(sourceRate), 24, 60);
}

async function sourceAudioCapture() {
  if (!elements.audio.checked) return { tracks: [], usesBridge: false };
  if (elements.video.srcObject) {
    return { tracks: elements.video.srcObject.getAudioTracks(), usesBridge: false };
  }
  let capture = null;
  try {
    capture = elements.video.captureStream?.() || elements.video.mozCaptureStream?.();
  } catch (error) {
    console.warn("Media-element capture is unavailable; trying Web Audio", error);
  }
  const tracks = capture?.getAudioTracks() || [];
  if (tracks.length) return { tracks, usesBridge: false };

  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext || state.sourceKind !== "video") return { tracks: [], usesBridge: false };
  if (!state.audioBridge) {
    const context = new AudioContext();
    const source = context.createMediaElementSource(elements.video);
    const destination = context.createMediaStreamDestination();
    source.connect(destination);
    state.audioBridge = { context, source, destination };
  }
  elements.video.muted = false;
  await state.audioBridge.context.resume();
  return { tracks: state.audioBridge.destination.stream.getAudioTracks(), usesBridge: true };
}

function resetRecordingUi() {
  elements.record.classList.remove("recording");
  elements.record.textContent = state.sourceKind === "video"
    ? "Render full video"
    : (isDynamicSource() ? "Record live video" : "Record video");
  elements.record.setAttribute("aria-pressed", "false");
  syncAudioControl();
  enableExports(Boolean(state.source));
}

function finishRecording(session) {
  if (session.finished) return;
  session.finished = true;
  clearInterval(session.timer);
  for (const track of session.canvasStream.getTracks()) track.stop();
  if (session.endedHandler) elements.video.removeEventListener("ended", session.endedHandler);
  if (session.usesAudioBridge) {
    elements.video.muted = true;
    void state.audioBridge?.context.suspend();
  }
  if (state.recording === session) state.recording = null;
  resetRecordingUi();
  const duration = performance.now() - session.startedAt;
  if (session.save && session.chunks.length) {
    const mimeType = session.recorder.mimeType || session.format.mimeType || "video/webm";
    const blob = new Blob(session.chunks, { type: mimeType });
    download(blob, outputName(session.format.extension));
    const audio = session.hasAudio ? "source audio included" : "video only";
    setStatus(`Recording saved · ${elapsedLabel(duration)} · ${humanBytes(blob.size)} · ${audio}.`);
  } else if (session.save) {
    setStatus("The browser stopped recording before it produced any media.", true);
  }
  if (session.fileVideo && session.resumePreview && state.source === elements.video) {
    elements.video.loop = true;
    if (elements.video.ended) elements.video.currentTime = 0;
    void elements.video.play();
    beginRenderLoop();
  }
  session.resolve();
}

async function startRecording() {
  if (!state.source || !isDynamicSource() || !recordingSupported()) {
    setStatus("Video recording is unavailable for this source or browser.", true);
    return;
  }
  let canvasStream = null;
  let session = null;
  let usesAudioBridge = false;
  const fileVideo = state.sourceKind === "video";
  try {
    if (fileVideo) {
      elements.video.loop = false;
      elements.video.pause();
      await seekVideo(elements.video, 0);
    }
    renderCurrent();
    const frameRate = recordingFrameRate();
    canvasStream = elements.canvas.captureStream(frameRate);
    const audioCapture = await sourceAudioCapture();
    usesAudioBridge = audioCapture.usesBridge;
    const audioTracks = audioCapture.tracks;
    const stream = new MediaStream([...canvasStream.getVideoTracks(), ...audioTracks]);
    const format = recordingFormat();
    const pixelsPerSecond = elements.canvas.width * elements.canvas.height * frameRate;
    const videoBitsPerSecond = clamp(Math.round(pixelsPerSecond * 0.12), 4_000_000, 30_000_000);
    let recorder;
    try {
      recorder = new MediaRecorder(stream, {
        ...(format.mimeType ? { mimeType: format.mimeType } : {}),
        videoBitsPerSecond,
        audioBitsPerSecond: 192_000,
      });
    } catch (error) {
      console.warn("Preferred recording settings unavailable; using browser defaults", error);
      recorder = new MediaRecorder(stream);
      format.mimeType = recorder.mimeType;
      format.extension = recorder.mimeType.includes("mp4") ? "mp4" : "webm";
    }
    let resolve;
    const done = new Promise((complete) => { resolve = complete; });
    session = {
      recorder,
      canvasStream,
      chunks: [],
      format,
      hasAudio: audioTracks.length > 0,
      usesAudioBridge: audioCapture.usesBridge,
      startedAt: performance.now(),
      save: true,
      resumePreview: true,
      finished: false,
      fileVideo,
      endedHandler: null,
      timer: 0,
      done,
      resolve,
    };
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) session.chunks.push(event.data);
    });
    recorder.addEventListener("stop", () => finishRecording(session), { once: true });
    recorder.addEventListener("error", (event) => {
      console.error("MediaRecorder failed", event.error);
      setStatus(`Recording failed: ${event.error?.message || "browser encoder error"}`, true);
    });
    state.recording = session;
    elements.record.classList.add("recording");
    elements.record.setAttribute("aria-pressed", "true");
    syncAudioControl();
    session.timer = window.setInterval(() => {
      elements.record.textContent = `Stop recording · ${elapsedLabel(performance.now() - session.startedAt)}`;
    }, 250);
    recorder.start(1000);
    if (fileVideo) {
      session.endedHandler = () => void stopRecording();
      elements.video.addEventListener("ended", session.endedHandler, { once: true });
      await elements.video.play();
    }
    const audioMessage = session.hasAudio
      ? "Rendered video and source audio are recording on one synchronized timeline."
      : "Recording rendered video without audio.";
    setStatus(audioMessage);
  } catch (error) {
    if (session) {
      session.save = false;
      session.resumePreview = false;
      if (session.recorder.state === "inactive") finishRecording(session);
      else session.recorder.stop();
      await session.done;
    } else if (canvasStream) {
      for (const track of canvasStream.getTracks()) track.stop();
    }
    if (usesAudioBridge) {
      elements.video.muted = true;
      void state.audioBridge?.context.suspend();
    }
    if (fileVideo) {
      elements.video.loop = true;
      void elements.video.play();
      beginRenderLoop();
    }
    setStatus(`Could not start recording: ${error.message}`, true);
    resetRecordingUi();
  }
}

async function stopRecording({ save = true, resumePreview = true } = {}) {
  const session = state.recording;
  if (!session) return;
  session.save = save;
  session.resumePreview = resumePreview;
  if (session.recorder.state === "inactive") {
    finishRecording(session);
  } else {
    session.recorder.stop();
  }
  await session.done;
}

async function toggleRecording() {
  if (state.recording) await stopRecording();
  else await startRecording();
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
    mode: elements.mode.value,
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

function studioEndpoint(path) {
  return new URL(path.replace(/^\//u, ""), document.baseURI);
}

async function loadStudioConfig() {
  try {
    const response = await fetch(studioEndpoint("api/config"), {
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
    const endpoint = studioEndpoint(`api/share?name=${encodeURIComponent(filename)}`);
    const response = await fetch(endpoint, {
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

function fullscreenElement() {
  return document.fullscreenElement || document.webkitFullscreenElement;
}

function syncFullscreenUi() {
  const active = Boolean(fullscreenElement());
  elements.fullscreen.textContent = active ? "Exit fullscreen" : "Fullscreen output";
  elements.fullscreen.setAttribute("aria-pressed", active ? "true" : "false");
}

async function toggleFullscreen() {
  try {
    if (fullscreenElement()) {
      const exit = document.exitFullscreen || document.webkitExitFullscreen;
      await exit.call(document);
    } else {
      const request = elements.canvasShell.requestFullscreen || elements.canvasShell.webkitRequestFullscreen;
      await request.call(elements.canvasShell);
    }
  } catch (error) {
    setStatus(`Fullscreen failed: ${error.message}`, true);
  }
}

function restoreSettings() {
  if (!window.location.hash) {
    elements.columns.value = String(adaptiveColumns());
    return;
  }
  const values = new URLSearchParams(window.location.hash.slice(1));
  const assign = (element, key, validate = () => true) => {
    const value = values.get(key);
    if (value !== null && validate(value)) element.value = value;
  };
  assign(elements.mode, "mode", (value) => [...elements.mode.options].some((option) => option.value === value));
  assign(elements.charset, "style", (value) => [...elements.charset.options].some((option) => option.value === value));
  assign(elements.customCharset, "glyphs");
  assign(elements.columns, "columns", (value) => Number(value) >= 32 && Number(value) <= 480);
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
  const densityMode = ["glyph", "edge"].includes(elements.mode.value);
  elements.charsetField.classList.toggle("hidden", !densityMode);
  elements.customField.classList.toggle("hidden", !densityMode || elements.charset.value !== "custom");
}

function rerender() {
  invalidateOptions();
  syncControlLabels();
  if (state.source && !isDynamicSource()) renderCurrent();
}

function standaloneMode() {
  return window.matchMedia("(display-mode: standalone)").matches
    || window.navigator.standalone === true;
}

function syncInstallButton() {
  const eligible = window.location.protocol === "https:" && !standaloneMode();
  elements.install.classList.toggle("hidden", !eligible);
}

async function installStudio() {
  if (state.installPrompt) {
    const prompt = state.installPrompt;
    state.installPrompt = null;
    await prompt.prompt();
    const choice = await prompt.userChoice;
    if (choice.outcome === "accepted") {
      elements.install.classList.add("hidden");
      setStatus("Glyph Forge is being installed.");
    } else {
      setStatus("Installation cancelled. The browser Studio remains ready here.");
    }
    return;
  }
  const appleMobile = /iPad|iPhone|iPod/u.test(navigator.userAgent)
    || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const safari = /Safari/u.test(navigator.userAgent) && !/Chrome|Chromium|CriOS|Edg/u.test(navigator.userAgent);
  if (appleMobile) {
    setStatus("To install on iPhone or iPad, open Share and choose Add to Home Screen.");
  } else if (safari) {
    setStatus("To install from Safari, choose Add to Dock from the File menu.");
  } else {
    setStatus("Choose Install app or Add to Home Screen from your browser menu.");
  }
}

function syncCapabilities() {
  const media = navigator.mediaDevices;
  const camera = Boolean(window.isSecureContext && media?.getUserMedia);
  const screen = Boolean(window.isSecureContext && media?.getDisplayMedia);
  elements.webcam.disabled = !camera;
  elements.webcam.title = camera ? "" : "Webcam capture needs a secure, supported browser";
  elements.screen.disabled = !screen;
  elements.screen.title = screen ? "" : "Screen capture is unavailable on this browser or device";
  if (!recordingSupported()) {
    elements.record.title = "This browser cannot record a canvas stream; still-image exports remain available";
  }
}

function bindInstallability() {
  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    state.installPrompt = event;
    syncInstallButton();
  });
  window.addEventListener("appinstalled", () => {
    state.installPrompt = null;
    elements.install.classList.add("hidden");
    setStatus("Glyph Forge installed. It can now launch like an app.");
  });
  window.matchMedia("(display-mode: standalone)").addEventListener?.("change", syncInstallButton);
  elements.install.addEventListener("click", () => void installStudio());
  syncInstallButton();
}

function bindLaunchQueue() {
  if (!window.launchQueue?.setConsumer) return;
  window.launchQueue.setConsumer(async (launchParams) => {
    const [handle] = launchParams.files || [];
    if (!handle) return;
    try {
      await openFile(await handle.getFile());
    } catch (error) {
      setStatus(`Could not open the shared file: ${error.message}`, true);
    }
  });
}

async function registerServiceWorker() {
  const installableOrigin = window.location.protocol === "https:"
    || window.location.hostname === "localhost";
  if (!("serviceWorker" in navigator) || !installableOrigin) return;
  try {
    await navigator.serviceWorker.register("./service-worker.js", {
      scope: "./",
      updateViaCache: "none",
    });
  } catch (error) {
    console.warn("Offline app installation is unavailable", error);
  }
}

function bindEvents() {
  const openPicker = () => elements.fileInput.click();
  elements.dropZone.addEventListener("click", openPicker);
  elements.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") openPicker();
  });
  elements.emptyOpen.addEventListener("click", openPicker);
  elements.fileInput.addEventListener("change", () => {
    const [file] = elements.fileInput.files;
    elements.fileInput.value = "";
    void openFile(file);
  });
  elements.webcam.addEventListener("click", () => void openLive("webcam"));
  elements.screen.addEventListener("click", () => void openLive("screen"));
  elements.stop.addEventListener("click", () => void stopSource());
  elements.useText.addEventListener("click", () => void openTextSource());
  elements.textSource.addEventListener("keydown", (event) => {
    if (event.key === "Enter") void openTextSource();
  });
  elements.glyphCodeButton.addEventListener("click", () => void openGlyphCode());
  elements.glyphCode.addEventListener("keydown", (event) => {
    if (event.key === "Enter") void openGlyphCode();
  });
  elements.png.addEventListener("click", savePng);
  elements.svg.addEventListener("click", saveSvg);
  elements.text.addEventListener("click", saveText);
  elements.share.addEventListener("click", shareOutput);
  elements.publish.addEventListener("click", publishLink);
  elements.link.addEventListener("click", copyStyleLink);
  elements.record.addEventListener("click", () => void toggleRecording());
  elements.fullscreen.addEventListener("click", () => void toggleFullscreen());
  document.addEventListener("fullscreenchange", syncFullscreenUi);
  document.addEventListener("webkitfullscreenchange", syncFullscreenUi);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state.source && !isDynamicSource()) renderCurrent();
  });

  for (const control of [
    elements.mode, elements.charset, elements.customCharset, elements.font, elements.columns,
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
    void openFile(event.dataTransfer.files[0]);
  });

  window.addEventListener("pagehide", () => void stopSource({ reset: false, saveRecording: false }));
}

restoreSettings();
syncControlLabels();
initializeRenderer();
syncCapabilities();
bindEvents();
bindInstallability();
bindLaunchQueue();
void registerServiceWorker();
void loadStudioConfig();
setStatus("Ready. Nothing is uploaded.");
