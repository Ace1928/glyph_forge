const PROJECT_SCHEMA = "glyph-forge-project";
const PRESET_SCHEMA = "glyph-forge-preset";
const SCHEMA_VERSION = 1;
const RENDER_CONTRACT_VERSION = 1;
const MAX_DOCUMENT_BYTES = 4 * 1024 * 1024;
const MAX_VARIANTS = 256;
const MAX_HISTORY = 100;

const REQUEST_KEYS = Object.freeze([
  "alignment", "background", "brightness", "cell_aspect", "charset",
  "contract_version", "contrast", "dither", "edge_algorithm",
  "edge_threshold", "fit", "font", "foreground", "height", "invert",
  "max_height", "max_width", "mode", "optimize", "output_format",
  "output_height", "output_width", "resample", "style", "threshold", "width",
]);
const PROJECT_KEYS = Object.freeze([
  "active_variant", "created_at", "metadata", "name", "schema",
  "schema_version", "source", "updated_at", "variants",
]);
const PRESET_KEYS = Object.freeze([
  "metadata", "name", "request", "schema", "schema_version",
]);
const VARIANT_KEYS = Object.freeze(["id", "name", "request"]);
const ASSET_KEYS = Object.freeze(["kind", "path"]);
const IDENTIFIER = /^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$/u;
const MODES = new Set(["glyph", "edge", "braille", "half-block", "quadrant"]);
const PLUGIN_MODE = /^plugin:[a-z0-9][a-z0-9._-]*\/[a-z0-9][a-z0-9._-]*$/u;
const FORMATS = new Set(["text", "ansi256", "truecolor", "html", "png", "svg"]);
const FITS = new Set(["contain", "cover", "stretch"]);
const ALIGNMENTS = new Set([
  "top-left", "top", "top-right", "left", "center", "right",
  "bottom-left", "bottom", "bottom-right",
]);
const EDGES = new Set(["sobel", "prewitt", "scharr", "laplacian", "canny"]);
const RESAMPLE = new Set(["nearest", "bilinear", "bicubic", "lanczos"]);
const WINDOWS_RESERVED = /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$/iu;

function fail(message) {
  throw new TypeError(message);
}

function plainObject(value, name) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(`${name} must be an object`);
  }
  return value;
}

function exactKeys(value, expected, name) {
  const keys = Object.keys(plainObject(value, name)).sort();
  const wanted = [...expected].sort();
  const missing = wanted.filter((key) => !keys.includes(key));
  const unknown = keys.filter((key) => !wanted.includes(key));
  if (missing.length) fail(`${name} is missing ${missing.join(", ")}`);
  if (unknown.length) fail(`${name} contains unknown fields: ${unknown.join(", ")}`);
}

function stringValue(value, name, maximum = 256) {
  if (typeof value !== "string" || !value.trim()) fail(`${name} cannot be empty`);
  const result = value.trim();
  if (result.length > maximum) fail(`${name} cannot exceed ${maximum} characters`);
  return result;
}

function integer(value, name, minimum, maximum, nullable = false) {
  if (nullable && value === null) return null;
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    fail(`${name} must be an integer between ${minimum} and ${maximum}`);
  }
  return value;
}

function finite(value, name, minimum, maximum) {
  if (typeof value !== "number" || !Number.isFinite(value)
      || value < minimum || value > maximum) {
    fail(`${name} must be between ${minimum} and ${maximum}`);
  }
  return value;
}

function booleanValue(value, name) {
  if (typeof value !== "boolean") fail(`${name} must be true or false`);
  return value;
}

function enumValue(value, choices, name) {
  if (typeof value !== "string" || !choices.has(value)) {
    fail(`unsupported ${name} ${JSON.stringify(value)}`);
  }
  return value;
}

function optionalString(value, name) {
  return value === null ? null : stringValue(value, name, 4096);
}

function timestamp(value, name) {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))
      || !/(?:Z|[+-]\d\d:\d\d)$/u.test(value)) {
    fail(`${name} must be an ISO 8601 timestamp with timezone`);
  }
  return value;
}

function jsonCompatible(value, depth = 0) {
  if (depth > 12) return false;
  if (value === null || ["string", "boolean"].includes(typeof value)) return true;
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) {
    return value.length <= 1024 && value.every((item) => jsonCompatible(item, depth + 1));
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    return entries.length <= 1024 && entries.every(([key, item]) => (
      key.length <= 256 && jsonCompatible(item, depth + 1)
    ));
  }
  return false;
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function validateMetadata(value) {
  plainObject(value, "metadata");
  if (!jsonCompatible(value)) fail("metadata must contain bounded finite JSON values");
  return clone(value);
}

function validateAsset(value) {
  exactKeys(value, ASSET_KEYS, "asset");
  const path = stringValue(value.path, "asset path", 4096).normalize("NFC");
  if (path.startsWith("/") || path.includes("\\") || /^[a-z]:/iu.test(path)) {
    fail("asset path must be portable and relative");
  }
  const parts = path.split("/");
  if (parts.some((part) => !part || part === "." || part === ".."
      || part !== part.replace(/[ .]+$/u, "") || WINDOWS_RESERVED.test(part)
      || /[<>:"|?*\u0000-\u001f]/u.test(part))) {
    fail("asset path contains a non-portable segment");
  }
  const kind = stringValue(value.kind, "asset kind", 32).toLowerCase();
  if (!IDENTIFIER.test(kind)) fail("asset kind must be a portable identifier");
  return { kind, path: parts.join("/") };
}

export function validateRenderRequest(value) {
  exactKeys(value, REQUEST_KEYS, "render request");
  if (value.contract_version !== RENDER_CONTRACT_VERSION) {
    fail(`unsupported render contract version ${value.contract_version}`);
  }
  const mode = stringValue(value.mode, "mode", 128).toLowerCase();
  if (!MODES.has(mode) && !PLUGIN_MODE.test(mode)) fail(`unsupported mode ${mode}`);
  const request = {
    width: integer(value.width, "width", 1, 4096),
    height: integer(value.height, "height", 1, 4096, true),
    mode,
    output_format: enumValue(value.output_format, FORMATS, "output format"),
    charset: stringValue(value.charset, "charset", 100000),
    invert: booleanValue(value.invert, "invert"),
    dither: booleanValue(value.dither, "dither"),
    threshold: integer(value.threshold, "threshold", 0, 255),
    edge_algorithm: enumValue(value.edge_algorithm, EDGES, "edge algorithm"),
    edge_threshold: integer(value.edge_threshold, "edge threshold", 0, 255),
    cell_aspect: finite(value.cell_aspect, "cell aspect", Number.MIN_VALUE, 1000),
    resample: enumValue(value.resample, RESAMPLE, "resample mode"),
    brightness: finite(value.brightness, "brightness", 0, 2),
    contrast: finite(value.contrast, "contrast", 0, 2),
    style: optionalString(value.style, "style"),
    optimize: booleanValue(value.optimize, "optimize"),
    max_width: integer(value.max_width, "max width", 1, 4096, true),
    max_height: integer(value.max_height, "max height", 1, 4096, true),
    output_width: integer(value.output_width, "output width", 1, 32768, true),
    output_height: integer(value.output_height, "output height", 1, 32768, true),
    fit: enumValue(value.fit, FITS, "fit mode"),
    alignment: enumValue(value.alignment, ALIGNMENTS, "alignment"),
    foreground: stringValue(value.foreground, "foreground", 128),
    background: stringValue(value.background, "background", 128),
    font: optionalString(value.font, "font"),
    contract_version: RENDER_CONTRACT_VERSION,
  };
  const graphical = request.output_format === "png" || request.output_format === "svg";
  if ((request.output_width !== null || request.output_height !== null) && !graphical) {
    fail("output dimensions require PNG or SVG");
  }
  if (request.output_format === "html" && request.mode !== "glyph") {
    fail("HTML output requires glyph mode");
  }
  if (request.style && ["ansi256", "truecolor", "html"].includes(request.output_format)) {
    fail("text styles require plain text, PNG, or SVG output");
  }
  return request;
}

function validateVariant(value) {
  exactKeys(value, VARIANT_KEYS, "variant");
  const identifier = stringValue(value.id, "variant identifier", 64).toLowerCase();
  if (!IDENTIFIER.test(identifier)) fail("variant identifier is not portable");
  return {
    id: identifier,
    name: stringValue(value.name, "variant name"),
    request: validateRenderRequest(value.request),
  };
}

export function validateProject(value) {
  exactKeys(value, PROJECT_KEYS, "project");
  if (value.schema !== PROJECT_SCHEMA || value.schema_version !== SCHEMA_VERSION) {
    fail("unsupported Glyph Forge project schema");
  }
  if (!Array.isArray(value.variants) || value.variants.length < 1
      || value.variants.length > MAX_VARIANTS) {
    fail(`projects require between 1 and ${MAX_VARIANTS} variants`);
  }
  const variants = value.variants.map(validateVariant);
  const identifiers = variants.map(({ id }) => id);
  if (new Set(identifiers).size !== identifiers.length) fail("variant IDs must be unique");
  const active = stringValue(value.active_variant, "active variant", 64).toLowerCase();
  if (!identifiers.includes(active)) fail("active variant does not exist");
  return {
    schema: PROJECT_SCHEMA,
    schema_version: SCHEMA_VERSION,
    name: stringValue(value.name, "project name"),
    source: validateAsset(value.source),
    variants,
    active_variant: active,
    created_at: timestamp(value.created_at, "created_at"),
    updated_at: timestamp(value.updated_at, "updated_at"),
    metadata: validateMetadata(value.metadata),
  };
}

export function validatePreset(value) {
  exactKeys(value, PRESET_KEYS, "preset");
  if (value.schema !== PRESET_SCHEMA || value.schema_version !== SCHEMA_VERSION) {
    fail("unsupported Glyph Forge preset schema");
  }
  return {
    schema: PRESET_SCHEMA,
    schema_version: SCHEMA_VERSION,
    name: stringValue(value.name, "preset name"),
    request: validateRenderRequest(value.request),
    metadata: validateMetadata(value.metadata),
  };
}

export function parseDocument(text, kind) {
  if (kind !== "project" && kind !== "preset") fail(`unsupported document kind ${kind}`);
  if (typeof text !== "string" || new TextEncoder().encode(text).length > MAX_DOCUMENT_BYTES) {
    fail("document exceeds the 4 MiB limit");
  }
  let value;
  try {
    value = JSON.parse(text);
  } catch (error) {
    fail(`document is not valid JSON: ${error.message}`);
  }
  return kind === "preset" ? validatePreset(value) : validateProject(value);
}

export function encodeDocument(value, kind) {
  if (kind !== "project" && kind !== "preset") fail(`unsupported document kind ${kind}`);
  const validated = kind === "preset" ? validatePreset(value) : validateProject(value);
  const text = `${JSON.stringify(validated, null, 2)}\n`;
  if (new TextEncoder().encode(text).length > MAX_DOCUMENT_BYTES) {
    fail("document exceeds the 4 MiB limit");
  }
  return text;
}

function portableFilename(filename) {
  let selected = String(filename || "source")
    .normalize("NFC")
    .replace(/[<>:"/\\|?*\u0000-\u001f]/gu, "-")
    .replace(/[ .]+$/u, "");
  if (!selected) selected = "source";
  if (WINDOWS_RESERVED.test(selected)) selected = `_${selected}`;
  return selected.slice(0, 240).replace(/[ .]+$/u, "") || "source";
}

export function createProjectDocument(
  name,
  filename,
  request,
  now = new Date().toISOString(),
  kind = "image",
) {
  return validateProject({
    schema: PROJECT_SCHEMA,
    schema_version: SCHEMA_VERSION,
    name,
    source: { kind, path: `assets/${portableFilename(filename)}` },
    variants: [{ id: "default", name: "Default", request }],
    active_variant: "default",
    created_at: now,
    updated_at: now,
    metadata: {},
  });
}

export function createPresetDocument(name, request, metadata = {}) {
  return validatePreset({
    schema: PRESET_SCHEMA,
    schema_version: SCHEMA_VERSION,
    name,
    request,
    metadata,
  });
}

function updated(project, changes) {
  return validateProject({
    ...clone(project),
    ...changes,
    updated_at: new Date().toISOString(),
  });
}

export class ProjectSessionModel {
  constructor(project, historyLimit = MAX_HISTORY) {
    if (!Number.isInteger(historyLimit) || historyLimit < 1 || historyLimit > 10000) {
      fail("history limit must be between 1 and 10000");
    }
    this.project = validateProject(project);
    this.historyLimit = historyLimit;
    this.undoStack = [];
    this.redoStack = [];
    this.dirty = false;
  }

  get active() {
    return this.project.variants.find(({ id }) => id === this.project.active_variant);
  }

  apply(project) {
    const next = validateProject(project);
    if (JSON.stringify(next) === JSON.stringify(this.project)) return this.project;
    this.undoStack.push(this.project);
    if (this.undoStack.length > this.historyLimit) this.undoStack.shift();
    this.redoStack = [];
    this.project = next;
    this.dirty = true;
    return this.project;
  }

  replaceActiveRequest(request) {
    const selected = validateRenderRequest(request);
    if (JSON.stringify(selected) === JSON.stringify(this.active.request)) return this.project;
    return this.apply(updated(this.project, {
      variants: this.project.variants.map((variant) => (
        variant.id === this.project.active_variant ? { ...variant, request: selected } : variant
      )),
    }));
  }

  addVariant(identifier, name, request = this.active.request) {
    const variant = validateVariant({ id: identifier, name, request });
    if (this.project.variants.some(({ id }) => id === variant.id)) {
      fail(`variant ${JSON.stringify(identifier)} already exists`);
    }
    if (this.project.variants.length >= MAX_VARIANTS) fail("project has too many variants");
    return this.apply(updated(this.project, {
      variants: [...this.project.variants, variant],
      active_variant: variant.id,
    }));
  }

  removeVariant(identifier) {
    if (this.project.variants.length === 1) fail("the last variant cannot be removed");
    const variants = this.project.variants.filter(({ id }) => id !== identifier);
    if (variants.length === this.project.variants.length) fail(`unknown variant ${identifier}`);
    return this.apply(updated(this.project, {
      variants,
      active_variant: this.project.active_variant === identifier
        ? variants[0].id
        : this.project.active_variant,
    }));
  }

  selectVariant(identifier) {
    if (!this.project.variants.some(({ id }) => id === identifier)) {
      fail(`unknown variant ${identifier}`);
    }
    if (identifier === this.project.active_variant) return this.project;
    return this.apply(updated(this.project, { active_variant: identifier }));
  }

  undo() {
    if (!this.undoStack.length) fail("nothing to undo");
    this.redoStack.push(this.project);
    this.project = this.undoStack.pop();
    this.dirty = true;
    return this.project;
  }

  redo() {
    if (!this.redoStack.length) fail("nothing to redo");
    this.undoStack.push(this.project);
    this.project = this.redoStack.pop();
    this.dirty = true;
    return this.project;
  }

  markSaved() {
    this.dirty = false;
  }
}

export const PROJECT_CONTRACT = Object.freeze({
  projectSchema: PROJECT_SCHEMA,
  presetSchema: PRESET_SCHEMA,
  schemaVersion: SCHEMA_VERSION,
  renderContractVersion: RENDER_CONTRACT_VERSION,
  maxDocumentBytes: MAX_DOCUMENT_BYTES,
  maxVariants: MAX_VARIANTS,
  maxHistory: MAX_HISTORY,
});
