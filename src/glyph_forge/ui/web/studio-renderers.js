"use strict";

export const BRAILLE_GLYPHS = Array.from(
  { length: 256 },
  (_, mask) => String.fromCodePoint(0x2800 + mask),
).join("");
export const QUADRANT_GLYPHS = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█";
export const EDGE_GLYPHS = "─│╱╲";

const MODE_IDS = Object.freeze({
  glyph: 0,
  braille: 1,
  quadrant: 2,
  "half-block": 3,
  edge: 4,
});
const MODE_SUBCELLS = Object.freeze({
  glyph: [1, 1],
  braille: [2, 4],
  quadrant: [2, 2],
  "half-block": [1, 2],
  edge: [1, 1],
});
const BRAILLE_BITS = Object.freeze([[1, 8], [2, 16], [4, 32], [64, 128]]);

export function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function rgbCss(color) {
  return `rgb(${Math.round(color[0])} ${Math.round(color[1])} ${Math.round(color[2])})`;
}

function mixedCss(options, density) {
  const color = options.background.map((channel, index) => (
    channel + (options.foreground[index] - channel) * density
  ) * 255);
  return rgbCss(color);
}

function adjustedPixel(pixels, offset, options) {
  const channel = (value) => clamp(
    ((value / 255 - 0.5) * options.contrast + 0.5) * options.brightness,
    0,
    1,
  );
  const red = channel(pixels[offset]);
  const green = channel(pixels[offset + 1]);
  const blue = channel(pixels[offset + 2]);
  let luma = red * 0.299 + green * 0.587 + blue * 0.114;
  if (options.invert) luma = 1 - luma;
  return [red * 255, green * 255, blue * 255, luma];
}

function averageColor(samples) {
  const total = samples.reduce((sum, sample) => [
    sum[0] + sample[0],
    sum[1] + sample[1],
    sum[2] + sample[2],
    sum[3] + sample[3],
  ], [0, 0, 0, 0]);
  return total.map((value) => value / samples.length);
}

export function sampleGlyphFrame(source, options, canvas, context) {
  const [subcolumns, subrows] = MODE_SUBCELLS[options.mode];
  const sampleWidth = options.columns * subcolumns;
  const sampleHeight = options.rows * subrows;
  canvas.width = sampleWidth;
  canvas.height = sampleHeight;
  context.drawImage(source, 0, 0, sampleWidth, sampleHeight);
  const pixels = context.getImageData(0, 0, sampleWidth, sampleHeight).data;
  const samples = Array.from({ length: sampleWidth * sampleHeight }, (_, index) => (
    adjustedPixel(pixels, index * 4, options)
  ));
  const at = (column, row) => samples[
    clamp(row, 0, sampleHeight - 1) * sampleWidth
      + clamp(column, 0, sampleWidth - 1)
  ];
  const densityGlyphs = Array.from(options.baseCharset);
  const glyphRows = [];
  const colors = [];
  const halfColors = [];

  for (let row = 0; row < options.rows; row += 1) {
    const glyphRow = [];
    const colorRow = [];
    const halfColorRow = [];
    for (let column = 0; column < options.columns; column += 1) {
      if (options.mode === "braille") {
        let mask = 0;
        const cellSamples = [];
        for (let y = 0; y < 4; y += 1) {
          for (let x = 0; x < 2; x += 1) {
            const sample = at(column * 2 + x, row * 4 + y);
            cellSamples.push(sample);
            if (sample[3] >= 0.5) mask += BRAILLE_BITS[y][x];
          }
        }
        glyphRow.push(String.fromCodePoint(0x2800 + mask));
        colorRow.push(averageColor(cellSamples));
      } else if (options.mode === "quadrant") {
        const cellSamples = [
          at(column * 2, row * 2),
          at(column * 2 + 1, row * 2),
          at(column * 2, row * 2 + 1),
          at(column * 2 + 1, row * 2 + 1),
        ];
        const mask = cellSamples.reduce((value, sample, index) => (
          sample[3] >= 0.5 ? value + (1 << index) : value
        ), 0);
        glyphRow.push(Array.from(QUADRANT_GLYPHS)[mask]);
        colorRow.push(averageColor(cellSamples));
      } else if (options.mode === "half-block") {
        const upper = at(column, row * 2);
        const lower = at(column, row * 2 + 1);
        glyphRow.push("▀");
        colorRow.push(upper);
        halfColorRow.push([upper, lower]);
      } else {
        const sample = at(column, row);
        const glyphIndex = Math.min(
          densityGlyphs.length - 1,
          Math.floor(sample[3] * densityGlyphs.length),
        );
        let glyph = densityGlyphs[glyphIndex];
        if (options.mode === "edge") {
          const gx = at(column + 1, row)[3] - at(column - 1, row)[3];
          const gy = at(column, row - 1)[3] - at(column, row + 1)[3];
          const horizontal = Math.abs(gx);
          const vertical = Math.abs(gy);
          if (Math.hypot(gx, gy) >= 0.12) {
            if (horizontal > vertical * 2) glyph = "│";
            else if (vertical > horizontal * 2) glyph = "─";
            else glyph = gx * gy >= 0 ? "╲" : "╱";
          }
        }
        glyphRow.push(glyph);
        colorRow.push(sample);
      }
    }
    glyphRows.push(glyphRow);
    colors.push(colorRow);
    halfColors.push(halfColorRow);
  }
  return {
    lines: glyphRows.map((row) => row.join("")),
    glyphRows,
    colors,
    halfColors,
    options,
  };
}

export class WebGLGlyphRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });
    if (!this.gl) throw new Error("WebGL2 is unavailable");
    this.program = this.createProgram();
    this.uniforms = {};
    for (const name of [
      "u_source", "u_atlas", "u_grid", "u_atlas_grid", "u_base_glyph_count",
      "u_mode", "u_background", "u_foreground", "u_source_color",
      "u_brightness", "u_contrast", "u_invert",
    ]) {
      this.uniforms[name] = this.gl.getUniformLocation(this.program, name);
    }
    this.createGeometry();
    this.sourceTexture = this.createTexture();
    this.atlasTexture = this.createTexture();
    this.atlasKey = "";
    this.atlasGrid = [1, 1];
    this.sourceTextureSize = [0, 0];
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
      uniform vec2 u_atlas_grid;
      uniform float u_base_glyph_count;
      uniform int u_mode;
      uniform vec3 u_background;
      uniform vec3 u_foreground;
      uniform bool u_source_color;
      uniform float u_brightness;
      uniform float u_contrast;
      uniform bool u_invert;

      vec3 adjusted_color(vec2 uv) {
        vec3 color = texture(u_source, clamp(uv, vec2(0.0), vec2(1.0))).rgb;
        color = (color - 0.5) * u_contrast + 0.5;
        return clamp(color * u_brightness, 0.0, 1.0);
      }

      float density(vec2 uv) {
        float value = dot(adjusted_color(uv), vec3(0.299, 0.587, 0.114));
        return u_invert ? 1.0 - value : value;
      }

      void main() {
        vec2 cell = floor(v_uv * u_grid);
        vec2 sample_uv = (cell + 0.5) / u_grid;
        vec2 local = fract(v_uv * u_grid);
        vec2 cell_size = 1.0 / u_grid;
        vec3 source_color = adjusted_color(sample_uv);
        float luma = density(sample_uv);
        float glyph = floor(clamp(luma, 0.0, 0.99999) * u_base_glyph_count);

        if (u_mode == 1) {
          glyph = 0.0;
          if (density((cell + vec2(0.25, 0.875)) / u_grid) >= 0.5) glyph += 1.0;
          if (density((cell + vec2(0.25, 0.625)) / u_grid) >= 0.5) glyph += 2.0;
          if (density((cell + vec2(0.25, 0.375)) / u_grid) >= 0.5) glyph += 4.0;
          if (density((cell + vec2(0.75, 0.875)) / u_grid) >= 0.5) glyph += 8.0;
          if (density((cell + vec2(0.75, 0.625)) / u_grid) >= 0.5) glyph += 16.0;
          if (density((cell + vec2(0.75, 0.375)) / u_grid) >= 0.5) glyph += 32.0;
          if (density((cell + vec2(0.25, 0.125)) / u_grid) >= 0.5) glyph += 64.0;
          if (density((cell + vec2(0.75, 0.125)) / u_grid) >= 0.5) glyph += 128.0;
        } else if (u_mode == 2) {
          glyph = 0.0;
          if (density((cell + vec2(0.25, 0.75)) / u_grid) >= 0.5) glyph += 1.0;
          if (density((cell + vec2(0.75, 0.75)) / u_grid) >= 0.5) glyph += 2.0;
          if (density((cell + vec2(0.25, 0.25)) / u_grid) >= 0.5) glyph += 4.0;
          if (density((cell + vec2(0.75, 0.25)) / u_grid) >= 0.5) glyph += 8.0;
        } else if (u_mode == 3) {
          float subcell_y = local.y >= 0.5 ? 0.75 : 0.25;
          vec2 half_uv = (cell + vec2(0.5, subcell_y)) / u_grid;
          vec3 half_color = adjusted_color(half_uv);
          if (u_source_color) {
            out_color = vec4(half_color, 1.0);
          } else {
            out_color = vec4(mix(u_background, u_foreground, density(half_uv)), 1.0);
          }
          return;
        } else if (u_mode == 4) {
          float left = density(sample_uv - vec2(cell_size.x, 0.0));
          float right = density(sample_uv + vec2(cell_size.x, 0.0));
          float below = density(sample_uv - vec2(0.0, cell_size.y));
          float above = density(sample_uv + vec2(0.0, cell_size.y));
          float gx = right - left;
          float gy = above - below;
          float ax = abs(gx);
          float ay = abs(gy);
          if (length(vec2(gx, gy)) >= 0.12) {
            float direction = ax > ay * 2.0
              ? 1.0
              : (ay > ax * 2.0 ? 0.0 : (gx * gy >= 0.0 ? 3.0 : 2.0));
            glyph = u_base_glyph_count + direction;
          }
        }

        float atlas_column = mod(glyph, u_atlas_grid.x);
        float atlas_row = floor(glyph / u_atlas_grid.x);
        vec2 atlas_uv = vec2(
          (atlas_column + local.x) / u_atlas_grid.x,
          (u_atlas_grid.y - atlas_row - 1.0 + local.y) / u_atlas_grid.y
        );
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
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW,
    );
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
    const key = `${options.mode}|${options.charset}|${options.font}`;
    if (key === this.atlasKey) return;
    const glyphs = Array.from(options.charset);
    const cell = 48;
    const columns = Math.min(16, glyphs.length);
    const rows = Math.ceil(glyphs.length / columns);
    const atlas = document.createElement("canvas");
    atlas.width = cell * columns;
    atlas.height = cell * rows;
    const context = atlas.getContext("2d", { alpha: false });
    context.fillStyle = "black";
    context.fillRect(0, 0, atlas.width, atlas.height);
    context.fillStyle = "white";
    context.font = `700 ${Math.round(cell * 0.82)}px ${options.font}`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    glyphs.forEach((glyph, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      context.fillText(
        glyph,
        column * cell + cell / 2,
        row * cell + cell * 0.52,
      );
    });
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.atlasTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      atlas,
    );
    this.atlasKey = key;
    this.atlasGrid = [columns, rows];
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
    if (
      options.sourceWidth !== this.sourceTextureSize[0]
      || options.sourceHeight !== this.sourceTextureSize[1]
    ) {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        source,
      );
      this.sourceTextureSize = [options.sourceWidth, options.sourceHeight];
    } else {
      gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        0,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        source,
      );
    }
    gl.uniform1i(this.uniforms.u_source, 0);
    gl.uniform1i(this.uniforms.u_atlas, 1);
    gl.uniform2f(this.uniforms.u_grid, options.columns, options.rows);
    gl.uniform2f(
      this.uniforms.u_atlas_grid,
      this.atlasGrid[0],
      this.atlasGrid[1],
    );
    gl.uniform1f(this.uniforms.u_base_glyph_count, options.baseGlyphCount);
    gl.uniform1i(this.uniforms.u_mode, MODE_IDS[options.mode]);
    gl.uniform3fv(this.uniforms.u_background, options.background);
    gl.uniform3fv(this.uniforms.u_foreground, options.foreground);
    gl.uniform1i(this.uniforms.u_source_color, options.sourceColor ? 1 : 0);
    gl.uniform1f(this.uniforms.u_brightness, options.brightness);
    gl.uniform1f(this.uniforms.u_contrast, options.contrast);
    gl.uniform1i(this.uniforms.u_invert, options.invert ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

export class CanvasGlyphRenderer {
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
    const frame = sampleGlyphFrame(
      source,
      options,
      this.sample,
      this.sampleContext,
    );
    const cellWidth = options.width / options.columns;
    const cellHeight = options.height / options.rows;
    this.context.fillStyle = options.backgroundCss;
    this.context.fillRect(0, 0, options.width, options.height);

    if (options.mode === "half-block") {
      for (let row = 0; row < options.rows; row += 1) {
        for (let column = 0; column < options.columns; column += 1) {
          const pair = frame.halfColors[row][column];
          this.context.fillStyle = options.sourceColor
            ? rgbCss(pair[0])
            : mixedCss(options, pair[0][3]);
          this.context.fillRect(
            column * cellWidth,
            row * cellHeight,
            cellWidth + 0.5,
            cellHeight / 2 + 0.5,
          );
          this.context.fillStyle = options.sourceColor
            ? rgbCss(pair[1])
            : mixedCss(options, pair[1][3]);
          this.context.fillRect(
            column * cellWidth,
            (row + 0.5) * cellHeight,
            cellWidth + 0.5,
            cellHeight / 2 + 0.5,
          );
        }
      }
      return;
    }

    this.context.font = `700 ${Math.ceil(cellHeight * 0.9)}px ${options.font}`;
    this.context.textAlign = "center";
    this.context.textBaseline = "middle";
    for (let row = 0; row < options.rows; row += 1) {
      for (let column = 0; column < options.columns; column += 1) {
        this.context.fillStyle = options.sourceColor
          ? rgbCss(frame.colors[row][column])
          : options.foregroundCss;
        this.context.fillText(
          frame.glyphRows[row][column],
          (column + 0.5) * cellWidth,
          (row + 0.52) * cellHeight,
        );
      }
    }
  }
}
