import { readFile, stat } from "node:fs/promises";
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import {
  ProjectSessionModel,
  encodeDocument,
  parseDocument,
} from "../../src/glyph_forge/ui/web/project-contract.js";
import { mapPixelGrid } from "../../src/glyph_forge/ui/web/studio-renderers.js";

const renderContract = JSON.parse(
  await readFile("tests/fixtures/render-contract-v1.json", "utf8"),
);
const projectContractText = await readFile("tests/fixtures/project-contract-v1.json", "utf8");
const projectContract = JSON.parse(projectContractText);
const sourceSvg = Buffer.from(
  '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="3">'
  + '<rect width="4" height="3" fill="#ffffff"/><circle cx="2" cy="1.5" r="1"/>'
  + "</svg>",
);

const seriousViolations = (violations) => violations.filter(
  ({ impact }) => impact === "serious" || impact === "critical",
);

test.beforeEach(async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto("./", { waitUntil: "domcontentloaded" });
  await expect(page).toHaveTitle("Glyph Forge Studio");
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(resolve)));
  expect(errors).toEqual([]);
});

test("forges text responsively in every render mode", async ({ page, isMobile }) => {
  await page.getByLabel("Or forge text").fill("Glyph Forge");
  const elapsed = await page.evaluate(() => new Promise((resolve, reject) => {
    const button = document.querySelector("#textSourceButton");
    const grid = document.querySelector("#gridMetric");
    const started = performance.now();
    const timeout = window.setTimeout(() => {
      observer.disconnect();
      reject(new Error("The first text frame did not render within 2 seconds"));
    }, 2_000);
    const observer = new MutationObserver(() => {
      if (grid.textContent === "—") return;
      window.clearTimeout(timeout);
      observer.disconnect();
      resolve(performance.now() - started);
    });
    observer.observe(grid, { childList: true, characterData: true, subtree: true });
    button.click();
  }));
  await expect(page.locator("#sourceName")).toContainText("Text · Glyph Forge");
  expect(elapsed).toBeLessThan(1_000);

  for (const mode of ["glyph", "edge", "braille", "half-block", "quadrant"]) {
    await page.locator("#modeSelect").selectOption(mode);
    await expect(page.locator("#gridMetric")).not.toHaveText("—");
  }
  const canvas = page.locator("#outputCanvas");
  await expect(canvas).toBeVisible();
  const dimensions = await canvas.evaluate(({ width, height }) => ({ width, height }));
  expect(dimensions.width).toBeGreaterThanOrEqual(320);
  expect(dimensions.height).toBeGreaterThanOrEqual(180);
  expect(await page.locator("#engineMetric").textContent()).toMatch(/WebGL2 GPU|Canvas 2D/u);

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  if (isMobile) {
    const stage = await page.locator("#preview").boundingBox();
    const controls = await page.locator("#controls").boundingBox();
    expect(stage.y).toBeLessThan(controls.y);
    const targets = await page.locator("button:visible").evaluateAll((buttons) => (
      buttons.map((button) => ({ label: button.textContent.trim(), height: button.getBoundingClientRect().height }))
    ));
    expect(targets.filter(({ height }) => height < 44)).toEqual([]);
  }
});

test("keeps exact output pixels independent from glyph density", async ({ page }) => {
  await expect(page.locator("#brightnessRange")).toHaveValue("1.12");
  await expect(page.locator("#contrastRange")).toHaveValue("1.08");
  await page.getByLabel("Or forge text").fill("Exact pixels");
  await page.getByRole("button", { name: "Use text" }).click();

  const setControl = (selector, value, eventName = "change") => (
    page.locator(selector).evaluate((input, parameters) => {
      input.value = parameters.value;
      input.dispatchEvent(new Event(parameters.eventName, { bubbles: true }));
    }, { value, eventName })
  );
  await setControl("#outputWidth", "1440");
  await expect(page.locator("#outputHeight")).toHaveValue("810");
  await page.locator("#aspectLock").uncheck();
  await setControl("#outputWidth", "777");
  await setControl("#outputHeight", "333");

  const canvas = page.locator("#outputCanvas");
  await expect.poll(() => canvas.evaluate(({ width, height }) => `${width}x${height}`)).toBe("777x333");
  await expect(page.locator("#outputSizeValue")).toHaveText("777×333 px");
  const gridBefore = await page.locator("#gridMetric").textContent();
  await setControl("#columnsRange", "64", "input");
  await expect(page.locator("#gridMetric")).not.toHaveText(gridBefore);
  expect(await canvas.evaluate(({ width, height }) => [width, height])).toEqual([777, 333]);

  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: "Save SVG" }).click();
  const download = await downloadEvent;
  const svg = await readFile(await download.path(), "utf8");
  expect(svg).toContain('viewBox="0 0 777 333"');
  expect(svg).toContain('textLength="777"');
  expect(svg).toContain("<text");
});

test("ships an accessible installable app shell", async ({ page }) => {
  const manifestResponse = await page.request.get("manifest.webmanifest");
  expect(manifestResponse.ok()).toBeTruthy();
  expect(manifestResponse.headers()["content-type"]).toContain("application/manifest+json");
  const manifest = await manifestResponse.json();
  expect(manifest.display).toBe("standalone");
  expect(manifest.start_url).toBe("./");
  expect(manifest.icons.some(({ sizes }) => sizes === "512x512")).toBeTruthy();

  const offlineShell = await page.evaluate(async () => {
    const registration = await navigator.serviceWorker.ready;
    const names = await caches.keys();
    const studioCache = names.find((name) => name.startsWith("glyph-forge-studio-"));
    const cache = studioCache ? await caches.open(studioCache) : null;
    const script = cache ? await cache.match(new URL("studio.js", document.baseURI)) : null;
    const contract = cache
      ? await cache.match(new URL("project-contract.js", document.baseURI))
      : null;
    return {
      active: Boolean(registration.active),
      studioCache,
      script: Boolean(script),
      contract: Boolean(contract),
    };
  });
  expect(offlineShell.active).toBeTruthy();
  expect(offlineShell.studioCache).toMatch(/^glyph-forge-studio-/u);
  expect(offlineShell.script).toBeTruthy();
  expect(offlineShell.contract).toBeTruthy();

  const results = await new AxeBuilder({ page }).analyze();
  expect(seriousViolations(results.violations)).toEqual([]);
});

test("recovers its app shell while offline", async ({ page, context, browserName }) => {
  test.skip(browserName !== "chromium", "Playwright offline navigation is reliable only in Chromium");
  await page.evaluate(async () => {
    await navigator.serviceWorker.ready;
  });
  await context.setOffline(true);
  try {
    await page.reload({ waitUntil: "domcontentloaded" });
    await expect(page).toHaveTitle("Glyph Forge Studio");
    await expect(page.getByRole("button", { name: "Open something" })).toBeVisible();
  } finally {
    await context.setOffline(false);
  }
});

test("keeps the dependency-free app shell within its performance budget", async () => {
  const files = [
    "src/glyph_forge/ui/web/index.html",
    "src/glyph_forge/ui/web/studio.css",
    "src/glyph_forge/ui/web/studio.js",
    "src/glyph_forge/ui/web/studio-renderers.js",
    "src/glyph_forge/ui/web/project-contract.js",
    "src/glyph_forge/ui/web/service-worker.js",
    "src/glyph_forge/ui/web/manifest.webmanifest",
  ];
  const sizes = await Promise.all(files.map(async (path) => (await stat(path)).size));
  expect(sizes.reduce((total, size) => total + size, 0)).toBeLessThan(160_000);
});

test("shares a strict portable project contract with native interfaces", async ({}, testInfo) => {
  test.skip(
    testInfo.project.name !== "desktop-chromium",
    "The pure document model only needs one JavaScript runtime",
  );
  const project = parseDocument(projectContractText, "project");
  expect(JSON.parse(encodeDocument(project, "project"))).toEqual(projectContract);

  const session = new ProjectSessionModel(project, 3);
  session.addVariant("bright", "Bright");
  session.replaceActiveRequest({ ...session.active.request, brightness: 1.4 });
  expect(session.project.variants).toHaveLength(2);
  expect(session.active.request.brightness).toBe(1.4);
  session.undo();
  expect(session.active.request.brightness).toBe(1.12);
  session.redo();
  expect(session.active.request.brightness).toBe(1.4);

  expect(() => parseDocument(JSON.stringify({
    ...projectContract,
    source: { kind: "image", path: "../escape.png" },
  }), "project")).toThrow(/portable|relative/u);
  expect(() => parseDocument(JSON.stringify({
    ...projectContract,
    surprise: true,
  }), "project")).toThrow(/unknown fields/u);
  expect(() => parseDocument(projectContractText, "unknown")).toThrow(/document kind/u);
  expect(() => parseDocument(JSON.stringify({
    ...projectContract,
    variants: [{
      ...projectContract.variants[0],
      request: {
        ...projectContract.variants[0].request,
        output_format: "html",
        style: "bold",
        output_width: null,
        output_height: null,
      },
    }],
  }), "project")).toThrow(/text styles/u);
});

test("autosaves, restores, variants, and exports browser projects", async ({ page }) => {
  await page.locator("#fileInput").setInputFiles({
    name: "fixture.svg",
    mimeType: "image/svg+xml",
    buffer: sourceSvg,
  });
  await expect(page.locator("#sourceName")).toHaveText("fixture.svg");
  await page.locator("#projectNameInput").fill("Browser fixture");
  await page.getByRole("button", { name: "New project" }).click();
  await expect(page.locator("#projectStatus")).toContainText("autosaved");

  await page.locator("#brightnessRange").evaluate((input) => {
    input.value = "1.35";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.locator("#brightnessValue")).toHaveText("1.35");
  await page.locator("#variantNameInput").fill("Bright poster");
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await expect(page.locator("#variantSelect")).toHaveValue("bright-poster");

  const projectDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Save project" }).click();
  const downloadedProject = JSON.parse(
    await readFile(await (await projectDownload).path(), "utf8"),
  );
  expect(downloadedProject.name).toBe("Browser fixture");
  expect(downloadedProject.source.path).toBe("assets/fixture.svg");
  expect(downloadedProject.variants).toHaveLength(2);
  expect(downloadedProject.variants[1].request.brightness).toBe(1.35);

  const presetDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export preset" }).click();
  const downloadedPreset = JSON.parse(
    await readFile(await (await presetDownload).path(), "utf8"),
  );
  expect(downloadedPreset.schema).toBe("glyph-forge-preset");
  expect(downloadedPreset.request.brightness).toBe(1.35);

  await page.reload({ waitUntil: "domcontentloaded" });
  await expect(page.locator("#projectNameInput")).toHaveValue("Browser fixture");
  await expect(page.locator("#variantSelect")).toHaveValue("bright-poster");
  await expect(page.locator("#projectStatus")).toContainText(/Restored|Recovered/u);

  await page.locator("#projectFileInput").setInputFiles({
    name: "portable.glyphforge.json",
    mimeType: "application/json",
    buffer: Buffer.from(projectContractText),
  });
  await expect(page.locator("#projectNameInput")).toHaveValue("Portable fixture");
  const roundTripDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Save project" }).click();
  const roundTripped = JSON.parse(
    await readFile(await (await roundTripDownload).path(), "utf8"),
  );
  expect(roundTripped.variants[0].request).toEqual(
    projectContract.variants[0].request,
  );
});

test("processes a bounded browser image batch", async ({ page }, testInfo) => {
  test.skip(
    testInfo.project.name !== "desktop-chromium",
    "Multiple automatic downloads are verified in one browser engine",
  );
  const downloads = [];
  page.on("download", (download) => downloads.push(download));
  await page.locator("#batchFileInput").setInputFiles([
    { name: "same.svg", mimeType: "image/svg+xml", buffer: sourceSvg },
    { name: "same.png.svg", mimeType: "image/svg+xml", buffer: sourceSvg },
  ]);
  await expect(page.locator("#batchQueueStatus")).toContainText("2 queued");
  await page.getByRole("button", { name: "Run batch" }).click();
  await expect(page.locator("#batchQueueStatus")).toContainText("Batch complete");
  expect(downloads).toHaveLength(2);
  expect(new Set(downloads.map((download) => download.suggestedFilename())).size).toBe(2);
});

test("matches the shared native/browser render contract", async ({}, testInfo) => {
  test.skip(
    testInfo.project.name !== "desktop-chromium",
    "The deterministic JavaScript kernel only needs one Node/browser project",
  );
  expect(renderContract.contract_version).toBe(1);
  for (const fixture of renderContract.cases) {
    const { sample, web, expected_lines: expectedLines } = fixture;
    const frame = mapPixelGrid(
      new Uint8ClampedArray(sample.rgba),
      sample.width,
      sample.height,
      {
        ...web,
        background: [0, 0, 0],
        foreground: [1, 1, 1],
      },
    );
    expect(frame.lines, fixture.name).toEqual(expectedLines);
    if (fixture.expected_rgb) {
      expect(
        frame.colors.flat().map((color) => color.slice(0, 3)),
        fixture.name,
      ).toEqual(fixture.expected_rgb);
    }
  }
});
