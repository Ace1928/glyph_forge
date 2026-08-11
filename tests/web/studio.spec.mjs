import { readFile, stat } from "node:fs/promises";
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

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
    return { active: Boolean(registration.active), studioCache, script: Boolean(script) };
  });
  expect(offlineShell.active).toBeTruthy();
  expect(offlineShell.studioCache).toMatch(/^glyph-forge-studio-/u);
  expect(offlineShell.script).toBeTruthy();

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
    "src/glyph_forge/ui/web/service-worker.js",
    "src/glyph_forge/ui/web/manifest.webmanifest",
  ];
  const sizes = await Promise.all(files.map(async (path) => (await stat(path)).size));
  expect(sizes.reduce((total, size) => total + size, 0)).toBeLessThan(160_000);
});
