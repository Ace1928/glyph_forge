import { stat } from "node:fs/promises";
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
  const started = await page.evaluate(() => performance.now());
  await page.getByRole("button", { name: "Use text" }).click();
  await expect(page.locator("#sourceName")).toContainText("Text · Glyph Forge");
  const elapsed = await page.evaluate((value) => performance.now() - value, started);
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
