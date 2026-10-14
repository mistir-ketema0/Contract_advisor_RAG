import { test, expect } from "@playwright/test";
import { resolve } from "node:path";

const fixture = resolve(
  import.meta.dirname,
  "../../examples/sample-contract.txt",
);

test.beforeEach(async ({ request }) => {
  const documents = await (await request.get("/api/documents")).json();
  for (const document of documents)
    await request.delete(`/api/documents/${document.id}`);
});

test("upload → select → question → answer and original citation", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /Your contracts,/ }),
  ).toBeVisible();
  await page.getByLabel("Upload contract").setInputFiles(fixture);
  await expect(
    page.getByText("sample-contract.txt", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "What is the termination notice period?" })
    .click();
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.locator(".answer-text")).toContainText("60 days");
  await expect(page.locator(".sources")).toContainText("5. Termination");
  await expect(page.locator("blockquote")).toContainText("60 days");
  await expect(page.locator(".sources")).toContainText("Page 1");
  await expect(
    page.getByText("Extractive demo · No AI model used"),
  ).toBeVisible();
  expect(errors).toEqual([]);
  if (process.env.CAPTURE_DEMO)
    await page.screenshot({
      path: resolve(import.meta.dirname, "../../docs/demo.png"),
      fullPage: true,
    });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Clear conversation" }).click();
  await page
    .getByLabel("Ask about your contracts")
    .fill("What is the annual insurance premium?");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.locator(".answer-heading")).toContainText(
    "Insufficient evidence",
  );
  await expect(page.locator(".sources")).toHaveCount(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});
