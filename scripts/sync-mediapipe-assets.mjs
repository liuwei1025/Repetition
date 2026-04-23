import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const rootDir = resolve(scriptDir, "..");
const sourceDir = resolve(rootDir, "node_modules/@mediapipe/tasks-vision/wasm");
const targetDir = resolve(rootDir, "public/vendor/mediapipe/wasm");

if (!existsSync(sourceDir)) {
  throw new Error(`MediaPipe WASM source not found: ${sourceDir}`);
}

mkdirSync(targetDir, { recursive: true });
cpSync(sourceDir, targetDir, { recursive: true });

console.log(`Synced MediaPipe WASM assets to ${targetDir}`);
