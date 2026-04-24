import { defineConfig } from "vite";

const DEFAULT_REPO_NAME = "Repetition";

export default defineConfig(({ command }) => ({
  base:
    command === "serve"
      ? "/"
      : `/${
          process.env.GITHUB_REPOSITORY?.split("/")[1] ?? DEFAULT_REPO_NAME
        }/`
}));
