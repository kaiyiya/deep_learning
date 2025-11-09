const { spawn } = require("child_process");
const path = require("path");

const config = require("./config");

function runPythonModule(moduleName, args = []) {
  return new Promise((resolve, reject) => {
    const pythonArgs = ["-m", moduleName, ...args];
    const child = spawn(config.PYTHON_BIN, pythonArgs, {
      cwd: config.PROJECT_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("close", (code) => {
      if (code !== 0) {
        const error = new Error(`Python 脚本退出码 ${code}: ${stderr}`);
        error.code = code;
        reject(error);
      } else {
        resolve(stdout.trim());
      }
    });
  });
}

let buildingPromise = null;

async function ensureIndexReady() {
  const fs = require("fs");
  const indexPath = path.join(config.ARTIFACTS_DIR, "index.npz");
  const metadataPath = path.join(config.ARTIFACTS_DIR, "metadata.json");
  if (!fs.existsSync(indexPath) || !fs.existsSync(metadataPath)) {
    if (!buildingPromise) {
      buildingPromise = runPythonModule(
        "python.scripts.build_index",
        []
      ).finally(() => {
        buildingPromise = null;
      });
    }
    await buildingPromise;
  }
}

async function searchSimilarImages(queryPath, topK = config.TOP_K) {
  await ensureIndexReady();
  const stdout = await runPythonModule("python.scripts.search", [
    "--query",
    queryPath,
    "--topk",
    String(topK),
    "--artifacts",
    config.ARTIFACTS_DIR,
  ]);
  const payload = JSON.parse(stdout);
  payload.results = payload.results.map((item) => ({
    ...item,
    imageUrl: getPublicImageUrl(item.relative_path),
  }));

  return payload;
}

function loadMetadata() {
  const fs = require("fs");
  const metadataPath = path.join(config.ARTIFACTS_DIR, "metadata.json");
  if (!fs.existsSync(metadataPath)) {
    throw new Error("metadata.json 不存在，请先构建索引");
  }
  const content = fs.readFileSync(metadataPath, "utf-8");
  return JSON.parse(content);
}

function getPublicImageUrl(relativePath) {
  const normalized = relativePath.replace(/\\/g, "/").replace(/^dataset\//, "");
  return `${config.STATIC_PREFIX}/${normalized}`;
}

module.exports = {
  runPythonModule,
  searchSimilarImages,
  ensureIndexReady,
  loadMetadata,
  getPublicImageUrl,
};
