const path = require("path");

const projectRoot = path.resolve(__dirname, "..");
const dataRoot = path.join(projectRoot, "data");

module.exports = {
  PORT: process.env.PORT || 3000,
  HOST: process.env.HOST || "0.0.0.0",
  PYTHON_BIN: process.env.PYTHON_BIN || "python",
  ARTIFACTS_DIR: process.env.ARTIFACTS_DIR || path.join(dataRoot, "artifacts"),
  DATASET_DIR: process.env.DATASET_DIR || path.join(dataRoot, "dataset"),
  STATIC_PREFIX: process.env.STATIC_PREFIX || "/dataset",
  MAX_UPLOAD_MB: Number(process.env.MAX_UPLOAD_MB || 10),
  TOP_K: Number(process.env.TOP_K || 12),
  PROJECT_ROOT: projectRoot,
};

