const express = require("express");
const cors = require("cors");
const morgan = require("morgan");
const multer = require("multer");
const fs = require("fs");
const path = require("path");

const config = require("./config");
const {
  searchSimilarImages,
  ensureIndexReady,
  loadMetadata,
  getPublicImageUrl,
} = require("./pythonBridge");

const app = express();

app.use(cors());
app.use(express.json());
app.use(morgan("dev"));

const uploadsDir = path.join(config.PROJECT_ROOT, "data", "uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: uploadsDir,
  filename: (_req, file, cb) => {
    const timestamp = Date.now();
    const safeName = file.originalname.replace(/[^a-zA-Z0-9.\-_]/g, "_");
    cb(null, `${timestamp}_${safeName}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: config.MAX_UPLOAD_MB * 1024 * 1024,
  },
});

app.use(
  config.STATIC_PREFIX,
  express.static(config.DATASET_DIR, {
    fallthrough: true,
    extensions: ["jpg", "jpeg", "png"],
  })
);

app.get("/api/health", async (_req, res) => {
  try {
    await ensureIndexReady();
    res.json({ status: "ok" });
  } catch (error) {
    res.status(500).json({ status: "error", message: error.message });
  }
});

app.get("/api/gallery", async (req, res) => {
  try {
    await ensureIndexReady();
    const metadata = loadMetadata();
    const limit = Math.min(Number(req.query.limit) || 40, metadata.length);
    const sample = metadata.slice(0, limit).map((item) => ({
      ...item,
      imageUrl: getPublicImageUrl(item.relative_path),
    }));

    const categories = Array.from(
      new Set(metadata.map((item) => item.label))
    ).sort();

    res.json({
      total: metadata.length,
      categories,
      items: sample,
    });
  } catch (error) {
    res.status(500).json({ message: "读取图库失败", detail: error.message });
  }
});

app.get("/api/stats", async (_req, res) => {
  try {
    await ensureIndexReady();
    const statsPath = path.join(config.ARTIFACTS_DIR, "stats.json");
    if (fs.existsSync(statsPath)) {
      const content = fs.readFileSync(statsPath, "utf-8");
      res.json(JSON.parse(content));
    } else {
      res.json({ message: "暂无统计信息" });
    }
  } catch (error) {
    res
      .status(500)
      .json({ message: "读取统计信息失败", detail: error.message });
  }
});

app.post("/api/search", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: "请上传名为 image 的查询图片" });
  }

  try {
    const result = await searchSimilarImages(
      req.file.path,
      Number(req.body.topK) || config.TOP_K
    );
    res.json(result);
  } catch (error) {
    console.error("检索失败", error);
    res.status(500).json({ message: "检索失败", detail: error.message });
  } finally {
    fs.promises
      .unlink(req.file.path)
      .catch((err) => console.warn(`删除临时文件失败: ${err.message}`));
  }
});

app.use(express.static(path.join(config.PROJECT_ROOT, "public")));

app.use((req, res) => {
  res.status(404).json({ message: "未找到资源" });
});

function startServer() {
  app.listen(config.PORT, config.HOST, () => {
    console.log(`Server listening on http://${config.HOST}:${config.PORT}`);
  });
}

startServer();
