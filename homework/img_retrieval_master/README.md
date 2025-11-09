# 图像检索系统（Node.js + Python）

本项目实现了一个基于公开数据集（TensorFlow Flower Photos）的图像检索系统，后端采用 Node.js（Express）提供 Web API 与静态资源托管，Python 使用预训练 ResNet50 模型提取图像特征、构建索引并执行相似度检索，前端以原生 HTML/CSS/JavaScript 呈现检索界面与结果。

## 功能亮点

- **一键初始化**：提供 npm 脚本自动下载数据集并构建索引。
- **无数据库依赖**：特征向量与元数据全部存储为本地文件，便于离线部署。
- **跨技术栈协作**：Node.js 通过 `child_process` 调用 Python 模块，打通前后端。
- **Web 可视化**：支持图片上传检索、相似图片展示、数据集概览与示例浏览。

## 目录结构

```
img_retrieval_master/
├── data/                    # 数据与索引生成目录
├── docs/                    # 文档资料
├── public/                  # 前端静态资源
├── python/                  # Python 脚本与依赖
│   ├── requirements.txt
│   └── scripts/
│       ├── build_index.py
│       ├── download_dataset.py
│       ├── search.py
│       └── utils.py
└── server/                  # Node.js 服务端
    ├── config.js
    ├── index.js
    └── pythonBridge.js
```

## 环境要求

- Node.js >= 16（推荐 18 或以上）
- Python >= 3.9，且可执行 `python` 命令
- 至少 4GB 空闲磁盘空间，用于存储数据集与索引

## 快速开始

1. **安装依赖**

   ```powershell
   cd C:\deep_learning\deep_learning\homework\img_retrieval_master
   npm install
   pip install -r python\requirements.txt
   ```

2. **下载数据集并构建索引（首次运行需要）**

   ```powershell
    npm run download-data
    npm run build-index
   ```

3. **启动服务**

   ```powershell
   npm start
   ```

   浏览器访问 `http://localhost:3000` 即可体验检索功能。

> 提示：`npm run dev` 使用 `nodemon` 热加载，适合开发调试；如需指定 Python 解释器，可设置环境变量 `PYTHON_BIN`。

## Python 脚本说明

- `download_dataset.py`：下载并解压公开花卉图像数据集。
- `build_index.py`：提取数据集特征向量，生成 `data/artifacts/index.npz`、`metadata.json` 与 `stats.json`。
- `search.py`：读取索引，对输入图片执行相似度检索并返回 JSON 结果。

均可通过模块方式调用，例如：

```powershell
python -m python.scripts.search --query path\to\image.jpg --topk 5
```

## Web API

- `GET /api/health`：检查服务健康状态，若缺少索引会自动触发构建。
- `GET /api/stats`：返回索引统计信息（图像数量、特征维度、生成时间、设备）。
- `GET /api/gallery?limit=24`：返回数据集示例图片及类别列表。
- `POST /api/search`：表单字段 `image` 上传查询图片，可选 `topK` 指定返回数量，返回相似图片列表。

所有图片资源可通过 `/dataset/<相对路径>` 访问（由 Express 静态托管）。

## 前端交互

- 首页展示系统简介与数据统计。
- “上传图片进行检索”区支持拖拽或选择图片、设置返回数量、查看实时状态。
- 检索结果以卡片形式展示，包含相似度与排名。
- 数据集样本区提供部分示例图片及类别概览。

## 配置项

可通过环境变量覆盖 `server/config.js` 中的默认配置：

| 环境变量       | 说明                         | 默认值          |
| -------------- | ---------------------------- | --------------- |
| `PORT`         | 服务监听端口                 | `3000`          |
| `HOST`         | 服务绑定地址                 | `0.0.0.0`       |
| `PYTHON_BIN`   | Python 可执行文件名称/路径   | `python`        |
| `ARTIFACTS_DIR`| 索引文件所在目录             | `data/artifacts`|
| `DATASET_DIR`  | 图像数据集根目录             | `data/dataset`  |
| `STATIC_PREFIX`| 数据集静态访问前缀           | `/dataset`      |
| `MAX_UPLOAD_MB`| 上传图片大小限制（MB）       | `10`            |
| `TOP_K`        | 默认返回的相似图片数量       | `12`            |

## 开发建议

- 首次运行时若下载速度较慢，可手动将 `flower_photos.tgz` 放置到 `data/artifacts` 目录后执行 `npm run build-index`。
- 若需替换数据集，请保证新图像存放于 `data/dataset/<自定义目录>`，并重新运行 `npm run build-index -- --dataset-root your_path`。
- 在 GPU 环境下运行 `python.scripts.build_index` 可自动利用 CUDA 提升特征提取速度。

## 许可

本项目仅供学习与研究使用，数据集版权归原作者所有。
