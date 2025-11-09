系统架构概览
==========

总体目标
--------

构建一个基于公开数据集的图像检索 Web 系统，使用 Node.js 提供 Web 服务与前端资源，Python 负责图像特征提取与检索逻辑，尽量避免依赖数据库，将索引持久化为文件。

目录结构
--------

- `server/`：Node.js Express 服务，统一对外提供 API，处理查询请求并与 Python 脚本交互。
- `public/`：前端静态资源（HTML、CSS、JavaScript），由 Express 直接托管。
- `python/`：Python 相关脚本与依赖，负责数据集下载、特征提取、索引构建与查询执行。
- `data/`：运行过程中生成或下载的数据，包括原始图片（子目录 `dataset/`）与特征索引（`artifacts/`）。
- `docs/`：项目文档。

关键流程
--------

1. **数据准备**：通过 `python/scripts/download_dataset.py` 下载公开数据集（TensorFlow Flower Photos），解压到 `data/dataset/`。
2. **索引构建**：运行 `python/scripts/build_index.py`，使用 PyTorch 的预训练 `resnet50` 提取图像特征，生成并保存索引文件（默认保存在 `data/artifacts/index.npz` 与 `data/artifacts/metadata.json`）。
3. **查询检索**：Web 前端上传查询图片，Node.js 接收后通过 `child_process` 调用 `python/scripts/search.py`，Python 加载索引并返回最相似图像的元信息，Node.js 将结果转发给前端展示。
4. **静态资源访问**：检索结果中的图片直接从 Express 静态目录（指向 `data/dataset/`）提供。

技术栈与约束
------------

- **后端**：Node.js (Express)、Multer（文件上传）、CORS、中间件处理。
- **前端**：原生 HTML/CSS/JavaScript，Fetch API 调用后端接口。
- **Python**：`torch`、`torchvision`、`numpy`、`tqdm`、`requests`、`Pillow` 等库。
- **数据存储**：索引数据存放于本地文件，无需数据库。
- **运行方式**：通过 npm 脚本启动 Node.js 服务；Python 脚本独立运行或由 Node.js 调用。

当前状态
--------

- Python 数据处理脚本、依赖管理已完成。
- Node.js API 及前端界面已落地并可运行。
- README 提供了初始化步骤与运行说明。

