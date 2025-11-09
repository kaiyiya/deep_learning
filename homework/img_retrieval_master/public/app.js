const $ = (selector) => document.querySelector(selector);

const statsEls = {
  totalImages: $("#total-images"),
  embeddingDim: $("#embedding-dim"),
  generatedAt: $("#generated-at"),
  device: $("#device"),
};

const galleryInfoEl = $("#gallery-info");
const galleryGridEl = $("#gallery-grid");
const searchForm = $("#search-form");
const fileInput = $("#image-input");
const fileLabel = $("#file-label");
const resultsEl = $("#results");
const statusEl = $("#search-status");

function formatDate(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

async function loadStats() {
  try {
    const response = await fetch("/api/stats");
    if (!response.ok) throw new Error("统计信息获取失败");
    const data = await response.json();
    statsEls.totalImages.textContent = data.total_images ?? "--";
    statsEls.embeddingDim.textContent = data.embedding_dim ?? "--";
    statsEls.generatedAt.textContent = formatDate(data.generated_at);
    statsEls.device.textContent = data.device ?? "--";
  } catch (error) {
    console.warn(error);
    statsEls.totalImages.textContent = "待生成";
    statsEls.embeddingDim.textContent = "--";
    statsEls.generatedAt.textContent = "--";
    statsEls.device.textContent = "--";
  }
}

function createCard(item) {
  const container = document.createElement("div");
  container.className = "card-item";

  const img = document.createElement("img");
  img.src = item.imageUrl;
  img.alt = item.label;
  img.loading = "lazy";

  const body = document.createElement("div");
  body.className = "card-body";

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = item.label;

  const name = document.createElement("span");
  name.className = "value";
  name.textContent = item.filename;

  body.appendChild(label);
  body.appendChild(name);

  if (item.rank !== undefined && item.score !== undefined) {
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = `相似度：${item.score.toFixed(4)} | 排名 ${item.rank}`;
    body.appendChild(score);
  }

  container.appendChild(img);
  container.appendChild(body);
  return container;
}

async function loadGallery() {
  try {
    const response = await fetch("/api/gallery?limit=24");
    if (!response.ok) throw new Error("图库信息获取失败");
    const data = await response.json();
    galleryInfoEl.textContent = `已加载 ${data.items.length}/${data.total} 张图片，类别数 ${data.categories.length}`;
    galleryGridEl.innerHTML = "";
    data.items.forEach((item) => {
      galleryGridEl.appendChild(createCard(item));
    });
  } catch (error) {
    galleryInfoEl.textContent = `加载失败：${error.message}`;
  }
}

fileInput.addEventListener("change", () => {
  if (fileInput.files?.length) {
    fileLabel.textContent = `已选择：${fileInput.files[0].name}`;
  } else {
    fileLabel.textContent = "选择或拖拽一张图片";
  }
});

searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files?.length) {
    statusEl.textContent = "请先选择图片";
    statusEl.classList.add("error");
    return;
  }

  statusEl.textContent = "正在检索，请稍候...";
  statusEl.classList.remove("error");
  resultsEl.innerHTML = "";

  const formData = new FormData(searchForm);

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.message || "检索失败");
    }

    const data = await response.json();
    if (!data.results?.length) {
      statusEl.textContent = "未找到相似图片，请尝试其他图片。";
      return;
    }

    statusEl.textContent = `检索完成，返回 ${data.results.length} 个结果。`;
    data.results.forEach((item) => {
      resultsEl.appendChild(createCard(item));
    });
  } catch (error) {
    statusEl.textContent = `检索失败：${error.message}`;
    statusEl.classList.add("error");
  }
});

window.addEventListener("DOMContentLoaded", () => {
  loadStats();
  loadGallery();
});
