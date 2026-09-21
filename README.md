# 日语发音评估与语音识别系统（OtoKoeNet）

基于深度学习的日语口语辅助系统：一个 **DualCTC 语音识别模型** + **FastAPI 后端**。
支持两个核心功能：

- **语音识别（听写）**：上传/录音任意音频 → 输出日文文本（假名或汉字）。
- **发音评估**：给定参考文本与用户音频 → 强制对齐 → 音素级打分（绿/黄/红）。

## 功能特性

- **任意格式音频识别**：`wav / flac / ogg / mp3 / m4a / aac / wma / opus` 等。
  内部先走 libsndfile（wav/flac/ogg/opus/mp3），不支持时自动回退 ffmpeg。
- **双路解码**：已知句走「句库最近邻」，未知句走「假名 → 汉字」转换。
- **用户系统**：JWT 鉴权、注册/登录、测试历史记录。
- **模型**：Conformer Encoder + 双 CTC 头（字符 / 莫拉 mora），EMA 训练。

## 目录结构

```
├── otokoenet/                # 模型核心代码
│   ├── model.py            # DualCTC（Conformer Encoder + 双 CTC 头）
│   ├── data.py             # 音频加载（含 ffmpeg 回退）、fbank、CMVN、数据装载
│   ├── decode.py           # CTC 折叠、句库最近邻、编辑距离、评估
│   ├── text.py             # 文本规范化、假名→莫拉、词表
│   ├── kana2kanji.py       # 假名→汉字 Viterbi 转换器
│   ├── align.py            # 发音评估：强制对齐 + 打分
│   └── train.py            # 训练入口
├── scripts/
│   ├── prepare.py          # JSUT 语料 → 特征缓存 + train/test manifest
│   ├── build_kana2kanji.py # 用语料 + JMdict 构建假名→汉字表
│   ├── infer.py            # 单音频文件识别 CLI
│   ├── eval_recognize.py   # 识别效果评估（CER）
│   └── eval_open.py        # 开放词汇评估
├── backend/                # FastAPI 服务
│   ├── run.py              # 启动入口（uvicorn）
│   └── app/                # 路由、鉴权、数据库、模型引擎
├── configs/basic5000.yaml  # 训练配置
├── requirements.txt        # 模型/脚本依赖
└── backend/requirements.txt# 后端依赖
```

## 环境与依赖

要求 Python 3.13+（建议用 venv / uv）。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -r backend/requirements.txt
```

可选：安装 [ffmpeg](https://ffmpeg.org/) 并加入 PATH —— 识别 `mp3/m4a/aac/wma` 等格式时必需。

## 数据准备

识别模型在 [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)（日语朗读语料，`basic5000` 子集）上训练。**语料库不入 Git**，部署时需自行下载：

```bash
# 1. 下载 JSUT basic5000 并解压到 data/jsut_ver1.1/（包含 basic5000/wav、transcript_utf8.txt）

# 2. 提取特征并划分 train/test（默认留出 200 句）
python scripts/prepare.py --jsut-root data/jsut_ver1.1 --cache-dir data/cache/basic5000

# 3. 构建 假名→汉字 转换表（需要 JMdict_e.gz 放到 data/jmdict/）
python scripts/build_kana2kanji.py \
  --transcript data/jsut_ver1.1/basic5000/transcript_utf8.txt \
  --jmdict data/jmdict/JMdict_e.gz \
  --out data/cache/basic5000/kana2kanji.json
```

以上命令会生成：`char_vocab.json`、`mora_vocab.json`、`mean.npy`、`std.npy`、`train.json`、`test.json`、各句 `.npy` 特征。

## 模型训练

```bash
python otokoenet/train.py --config configs/basic5000.yaml
```

训练产出存到 `runs/<save_dir>/`，其中 `best.pt` / `last.pt` 含 `model`、`ema`、`config` 字段。

## 识别与评估

```bash
# 任意音频文件 → 日文文本（known：句库命中；open：假名→汉字）
python scripts/infer.py data/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav

# 在留出的 test 集上评估 CER（closed / open 分路）
python scripts/eval_recognize.py --ckpt runs/basic5000/best.pt

# 开放词汇评估（用 utparaphrase512 语料，训练未见过）
python scripts/eval_open.py
```

## 后端 API 与数据库

### 建数据库

默认 **SQLite**，库文件 `data/app.db`，**启动后端时自动建表**（无需手动初始化）。
也可手动建库：

```bash
cd backend
python -c "from app.database import init_db; init_db()"
```

切换为 MySQL / PostgreSQL 等，只需设置环境变量（需另装对应驱动，如 `pip install pymysql`）：

```bash
export OTOKOE_DB_URL="mysql+pymysql://user:pass@127.0.0.1:3306/otokoenet"
```

表结构（见 `backend/app/models.py`）：

| 表 | 说明 | 主要字段 |
| --- | --- | --- |
| `users` | 用户 | user_id, username, password_hash, created_at |
| `records` | 测试记录 | record_id, user_id(FK), test_type, ref_text, result_text, score, audio_path, created_at |

### 启动后端

```bash
cd backend
python run.py            # http://127.0.0.1:8000
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

接口文档：启动后访问 `http://127.0.0.1:8000/docs`（Swagger）。

### 接口一览

| 方法 | 路径 | 鉴权 | 说明 |
| --- | --- | --- | --- |
| POST | `/api/auth/register` | 否 | 注册，返回 `access_token` |
| POST | `/api/auth/login` | 否 | 登录，返回 `access_token` |
| POST | `/api/recognize` | Bearer | multipart 上传 `file`，返回 `{recognized_text}` |
| POST | `/api/evaluate` | Bearer | multipart 上传 `file` + `ref_text`，返回逐音素得分 |
| GET | `/api/texts` | 否 | 练习题文本库（含假名注音） |
| GET | `/api/records` | Bearer | 当前用户测试历史 |
| GET | `/api/health` | 否 | 健康检查 |

鉴权流程：`register`/`login` 返回 `access_token` → 后续请求带请求头
`Authorization: Bearer <token>`。

环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OTOKOE_DB_URL` | `sqlite:///data/app.db` | 数据库连接串 |
| `OTOKOE_SECRET` | `dev-secret-change-me` | JWT 密钥，**生产必须改** |

## 前端与后端协作

系统为**前后端分离**架构，前端只需按约定对接 `/api/*`。

### 1. 约定

- 所有业务接口统一挂在 `/api` 前缀下；前端只访问 `/api`。
- CORS 已全开（`allow_origins=["*"]`），开发期跨域直接可调。
- 音频上传字段名：`file`；评估额外字段：`ref_text`。
- 后端整文件读入内存，上传音频不宜过大（单文件建议 < 50MB）。

### 2. 本地开发（前端 dev server + 后端 8000）

Vite 等 dev server 配置 proxy，把 `/api` 转发到后端：

```js
// vite.config.js
export default {
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
};
```

### 3. 生产部署（nginx 托管前端 + 反代后端，推荐）

```
server {
    listen 80;
    client_max_body_size 50m;          # 音频上传较大时调大

    location / {
        root /var/www/otokoenet-frontend/dist;   # 前端构建产物
        try_files $uri /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

后端生产运行（注意每个 worker 都会加载一份模型，按内存决定 worker 数）：

```bash
cd backend
OTOKOE_SECRET="换成强随机值" uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. 部署要点

- **模型文件**：确保 `runs/basic5000/best.pt` 与 `data/cache/basic5000/*`（词表、mean/std、train/test manifest）在部署机存在，路径与
  `backend/app/config.py` 中的 `ckpt_path` / `cache_dir` 一致（或用软链/环境变量调整）。
- **首发启动**：FastAPI 的 lifespan 会自动 `init_db()` 并加载模型。
- **鉴权**：Token 24h 有效；生产务必修改 `OTOKOE_SECRET`。

## GitHub 发布指引

### 该推什么 / 不该推什么

仓库里**没有前端**（前端为 Vue/React 独立工程，另建仓库）；本仓库 = **模型源码 + 后端**。

| 类别 | 内容 | 建议 |
| --- | --- | --- |
| 源码 | `otokoenet/`、`scripts/`、`backend/`、`configs/`、`develop.md` | ✅ 推 |
| 依赖 | `requirements.txt`、`backend/requirements.txt` | ✅ 推 |
| 文档 | `README.md`、`.gitignore` | ✅ 推 |
| 模型权重 | `runs/basic5000/best.pt`（约 75MB） | ⚠️ 可选，用 Git LFS 或提供网盘链接 |
| 语料/特征 | `data/`（约 4GB，含 JSUT 语料与缓存） | ❌ 不推，部署时重新下载生成 |
| 其余检查点 | `runs/` 其它 `.pt`（共约 460MB） | ❌ 不推 |
| 环境 | `.venv/`、`log/`、`.idea/`、`__pycache__/` | ❌ 不推（已进 `.gitignore`） |
| 资料 | `参考论文/`（版权 PDF） | ❌ 不推 |

### 推送步骤

```bash
# 1. 初始化 LFS（可选，仅当要推模型权重）
git lfs install
git lfs track "runs/basic5000/best.pt"

# 2. 提交
git add .
git status          # 确认 data/、runs/ 等未被加入
git commit -m "..."

# 3. 关联远端并推送
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git branch -M main
git push -u origin main
```

> 若发现 `data/`、`runs/`、`.venv` 被 add，检查 `.gitignore` 后再 `git rm -r --cached <路径>`。

## 参考

- JSUT corpus：https://sites.google.com/site/shinnosuketakamichi/publication/jsut
- JMdict / JMnedict：https://www.edrdg.org/jmdict/