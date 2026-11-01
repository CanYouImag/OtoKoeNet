# 日语发音评估与语音识别系统（OtoKoeNet）

基于深度学习的日语口语辅助系统：一个 **DualCTC 语音识别模型** + **FastAPI 无状态后端**。
系统为**前后端分离**架构，前端是独立的 Vue/React 工程（**本仓库不含前端**），
后端只做推理、不存任何用户数据。

## 项目能干什么

**两大核心功能**（`backend/app/routers/`）：

- **语音识别（听写）** `POST /api/recognize`：上传/录音任意音频 → 输出日文文本（假名或汉字）。
- **发音评估** `POST /api/evaluate`：给定参考文本 + 用户音频 → CTC 强制对齐 → 逐音节（mora）打分（绿 ≥0.75 / 黄 ≥0.45 / 红），并给出总评分。

**能力明细**

| 能力 | 说明 |
| --- | --- |
| 任意格式音频 | `wav/flac/ogg/mp3/opus` 走 libsndfile，`mp3/m4a/aac/wma/mp4/webm` 自动回退 ffmpeg |
| 双路解码 | 已知句走「句库最近邻」，未知句走「假名 → 汉字」Viterbi 转换 |
| Char-Mora 联合解码 | 阶段 19：mora beam + 假名→汉字 与 char beam 候选合池重排，汉字 CER 32%→27.6% |
| 发音评估细节 | 每个 mora 返回 `score/color/index/start_ms/end_ms/duration_ms/rel_duration/suspect_long_vowel` |
| 長音疑似漏读 | `suspect_long_vowels` 规则：只加标志、不改分（低假阳率工作点） |
| 拒绝机制 | `AlignmentError`/词表外 mora/损坏/为空/过短音频 → 400；整句 `min_mean_logprob<-3.0` 判「不符」 |
| OOD 检测 | 阶段 19 逻辑回归检测域外输入，默认关闭（`OTOKOE_OOD=1` 启用） |
| 无状态部署 | 免数据库、免鉴权；上传音频**推理后即删**，测试记录存于浏览器本地 |
| 模型 | Conformer Encoder + 双 CTC 头（字符 / mora），EMA 训练 |

## 系统设计

前后端分离：前端负责录音/上传与结果可视化，后端提供 RESTful API，核心模型独立部署。

- **前端交互层**：Vue 3 + Element Plus / React + Ant Design。负责在线录音（麦克风录 WAV）、文件上传、结果可视化（汉字注音高亮）。
- **后端服务层**：Python + FastAPI。音频预处理、模型调度、结果返回，无业务状态。
- **算法模型层**：PyTorch。Conformer Encoder（4× 时间下采样，80 维 Kaldi fbank）+ 双 CTC 头；发音评估用 **CTC 强制对齐 + GOP 打分**，识别用 prefix beam search。
- **数据处理层**：pykakasi + jaconv + Sudachi（文本规范化 / 假名→mora / G2P）、torchaudio/kaldi fbank。
- **数据存储层**：**无数据库**。测试记录存于前端浏览器本地（localStorage / IndexedDB），音频不持久化。

**核心数据流**：

```
JSUT 语料 → scripts/prepare.py（fbank + CMVN 烘焙 + train/val/test 划分）
         → otokoenet/train.py（DualCTC 双头 CTC + EMA） → runs/<exp>/best.pt
         → backend/app/ml/engine.py（在线 fbank+CMVN → beam search → 句库/转换表/联合解码）
         → /api/recognize 与 /api/evaluate → 前端
```

## 目录结构

```
OtoKoeNet/
├── otokoenet/            # 模型核心（PyTorch）
│   ├── model.py        # DualCTC：ConvSubsampling + ConformerEncoder + 双 CTC 头 / 中间层 CTC
│   ├── data.py         # 音频加载（ffmpeg 回退）、Kaldi fbank、CMVN、缓存数据集、SpecAugment
│   ├── decode.py       # CTC 折叠、prefix beam search、n-gram LM、句库最近邻、编辑距离
│   ├── joint_decode.py # Char-Mora 联合解码（阶段 19）
│   ├── text.py         # 文本规范化、假名→mora、词表
│   ├── kana2kanji.py   # 假名→汉字 Viterbi 转换器
│   ├── align.py        # CTC 强制对齐 + GOP 打分、長音规则、不可行拒绝
│   ├── ood.py          # OOD 检测（LogisticOOD，默认关）
│   ├── augment.py      # MUSAN 噪声 + RIR 混响（波形级增强，仅 train）
│   └── train.py        # 训练循环、EMA、WarmupCosine、SpecAugment/speed_perturb、断点续训
├── scripts/             # 数据准备 / 评测 / 回归测试
│   ├── prepare.py      # JSUT→特征缓存+manifest；build_kana2kanji.py；prepare_hf.py；check_cmvn.py
│   ├── infer.py        # 单音频文件识别 CLI
│   ├── eval_recognize.py / eval_open.py / eval_5a.py        # 识别评估
│   ├── eval_engine_e2e.py / eval_pronun.py / eval_long_vowel.py
│   ├── eval_joint_decode.py / train_ood.py / fit_calibration.py
│   ├── test_align.py / test_mask_invariance.py / test_engine_load.py / test_joint_decode.py
│   └── train_tmux.sh / download_musan.sh
├── backend/             # FastAPI 无状态推理服务
│   ├── run.py          # uvicorn 启动入口
│   └── app/
│       ├── main.py     # 应用装配、CORS、健康检查
│       ├── config.py   # 部署默认（ckpt/cache 路径、解码参数、开关）
│       ├── schemas.py  # Pydantic 模型（MoraScore 等）
│       ├── bank.py     # 练习题库与 text_to_mora
│       ├── ml/engine.py# 推理引擎（严格加载、fbank+CMVN、联合解码、对齐）
│       └── routers/    # recognize / evaluate / texts
├── configs/             # 训练配置（basic5000_*、jsut_full_stage{14,17,18}、stage16_grad_probe）
├── requirements.txt     # 模型/脚本依赖
└── backend/requirements.txt  # 后端依赖

# 不入 Git（.gitignore 已忽略，见「GitHub 发布指引」）
data/    # 原始语料、特征缓存、词典、MUSAN/RIRS（约十余 GB）
runs/    # 训练产物（best.pt / last.pt / log.csv）
log/     # 19 个阶段的实验日志与验收记录（本地账本）
backup_*/  # 本地模型快照
```

## 启动后端（最简运行方式）

### 1. 环境与依赖

要求 Python 3.11+（建议 venv / uv）：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt        # PyTorch、torchaudio、numpy、pykakasi、jaconv 等
pip install -r backend/requirements.txt  # fastapi、uvicorn、python-multipart
```

可选：安装 [ffmpeg](https://ffmpeg.org/) 并加入 PATH —— 识别 `mp3/m4a/aac/wma` 等格式时必需。

### 2. 必需文件（不在 Git 里，需自行准备）

`backend/app/config.py` 的默认路径指向：

| 文件 | 作用 |
| --- | --- |
| `runs/jsut_full_stage14/best.pt` | 模型 checkpoint（当前生产基线，ckpt_step=27249） |
| `data/cache/jsut_full_v1/{char_vocab.json, mora_vocab.json, mean.npy, std.npy, train.json}` | 词表、CMVN、训练 manifest |

可选用 `OTOKOE_STRICT_LOAD=0` 之类关闭严格校验，但 `cache` 与 `ckpt` 不一致默认会**启动失败**。缺 `kana2kanji.json` 时识别只回退输出假名（有警告）。

### 3. 启动

```bash
cd backend
python run.py            # http://127.0.0.1:8000
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动时 lifespan 自动加载模型，任一前置文件缺失会直接启动失败。Swagger 文档：`http://127.0.0.1:8000/docs`。

> **注意**：必须在 `backend/` 目录下运行（`config.py` 以仓库根定位数据路径）。

### 4. 环境变量（全部可选，均有默认值）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `OTOKOE_JOINT_DECODE` | `1` | 启用 Char-Mora 联合解码（`0` 关闭，缺 pykakasi 自动回退） |
| `OTOKOE_MIN_MEAN_LOGPROB` | `-3.0` | 整句对齐最小平均对数概率阈值（低于则判「不符」） |
| `OTOKOE_STRICT_LOAD` | `1` | 严格加载 checkpoint（缺键/形状不符立即失败） |
| `OTOKOE_ALLOW_CACHE_MISMATCH` | `0` | 允许 cache/ckpt 指纹不一致启动 |
| `OTOKOE_OOD` | `0` | 启用 OOD 检测 |
| `OTOKOE_OOD_DETECTOR` | `data/cache/jsut_full_v1/ood_detector.json` | OOD 检测器路径 |

### 5. 接口一览（全部免鉴权）

| 方法 | 路径 | 请求 | 响应 |
| --- | --- | --- | --- |
| POST | `/api/recognize` | multipart `file`（audio） | `{ "recognized_text": "こんにちは" }` |
| POST | `/api/evaluate` | multipart `file` + `ref_text` | 见下 |
| GET | `/api/texts` | — | 带假名注音的练习题库 |
| GET | `/api/health` | — | `{ "status": "ok", "model": {...} }`（确认模型已加载） |

`/api/evaluate` 响应示例：

```json
{
  "ref_text": "水をマレーシアから買わなくてはならないのです。",
  "total_score": 89.1,
  "duration": 0.0336,
  "details": [
    { "phoneme": "み", "score": 79.6, "color": "green", "index": 0,
      "start_ms": 320.0, "end_ms": 360.0,
      "duration_ms": 42.9, "rel_duration": 0.41, "suspect_long_vowel": false }
  ]
}
```

`details[].color`：green≥0.75 / yellow≥0.45 / red。`suspect_long_vowel` 只作提示、**不自动改分**。
错误语义（400，JSON body）：`align_failed`（对不齐）、`ood_audio`（启用 OOD 时域外输入）、词表外 mora 明文提示。

## 前端对接

### 接口约定

- 所有业务接口统一挂在 `/api` 前缀下，CORS 已全开（`allow_origins=["*"]`）。
- **无需鉴权**：不要求任何请求头，无用户概念。
- 音频上传字段名：`file`（评估额外加 `ref_text`）；后端整文件读入内存，单文件建议 < 50MB。
- 后端音频**推理后即删**，前端需要回放时应自留音频 blob。

### 前端功能需求

**发音评估模块**
- 文本展示与注音：展示练习文本（假名、单词、含促音/拗音的句子），用 `<ruby>` 实现汉字上方假名注音。
- 音频采集：在线录音（浏览器麦克风录 WAV）或文件上传（本地 MP3/WAV）。
- 结果反馈：按颜色区分发音质量并显示总评分，标注疑似漏读的長音。

**语音识别模块**
- 自由输入：无需参考文本，直接录音或上传音频。
- 内容识别：模型返回日文文本序列。
- 结果展示：文本框展示识别出的假名/汉字，支持复制。

### 本地存储设计（测试记录）

后端**不持久化任何用户数据**，测试记录由前端写入浏览器本地：

- **localStorage**：文本与分数类记录（JSON，键如 `otokoenet.records`）。
- **IndexedDB**：录音回放音频 blob（可选，超出 localStorage 5MB 限额时使用）。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 唯一标识 |
| `test_type` | string | `EVALUATE` / `RECOGNIZE` |
| `ref_text` | string | 参考文本（评估模式用） |
| `result_text` | string | 识别结果（识别模式用） |
| `score` | float | 综合得分（评估） |
| `details` | array | 逐音素得分详情（评估，同 `/api/evaluate` 的 details） |
| `audio_blob` | Blob | 录音回放（存 IndexedDB，可选） |
| `created_at` | datetime | 测试时间 |

### 本地开发（前端 dev server + 后端 8000）

Vite 等 dev server 配置 proxy 把 `/api` 转发到后端：

```js
// vite.config.js
export default {
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
};
```

### 生产部署（nginx 托管前端 + 反代后端）

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

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000   # 每个 worker 各加载一份模型，按内存定 worker 数
```

## 数据准备与模型训练（复现/重训）

> 语料与训练产物均**不入 Git**，需按以下步骤重新生成。生产基线相关产物见 `log/`。

### 数据准备

```bash
# 1. 下载 JSUT ver1.1（9 子集约 4GB）解压到 data/jsut_ver1.1/（wav + transcript_utf8.txt）

# 2. 抽取特征 + 划分（生产用 jsut_full_v1，含 train/val/test/probe 与词表/CMVN）
python scripts/prepare.py --jsut-root data/jsut_ver1.1 --cache-dir data/cache/jsut_full_v1 \
  --subsets basic5000,countersuffix26,loanword128,onomatopee300,precedent130,repeat500,travel1000,utparaphrase512,voiceactress100 \
  --vocab-from <上一份词表目录> --holdout-from <上一份缓存目录>   # 保持 val/test 可比、词表前缀一致

# 3. 构建 假名→汉字 转换表（需要 JMdict_e.gz 放到 data/jmdict/）
python scripts/build_kana2kanji.py \
  --transcript data/jsut_ver1.1/basic5000/transcript_utf8.txt \
  --jmdict data/jmdict/JMdict_e.gz \
  --out data/cache/jsut_full_v1/kana2kanji.json

# 4. 阶段 18 波形级增强（可选）：MUSAN 11GB + RIRS 1.5GB
bash scripts/download_musan.sh
```

生成物：`<utt>.npy`（fbank + CMVN 烘焙）、`char_vocab.json`、`mora_vocab.json`、`mean.npy`、`std.npy`、`train/val/test/probe.json`。

### 训练

```bash
python otokoenet/train.py --config configs/jsut_full_stage14.yaml   # 生产基线配方
# 常用：--resume 断点续训 / --fresh 重来 / --device auto（CUDA 优先）
# 长任务守护：bash scripts/train_tmux.sh <config>（缓存幂等生成 + 崩溃自动重启）
```

训练产出到 `runs/<save_dir>/`：`best.pt`（含 `model`/`ema`/`config`）、`last.pt`、滚动快照、`log.csv`。

### 评测与回归

```bash
python scripts/infer.py data/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav   # 单音频识别
python scripts/eval_recognize.py --ckpt runs/jsut_full_stage14/best.pt      # test CER（closed/open 分路）
python scripts/eval_engine_e2e.py                                            # 走后端 Engine 的端到端
python scripts/eval_pronun.py                                                # 发音指标
# 回归测试（独立脚本）
python scripts/test_engine_load.py   # 7 项验收 + 延迟基准
python scripts/test_align.py
python scripts/test_mask_invariance.py
python scripts/test_joint_decode.py
```

## 评估指标与验收

| 指标 | 目标 | 现状（JSUT test） |
| --- | --- | --- |
| 假名识别 CER | < 12% | **6.93% ✅** |
| 汉字识别 CER | — | ~27.6%（阶段 19 联合解码后；此前 32.02%） |
| mora MER | — | ~6.5%（beam=24, lm=0.2） |
| 文本-音频一致性判对（域内） | — | 99.75% |
| mora 定位 | — | 94.3% |
| 推理延迟 | < 3s | p95 1.33s ✅ |
| 发音打分 vs 人工 Pearson | 高 | **未验证**（`eval_pronun.py --human-scores` 有接口，缺人工评分数据；域外判对率 89.5%） |
| 長音规则 | 低假阳 | FPR 0~3%、召回 55~88% |

**非功能需求**：响应时间 <3s（达标）；开发阶段单用户，架构预留多用户并发；前端兼容 Chrome/Edge/Firefox。

**开发路线与实验记录**：仓库通过 `log/`（本地账本，不上 Git）分 **19 个阶段**记录数据准备 → 训练 → 端到端验收 → 消融的完整过程；`log/README.md` 是阶段索引，`log/formula.md` 是底层公式与文件组织速查。

## GitHub 发布指引

### 该推什么 / 不该推什么

仓库 = **模型源码 + 无状态后端**（无前端、无数据库、无鉴权）。

| 类别 | 内容 | 建议 |
| --- | --- | --- |
| 源码 | `otokoenet/`、`scripts/`、`backend/`、`configs/` | ✅ 推 |
| 依赖 | `requirements.txt`、`backend/requirements.txt` | ✅ 推 |
| 文档 | `README.md`、`.gitignore` | ✅ 推 |
| **数据目录** | `data/`（JSUT 语料、特征缓存、词典、MUSAN/RIRS，约十余 GB） | ❌ **不推**（已在 `.gitignore`，部署时重新下载生成） |
| 训练产物 | `runs/`（含 `jsut_full_stage14/best.pt` 约 75MB） | ❌ 不推；如需分发模型用 Git LFS 或网盘 |
| 实验日志 | `log/` | ❌ 不推（已在 `.gitignore`） |
| 环境/杂项 | `.venv/`、`__pycache__/`、`.idea/`、`backup_*/` | ❌ 不推（已在 `.gitignore`） |

### 推送步骤

```bash
# 1.（可选）分发模型权重时用 LFS
git lfs install
git lfs track "runs/jsut_full_stage14/best.pt"

# 2. 提交
git add .
git status          # 确认 data/、runs/、log/ 等未被加入
git commit -m "..."

# 3. 关联远端并推送
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git branch -M main
git push -u origin main
```

> 若发现 `data/`、`runs/`、`log/` 被 add，先检查 `.gitignore` 再 `git rm -r --cached <路径>`。

## 参考

- JSUT corpus：https://sites.google.com/site/shinnosuketakamichi/publication/jsut
- JMdict / JMnedict：https://www.edrdg.org/jmdict/