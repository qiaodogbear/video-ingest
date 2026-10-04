# 交付与验收报告

日期：2026-10（本机实测）
对应任务书：`local_video_agent_bootstrap.md`

**重要区分**：本报告严格区分「已验证」「已实现但未实网验证」「仅设计」三类。
不得把可执行示例、第三方宣称或单元测试通过，写成"某个视频已成功转录"。

---

## 1. 交付物

| 交付物 | 路径 | 状态 |
|---|---|---|
| CLI 工具 | `video_ingest/`（本仓库内） | 已验证 |
| 命令行入口 | `video_ingest/cli.py`，控制台脚本 `video-ingest` | 已验证 |
| 单元测试 | `tests/`（157 项） | 已验证（`157 passed`） |
| 静态检查 | `ruff --select F,E9,B,C4,RET` | 已验证（全绿） |
| SOP 文档 | [SOP.md](SOP.md) | 已验证 |
| Skill | `skill/video-ingest/` | 已验证（官方校验器通过） |
| 锁定清单 | `requirements.lock.txt` | 已验证 |
| 示例校正表 | `corrections.example.json` | 已验证 |
| CI | `.github/workflows/tests.yml` | 已编写（未在 GitHub 上运行过） |

## 2. 八条 CLI 命令

```bash
video-ingest doctor                                            # 环境自检
video-ingest probe   --url "<URL>" --save-job                  # 只探测
video-ingest ingest  --url "<URL>" --model medium              # 取材+转写
video-ingest ingest  --media "<本地文件>" --language zh
video-ingest validate --job-dir "<job>"                        # 校验
video-ingest chunks   --job-dir "<job>"                        # 分块
video-ingest frames   --job-dir "<job>" --video "<视频>"        # 画面帧
video-ingest ocr      --job-dir "<job>" [--search "<词>"]       # 画面文字
video-ingest batch    --url "<URL>" --pages "1-5"              # 多分P
```

退出码：`0` 成功 / `2` 用法错误 / `3` 被阻断 / `4` 失败。

---

## 3. 已验证（实网）

### 3.1 无字幕视频的完整 ASR 路径

对象：`BV1qNYC6eEMj`（407s，1 个分P）

| 指标 | 实测值 |
|---|---|
| 字幕探测（无 cookie） | `needs_auth_check`（正确，未误判为"没有字幕"） |
| `danmaku` 剔除 | `danmaku_only = true`（正确） |
| 音频下载 | `audio.m4a` 3.8 MB |
| 转写 | `medium` + int8 CPU，71 个片段，327–346s |
| 解码时长 vs 声称时长 | 406.08s vs 407s，偏差 **0.23%** |
| `processing_status` | `resource_processed` |
| `full_video_coverage` | `supported`（依据已记录） |
| `verification_status` | `unreviewed`（未抽样比对，不声称准确） |
| 分块覆盖 | `complete: true`，`missing: []` |
| 校正分层 | 原文保留 `素查表`/`费慢`，校正稿为 `速查表`/`费曼` |

### 3.2 本地媒体路径

对象：合成的 3 秒 16k 单声道 wav（纯音 + 静音）

- 解码、重采样、转写全链路跑通（`tiny` 模型）
- 产出 0 片段 → 如实记为 `processing_status=failed`、`full_video_coverage=uncertain`、`ok=false`、退出码 4
- 这是**正确行为**：跑完流程不等于拿到内容

### 3.3 画面帧与字幕配对（`frames`）

对象：同一视频，1080p，407s

| 指标 | 实测值 |
|---|---|
| 采样点 | 1016（步长 0.4s，逐帧解码） |
| 画面曝光段 | 85 |
| 提取帧 | 70（`frames_missing = 0`） |
| 帧被多片段共享 | 17 帧，最多被 **6 条字幕**共用一帧 |
| 每片段帧数 | 1 帧 × 45，2 帧 × 22，3 帧 × 3，4 帧 × 1 |
| 帧状态 | `ok` 74，`mid_transition` 28 |
| 孤儿曝光段（无字幕引用） | 15 |
| 人工核对 | `frame-run-0004`、`frame-run-0022` 等帧文字与图表清晰可读 |

**关键实测结论**：

| 发现 | 数值 | 影响 |
|---|---|---|
| 按 `cue.start` 取帧落在画面变化 1s 内 | **83%** | 单点取帧不可用，取到的基本是转场空画面 |
| 画面变化后稳定所需 | 中位 1.18s，最大 1.44s | 固定偏移是对单个视频的拟合，换视频失效 |
| 一个 cue 跨越多个曝光段 | 约 10% | 必须区间求交，否则漏画面 |
| 最长被标 `mid_transition` 的曝光段 | 0.4s | 标记针对的是段太短，不是帧质量差 |

### 3.4 多分P批处理（`batch`）

对象：`BV1Gf4y1y7wc`（996 个分P），处理 p1–p2

| 项 | 实测 |
|---|---|
| 结果 | `ok: 2`，两P均 `resource_processed`，`supported` |
| 分P隔离 | p1 cid=281031471 / p2 cid=281031531，产物目录独立，未串P |
| 失败续跑 | 首次因模型缓存 ACL 失败 → **两页都记为 failed 但仍继续处理**，退出码 1（正确行为） |
| 复用 | 重跑同样范围：`reused: 2`，耗时 **0.4s**（首次约 3–4 分钟） |

**守卫项实测**（全部拒绝，退出码非 0）：

| 输入 | 结果 |
|---|---|
| `--pages " "` | 拒绝："必须显式指定分P范围…不提供'全部'默认值" |
| 单P视频请求 `--pages 1-5` | 拒绝："该视频只有 1 个分P，但请求了: 2, 3, 4, 5" |
| `--pages 9-3` | 拒绝："区间上下界颠倒" |

### 3.5 画面文字 OCR 与三方绑定（`ocr`）

对象：同一 407s 视频的 70 帧

| 项 | 实测 |
|---|---|
| 引擎 | rapidocr-onnxruntime |
| 帧数 / 耗时 | 70 帧 / 121.4s（**1.73s/帧**） |
| 反向检索词条 | 10950 |
| 口播/画面分歧帧 | **7**（阈值 0.05） |
| 文字可读性 | 幻灯片正文、阶段标题、编号均可读出 |

**反向检索实测**（这些词**口播里没有**，只能从画面得到）：

| 查询 | 命中帧 | 涉及片段时间 |
|---|---|---|
| 十四课 | 4 | 197.8–212.5s |
| 复利 | 4+ | 31.7–42.4s |
| 费曼 | 4+ | 247.0–265.8s |
| 去掉形式 | 4+ | 333.7–339.1s |

**分歧检测样例**：42s 处口播"我们从最反直觉的一笔账讲起"，画面却是
"差距=指数级 / 干活效率·线性 / 学习速度：复利"；201s 处口播讲"第五步
资源筛选"，画面是"05/阶段二 聚焦核心：两成核心，十四课计划"。后者口播
**一个字都没提**。

### 3.6 语言校验（`--language auto` 与不匹配告警）

同一段英文音频（TED-Ed p2，240s）实测：

| 传入 language | `info.language` | 实际产出 |
|---|---|---|
| `None` | **en** (0.907) | "We've all seen movies about terrible insects…" ✅ |
| `en` | en (1.000) | 正确英文 ✅ |
| `zh` | **zh (1.000)** | "我認為我們可以講到孫宏的驚喜中國文化…" ❌ **完全捏造** |

**关键结论**：强制指定语言后 `info.language` 只是**回显**传入值，prob 还变成
1.000，因此"比对 transcribe 返回的检测语言"完全无效。本工具改为**单独跑一次
检测**（开头 30s、beam_size=1）。

修正后实测（强制 `zh` 跑英文音频）：

```text
[ingest] language probe: en (prob=0.941)
asr.language_mismatch  : True
verification_status    : issues_found
note: 警示：指定语言 zh，但 ASR 检测到语音为 en…识别的**内容**不可信
```

改用 `--language auto` 后同一音频产出正确英文全文。

### 3.7 分享可用性

| 项 | 实测 |
|---|---|
| 解压后在独立目录、独立 venv 安装 | 通过 |
| 从项目目录外执行 `doctor` | `ok: true`（不再依赖 cwd） |
| 接收者跑测试 | `157 passed` |
| 打包安全扫描 | 凭证值 / 本机绝对路径 / 多媒体 三项全绿 |

### 3.8 单元测试覆盖的关键约束

- 三态判定（`found` / `needs_auth_check` / `empty`）四组用例
- `danmaku` 不得当字幕；只有翻译字幕不得当原语字幕
- `--list-subs` 表头解析回归用例（锁死曾出现的顺序 bug）
- 覆盖率必须完整、片段不被拆块、重叠不污染主覆盖
- 空片段 / 负时间 / `end < start` / 超时长 / 合法重叠的处理
- 原文不被校正覆盖；最长键优先替换
- URL 去追踪参数但保留 `p`；拒绝非 http、私网、未知站点
- 凭证与签名 URL 脱敏
- manifest 一致性（非法枚举、0 片段声称成功、无依据的 `supported`、来源矛盾）
- 分页表达式解析与三个守卫；产物复用只认 `resource_processed`
- 画面曝光段分割、区间求交、取样点落在段内、M:N 去重、孤儿段报告
- 帧状态判定（短段按段时长、`blank` 需无结构、`high_change`）
- OCR 文本归一化、n-gram 覆盖率、候选词抽取、反向检索、分歧标记
- 语言不匹配的检出与状态升级；`auto` 不触发；区域变体不误报

---

## 4. 已实现但**未实网验证**

| 项 | 原因 | 现有保障 |
|---|---|---|
| 带 cookie 的字幕下载正向路径 | 开发机 Chrome 与 Edge 均在运行，cookie 库被锁（yt-dlp#10927），未取得凭证 | 代码路径已实现；SRT/VTT/JSON 解析、字幕键优先级、字幕路线分块与覆盖均有单元测试 |
| GPU 转写（`--device cuda`） | 已检测到 RTX 2070 SUPER，但未做 CUDA 路径实测 | 参数已透传；CPU 路径已验证 |
| GitHub Actions CI | 尚未推送到 GitHub | 工作流已编写，测试命令本地已验证一致 |
| 非 Windows 平台 | 开发环境为 Windows 11 | 代码尽量跨平台（`sys.executable`、`pathlib`），但未实测 |

## 5. 仅设计 / 不在范围内

- 说话人分离（未实现；单人讲解场景无收益，且需引入 torch 级依赖）
- **画面语义理解**：`frames` 只取帧、`ocr` 只取文字，都不解释画面
- 字幕轨与画面帧的 OCR 交叉校验（当前靠 Agent 目视）
- 弹幕内容分析
- 商业 API 适配（如 BibiGPT）

**画面部分的能力边界**：

| 项 | 状态 |
|---|---|
| 画面曝光段分割 / 区间求交配对 | 已验证 |
| 帧提取与状态标记 | 已验证 |
| 画面文字 OCR 与反向检索 | 已验证 |
| 口播/画面分歧检测 | 已验证（低重合判据，非语义判断） |
| 画面**语义**理解 | 不在本工具范围（交给 Agent） |
| 逐帧 OCR / 全视频帧扫描 | 不做；< 0.4s 的闪屏可能漏检 |

---

## 6. 开发过程中发现并修复的真实缺陷

这些是实测暴露的，不是推演出来的：

| # | 缺陷 | 影响 | 处置 |
|---|---|---|---|
| 1 | `--list-subs` 表头行以 `[info]` 开头，而"以 `[` 开头即跳过"的判断在前 | 永远进不了表格，`danmaku_only` 恒为 `false`，弹幕可能被误判 | 调整判断顺序 + 回归测试 |
| 2 | `yt_dlp.version` 读取错误（模块无 `__version__`） | 运行报告版本为空，破坏可复现性 | 改用 `importlib.metadata` |
| 3 | 分块重叠片段被重复计入主覆盖集合 | `coverage.complete` 误报 false | 区分"新增片段"与"重叠片段" |
| 4 | `new_manifest` 初始 `processing_status="failed"`，而成功判定写成 `not in (...)` | **成功任务被标为 failed** | 显式改判 |
| 5 | 0 片段仍标 `resource_processed` + `supported` | 会输出"看起来成功、实际无内容"的结论 | 如实记为 failed，`ok=false`，退出码 4 |
| 6 | `av 19.0.1` 与 `faster-whisper 1.2.1` 的 `metadata_errors` 不兼容 | 转写直接 `TypeError` | 自行用 PyAV 解码为 16k float32 数组传入模型 |
| 7 | HuggingFace Xet 通道导致模型下载卡 0 字节 | 首次转写无法开始 | `HF_HUB_DISABLE_XET=1` |
| 8 | `pip install "yt-dlp[default]" faster-whisper` 触发解析器死循环 | 1718s CPU 无进展 | 分两步安装并换镜像源 |
| 9 | `frames` 写入 manifest 时用 `man["visual"]`，旧 manifest 无该键 | `KeyError: 'visual'`，产物已生成但 manifest 未更新 | 改用 `setdefault` + 回归测试 |
| 10 | `frames` / `ocr` 使用 `sha256_file`、`add_note` 但未导入 | manifest 静默更新失败（仅告警），证据链断裂 | 补导入 |
| 11 | 取样点用"交集中点" | 同一张幻灯片被 N 条字幕提取 N 次；同一画面多个 frame_id | 改为**曝光段中点**，保证 1 帧 ↔ 1 视觉状态 |
| 12 | 用"距边界距离/段长"判断短段中转场 | 取样点在段中点，该比值恒 0.5，判据永不触发（70 帧全标 ok） | 改为按**段时长**判断（<0.6s） |
| 13 | `blank` 判据只看亮度 | 该视频 0 帧被判空白——黑底白字幻灯片亮度同样很低 | 要求**亮度极端 + 完全无结构**同时成立 |
| 14 | 包未安装，只能在本目录运行 | 接收者从别处执行报 `No module named video_ingest` | 文档要求 `pip install -e .`；已实测 |
| 15 | Skill 内硬编码了开发者本机的绝对路径（7 处） | 接收者机器上完全不可用 | 改为 `$TOOL` 变量 + 明确前提说明 |
| 16 | 无 README，外部读者不知从何入手 | 无法独立安装使用 | 新增 README.md |
| 17 | `doctor` 不报告 pytest，而 README 让用户跑测试 | 缺 pytest 时 `doctor` 仍报 ok，误导 | 新增 `test_runner` 检查与告警（不影响取材） |
| 18 | 安全扫描把"提到凭证名"误判为"含凭证值" | 大量误报，告警会被忽略 | 改为**名字+值**配比 + **熵 + 大小写数字混排**；已双向验证 |
| 19 | 语言校验用 `transcribe` 返回的检测语言 | **实测该值在强制指定语言时只是回显**，校验恒为通过，等于没有 | 改为**单独跑一次语言检测**再比对 |
| 20 | 强制语言与音频不符时无任何警示 | 英文音频用 `zh` 产出流畅但**完全捏造**的中文，看起来像正常结果 | 记 `language_mismatch` + 警示 + 升级 `issues_found`，并支持 `--language auto` |
| 21 | 语言检测与转写各加载一次 whisper 模型 | medium 模型白加载两次（各数秒、数百 MB 内存） | 加进程内模型缓存（实测二次加载 1.05s → 0.0000s） |
| 22 | `__main__.py` 膨胀到 948 行 | 可维护性差 | 实现移入 `cli.py`，`__main__.py` 变 24 行薄转发并保留兼容导入 |
| 23 | 用 PowerShell 5.1 的 `Set-Content`/`Get-Content -Raw` 改文档 | **静默破坏 UTF-8**（按 GBK 解码、写入带 BOM），README 与本文档曾损坏 | 改为只用支持 UTF-8 的写入方式；加全仓库 UTF-8 合法性扫描 |

其中 #4、#5 是**状态机类缺陷**：工具"跑成功了"却报告失败，或"什么都没拿到"却报告成功。两者都会直接误导下游总结。

#11–#13 是**画面配对类缺陷**：它们不会报错，只会安静地产生错误或低质结果（重复提取、标记失效、该跳过的不跳过）。

#19、#20 是**静默内容捏造类缺陷**：不报错、输出流畅、看起来完全正常，而内容是虚构的。这是最危险的一类。

#23 是**文档编码类缺陷**：不报错，但文件已损坏，且只有推送后才会被发现。

## 7. 环境改动（供用户决定是否清理）

| 项 | 位置 | 体积 |
|---|---|---|
| 项目 `.venv` | `video-research-agent/.venv` | 约 450 MB |
| 依赖 | yt-dlp、faster-whisper、ctranslate2、av、onnxruntime、opencv、pillow、numpy、pytest、ruff | — |
| HF 模型缓存（用户级） | `~/.cache/huggingface/hub` | 约 2.9 GB |
| HF 模型缓存（项目内，隔离） | `video-research-agent/.hf-cache` | 约 80 MB |
| 任务产物 | `video-research-agent/output/` | 含下载的音视频与帧 |

清理方式：删除 `output/`、`.hf-cache/`；用户级模型缓存可保留（下次免下载），
需要回收空间时删除即可。

## 8. 结论

任务书要求的阶段 A–E 均已完成实现与验收：

- **阶段 A（环境与探测）**：完成，实网探测真实 BV 链接，失败状态可辨
- **阶段 B（字幕全文）**：解析、导出、manifest 与测试完成；**下载环节缺实网验证**（缺 cookie）
- **阶段 C（本地 ASR）**：完成，实测 407s 视频全链路；含语言不符检测
- **阶段 D（Agent 全文解析）**：确定性分块、覆盖检查、Skill、输出模板完成
- **阶段 E（固化）**：版本锁定、缓存策略、Skill 官方校验通过、CI 与打包脚本就绪

后续扩展（多分P批处理、画面帧配对、OCR 三方绑定）亦已完成并实网验收。

唯一遗留的实网空白是**带 cookie 的字幕下载**，其前置障碍（cookie 库被占用）
已定位并给出了不违反任务书 §7.2 的导出步骤。
