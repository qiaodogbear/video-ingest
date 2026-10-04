# 导出 B 站 cookie（Netscape 格式）

用途：让 `probe` 能区分"平台侧确实没有字幕"与"有字幕但需登录才可见"。

**凭证安全**：导出的文件等同你的登录态。放在笔记库和代码仓库**之外**，加入 `.gitignore`，不要贴进聊天、日志或提示词。

---

## 方法一：浏览器扩展导出（推荐，最省事）

1. 在 Chrome/Edge 应用商店装一个能导出 Netscape 格式的扩展，例如 **Get cookies.txt LOCALLY**（纯本地，不上传）。
2. 浏览器登录 B 站。
3. 打开 `https://www.bilibili.com`。
4. 点扩展图标 → 导出/Export → 保存为 `cookies.txt`。
5. 放到**仓库之外**的任意位置，例如 `~/.secrets/bilibili-cookies.txt`。

用之前先确认文件头是一行注释：

```
# Netscape HTTP Cookie File
```

---

## 方法二：用 yt-dlp 直接导出（无需装扩展）

**前提：必须完全关闭 Chrome**（否则 cookie 数据库被锁，报
`Could not copy Chrome cookie database`，见 yt-dlp#10927）。

```powershell
# 1. 完全退出 Chrome（确认没有残留进程）
Get-Process chrome -ErrorAction SilentlyContinue | Stop-Process

# 2. 导出到仓库外的位置（下文用 $cookie 代表该路径）
$py = ".\.venv\Scripts\python.exe"          # 项目内解释器
$cookie = "$HOME\.secrets\bilibili-cookies.txt"
New-Item -ItemType Directory -Force (Split-Path $cookie) | Out-Null

& $py -m yt_dlp --ignore-config --cookies-from-browser chrome `
    --cookies $cookie `
    --skip-download --simulate "https://www.bilibili.com/video/BV1qNYC6eEMj/"

# 3. 确认导出成功
Get-Content $cookie -TotalCount 1
```

第三行命令应输出 `# Netscape HTTP Cookie File`。要 Edge 就把 `chrome` 换成 `edge`。

> 本次尝试时 Chrome 与 Edge 都在运行（各十几个进程），导出被锁失败。这是**预期行为**，不是工具缺陷。**不要**用关闭浏览器加密、获取管理员权限或混用 WSL 去绕过——按方法一或下面的方法三做。

---

## 方法三：DevTools 手工取值

1. 浏览器登录 B 站，F12 → Application/应用 → Cookies → `https://www.bilibili.com`。
2. 记下 `SESSDATA`、`bili_jct`、`buvid3`、`DedeUserID` 的值。
3. 按 Netscape 格式自己写一个文件（**制表符分隔**，7 列）：

```
# Netscape HTTP Cookie File
.bilibili.com	TRUE	/	FALSE	<过期unix时间戳>	SESSDATA	<值>
.bilibili.com	TRUE	/	FALSE	<过期unix时间戳>	bili_jct	<值>
.bilibili.com	TRUE	/	FALSE	<过期unix时间戳>	buvid3	<值>
.bilibili.com	TRUE	/	FALSE	<过期unix时间戳>	DedeUserID	<值>
```

---

## 使用方法

```powershell
# 在项目根目录执行；$cookie 指向你导出的文件
& .\.venv\Scripts\python.exe -m video_ingest probe --url "<URL>" --cookies "$HOME\.secrets\bilibili-cookies.txt"
```

带 cookie 后：

- 有字幕 → `probe_state = found`，ingest 会下载字幕、**跳过 ASR**（省几分钟且无识别误差）
- 仍为空 → `probe_state = empty`，此时才能判定"平台侧无原语字幕"，走 ASR

## 有效期与失效

`SESSDATA` 有效期通常以月计。若突然出现 `needs_login` 或探测报权限错误，重新导出即可。

**不要把 cookie 文件放进项目的 `output\` 目录**——该目录是任务产物目录，容易被整体打包或分享。
