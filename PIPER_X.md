# Piper-X 云端推理服务

本地 `piper-x` client 读取相机、双臂状态和任务指令，经 SSH 隧道发送到云端 WebSocket。云端完成图像拼接、状态归一化、模型推理和动作反归一化，返回绝对末端目标。

## 代码分工

| 文件 | 职责 |
|---|---|
| `openwam/deploy/piper_x.py` | Piper-X checkpoint、EEF20 输入输出校验及握手约定 |
| `scripts/serve_piper_x.py` | 服务入口，复用 OpenWAM 参数解析、模型加载和推理 |
| `scripts/start_piper_x_ws.sh` | 后台启动、重复启动检查、日志与 PID 管理 |
| `scripts/check_piper_x_ws.py` | 连接真实服务；默认检查握手，可选合成观测推理 |

服务端不添加指令前缀、不交换颜色通道、不补偿 TCP。上述输入约定由 client 按训练数据准备；安装差异由硬件对齐处理。

## 启动与连接

云端仓库：`/vePFS/tangyuting/OpenWAM-piper`。以下命令在云端执行：

```bash
cd /vePFS/tangyuting/OpenWAM-piper
bash scripts/start_piper_x_ws.sh
tail -f logs/piper_x_ws/server.log
```

默认使用环境 `/vePFS/tangyuting/envs/openwam`，在 `cuda:0` 加载权重，执行10步同步去噪，按完整动作块返回。服务仅监听 `127.0.0.1:8848`。启动脚本返回代表开始加载，握手成功才代表服务就绪；重复执行不会重启已记录的进程。

在本地建立隧道并保持终端开启：

```bash
ssh -p 62507 -N -L 18848:127.0.0.1:8848 root@115.190.155.12
```

本地 client 连接 `ws://127.0.0.1:18848`。若已使用本地 `openwam_tunnel.sh` 将端口转发到8848，则继续使用原有的 `ws://127.0.0.1:8848`，无需再建隧道。

## 输入输出约定

| 项目 | 约定 |
|---|---|
| 状态与动作 | EEF20，左臂在前，右臂在后；每臂为位置3维、旋转6维、夹爪1维 |
| 位置 | 米，各臂自身基座坐标系；保持录制数据的末端定义 |
| 旋转 | 旋转矩阵前两列拼接：`[R00,R10,R20,R01,R11,R21]` |
| 夹爪 | 0闭、1开；client 按实际行程将宽度换算为比例 |
| 归一化 | client 发送原始物理量；云端按 checkpoint 的 min/max 归一化，映射到80维槽位0–9、34–43 |
| 返回动作 | 云端逆映射并反归一化，当前权重每次返回 `32×20`；表示绝对目标位姿 |
| 时间 | 32步对应连续采样点 `state[t+1:t+33]`；`video_stride=4` 只抽视频，动作采样率待确认 |

三路图像使用无损 PNG 的 base64，字段依次为 `head_camera`、`left_wrist_camera`、`right_wrist_camera`，对应训练相机 `cam_head`、`cam_left_wrist`、`cam_right_wrist`。当前权重使用三视图，云端拼接为宽320×高384。协议允许缺失腕部图像并填黑，正式回放应提供完整三视图。

请求结构（`state` 须替换为实际20个数值）：

```json
{
  "type": "obs",
  "images": {
    "head_camera": "<PNG的base64>",
    "left_wrist_camera": "<PNG的base64>",
    "right_wrist_camera": "<PNG的base64>"
  },
  "prompt": "A video recorded from a robot's point of view executing the following instruction: pick",
  "state": ["实际EEF20数值"]
}
```

client 加训练指令前缀一次，云端原样使用。`pick` 是本次 bag 的指令，不属于当前公开权重列出的6类训练任务，模板一致不代表任务分布一致。

`{"type":"ping"}` 返回 `pong`，包含表示、维数、动作块长度和归一化约定。`obs` 返回 `type=action`，附带 `action`、`num_actions`、`action_dim`、`step`、`latency_ms`。`{"type":"reset"}` 清空共享策略状态并返回 `reset_ack`。状态宽度错误、非有限值、退化旋转和夹爪越界均拒绝处理。

## 当前权重与后续重训

当前目录为 `/vePFS/data/openwam/OpenWAM-Alpha-Real-RoboDojo-Piper-X`，权重文件为 `checkpoint_step_30000.safetensors`，配置和归一化统计来自同一目录。

**当前公开旧权重要求 BGR 数值输入**：client 将 RGB 的红蓝通道交换一次，再装入标准 RGB PNG；云端不重复交换。旧权重发布早于 [上游颜色修复](https://github.com/OpenWAM-Official/OpenWAM/commit/519ec054373992e507bec8a574f66cc828dc7cc5)。后续使用修复后的 reader 重训，应将 client 改为 RGB，并随模型记录颜色、末端、夹爪行程、相机布局和动作采样率。

启动脚本支持环境变量 `PIPER_MODEL`、`PIPER_CKPT`、`PIPER_ENV`、`PIPER_DEVICE`、`PIPER_HOST`、`PIPER_PORT`、`PIPER_LOG`。例如新模型启动前设置：

```bash
PIPER_MODEL=/实际的新模型目录 PIPER_CKPT=checkpoint_step_30000.safetensors \
  bash scripts/start_piper_x_ws.sh
```

这不会替换已运行的模型；切换前需确认旧进程已停止。新模型仍须满足本页 EEF20、80维映射和 min/max 约定，否则需同步更新适配模块。权重来源记录在模型目录的 `deployment_assets.json`。

## 检查

在云端执行，默认只检查握手，不推理、不重置：

```bash
/vePFS/tangyuting/envs/openwam/bin/python scripts/check_piper_x_ws.py \
  --output logs/piper_x_ws/handshake.json
```

增加 `--infer` 会检查错误维数拒绝、发送两条合成观测并验证动作形状和数值，同时重置共享策略状态；应在没有其他 client 使用服务时运行。它调用真实模型，灰色合成图像仅用于协议检查，不能评估策略效果。

此前真实 bag 完整回放：249条完整观测全部成功返回 `32×20`，共7,968步。输入与训练 reader 的表示、归一化和图像路径已核验，但预测轨迹仍有异常；未验证闭环任务成功率。记录的末端与官方示例存在安装差异，主代码未加偏移；独立临时偏移实验改善部分位置偏差，未改善动作块内部跳动。视频展示帧率不作为动作执行频率。
