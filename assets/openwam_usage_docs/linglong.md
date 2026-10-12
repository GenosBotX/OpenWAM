# Linglong 训练、评测与云端推理

本文描述 Linglong 的数据输入、训练适配、离线动作评测、视频生成和云端 WebSocket 服务。命令按本工作区的路径编写。

## 1. 运行环境和目录

训练、评测和云端推理使用同一套 OpenWAM 环境：

| 项目 | 路径 |
| --- | --- |
| 仓库 | `/iag_ad_01/ad/xiejinxia/code/test/OpenWAM` |
| Python | `/iag_ad_01/ad/xiejinxia/envs/dreamzero/bin/python` |
| 合并训练集 | `data/linglong_0923_1010_openwam_train_15hz` |
| 合并测试集 | `data/linglong_0923_1010_openwam_test_15hz` |
| 训练配置 | [configs/train_linglong_0923_1010.yaml](../../configs/train_linglong_0923_1010.yaml) |
| 训练输出根目录 | `outputs/linglong_0923_1010_32gpu_official_finetune` |

以下评测和部署示例使用已有 run `2026-10-12_01-59-42`。评测脚本自动选择该目录中 step 编号最大的权重；示例服务启动命令显式指定 `checkpoint_step_400.safetensors`。评测其他训练结果时，将 checkpoint 目录换成目标 run。

Python 依赖声明在 [pyproject.toml](../../pyproject.toml)。离线推理需要 CUDA、PyTorch 和模型资源；数据读取使用 PyArrow、Pillow、NumPy；视频导出使用 imageio/ffmpeg；WebSocket 服务使用 websockets。以下命令直接调用已配置的环境，不修改其依赖。

## 2. 数据与训练输入

### 2.1 24 维状态和动作

Linglong 使用关节空间表示：

- `observation.state`：24 维测量关节位置 `q`。
- `action`：24 维绝对目标关节位置 `q_exp`。
- 三路图像：头部、左腕、右腕。
- prompt：episode 对应的任务文本。

索引从 0 开始，区间采用左闭右开的 Python 切片写法。

| 原始维度 | 语义 | 合并训练配置中的 80 维目标槽位 |
| --- | --- | --- |
| `0:7` | 左臂 7 关节 | `0:7` |
| `7` | 左夹爪 | `9` |
| `8:15` | 右臂 7 关节 | `34:41` |
| `15` | 右夹爪 | `43` |
| `16:20` | 腰部 4 关节 | `68:72` |
| `20:22` | 头部 2 关节 | `72:74` |
| `22:24` | 底盘占位 | `74:76`，无效维度 |

转换和读取保留采集数据的数值单位、零位及关节排列。机器人客户端应按采集端相同的语义提供测量值并解释目标值。

### 2.2 数据格式、采样和 prompt

[scripts/convert_linglong.py](../../scripts/convert_linglong.py) 读取 processed 数据中的 `inventory.json`、`aligned.csv` 和 `alignment.jsonl`，按头部时间戳在 15 Hz 网格上选择最近记录，输出 episode parquet、图像路径、元数据和归一化统计。该脚本的固定 prompt 包含 blue block；多颜色采集应在 episode 元数据中使用对应任务描述。

当前合并数据的 manifest 记录：

| 数据集 | episode 数 | 原始帧数 |
| --- | --- | --- |
| 训练集 | 184 | 36,281 |
| 测试集 | 20 | 4,489 |

图像通过 `images/0923`、`images/1010` 链接访问 decoded 源目录，运行时需要保留这些链接及其目标。parquet 和图像路径共同组成完整数据集。

读取器优先使用 `meta/episodes.jsonl` 中的 `tasks[0]`，缺失时使用 manifest 的 prompt。合并数据包含：

```text
0923:
Use the right arm to pick up the blue block from the table and place it in the box.

1010:
Use the right arm to pick up the block from the table and place it in the box.
```

### 2.3 时间窗口与图像布局

[LinglongDataset](../../openwam/dataloader/linglong.py) 使用：

| 配置 | 数值和含义 |
| --- | --- |
| `num_frames` | 33 个原始时刻 |
| 动作监督 | 窗口前 32 行的 `q_exp`，形状为 `(32, 24)` |
| proprio | 窗口第一行的测量 `q` |
| `video_stride` | 4，选取相对帧 `0,4,8,...,32`，共 9 帧 |
| `window_stride` | 训练时为 1 |
| 时间尺度 | 动作 15 Hz，采样视频 3.75 Hz |

三路 RGB 图像使用 BILINEAR resize 后拼接成高 384、宽 320 的画布：上方头部图像为 `256×320`，下方左右腕图像各为 `128×160`，尺寸按高×宽表示。训练和服务端均使用 [multiview.py](../../openwam/dataloader/transforms/multiview.py) 的布局函数。

### 2.4 归一化、映射与有效维度

训练链路为：

```text
24 维 q / q_exp
  → 各自的训练集 min-max 统计
  → 24 维归一化结果
  → 按 unify_action_map 映射到 80 维
  → 有效维度掩码
  → 视频与动作联合训练
```

合并训练配置的映射为 `["0-6", 9, "34-40", 43, "68-75"]`。动作和状态使用相同槽位，归一化统计分别保存为 `joint` 和 `joint_state`。统计基于合并训练帧计算，测试集复用这份统计。

`active_raw_action_dims` 为 `0..21`。底盘占位在归一化后置零并从有效掩码中排除。推理由 [model_loader.py](../../openwam/deploy/model_loader.py) 和 [base.py](../../openwam/model/architectures/base.py) 处理有效维度，输出经过 80→24 维提取和反归一化，底盘两维置零。

映射、统计和图像配置以所加载 checkpoint 的 `config.yaml` 为准。`configs/dataloader/linglong.yaml` 是独立的数据配置入口；本文命令通过 `train_linglong_0923_1010` 完整配置启动。

## 3. 训练启动与 checkpoint

### 3.1 训练配置

初始化权重由 `training.finetune_ckpt_path` 指定：

```text
assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model
```

模型为 dual_system / joint_self_attn，使用 Wan2.2 TI2V-5B 视频骨干，state/action 维度均为 80。训练更新 ActionDiT、视频 DiT 及未冻结的联合模块；VAE、文本编码器和配置中列出的编码模块保持冻结。视频与动作损失权重均为 1。

| 参数 | 配置值 |
| --- | --- |
| GPU 数 | 4 节点 × 8 GPU = 32 |
| 每 GPU batch size | 1 |
| 梯度累积 | 24 |
| 完整累积周期的有效 batch size | `1 × 32 × 24 = 768` |
| 学习率 | `1e-4` |
| scheduler | cosine，warmup ratio 0.05 |
| epoch | 5 |
| 精度 / 分片 | bf16 / DeepSpeed ZeRO-2 |
| 梯度检查点 | 开启 |
| 保存间隔 / 保留数 | `save_steps=200` / `keep_last_k_ckpts=2` |
| 完整训练状态 | `save_full_states_for_resume=true` |

### 3.2 火山云入口命令

在配置了 4 个 worker、每 worker 8 GPU 的火山云任务中设置：

```bash
cd /iag_ad_01/ad/xiejinxia/code/test/OpenWAM && \
NPROC=8 EXPECTED_WORLD_SIZE=32 \
bash scripts/launch_volcano_linglong_0923_1010_32gpu.sh
```

[启动脚本](../../scripts/launch_volcano_linglong_0923_1010_32gpu.sh) 解析平台 worker 主机列表、节点 rank 和 master 地址，检查总 GPU 数，然后通过 torch.distributed.run 启动 [scripts/train.py](../../scripts/train.py)。脚本使用固定的合并数据配置，清除 `OPENWAM_LINGLONG_DATASET_DIR` 和 `OPENWAM_LINGLONG_SOURCE_ROOT` 环境覆盖。

任务模板为 [volcano_openwam_linglong_0923_1010_32gpu.yaml](../../volcano_openwam_linglong_0923_1010_32gpu.yaml)。镜像、队列和 GPU 规格在提交任务时按平台配置填写。

### 3.3 输出与恢复训练

每次训练生成时间戳 run 目录，包含以下文件：

```text
outputs/linglong_0923_1010_32gpu_official_finetune/<run>/
  config.yaml
  checkpoint_step_<N>.safetensors
  normalization_stats.npy
  tokenizer/
  accel_state_step_<N>/
    trainer_state.json
    ... Accelerate / DeepSpeed 状态文件 ...
```

完整状态由全部 rank 参与保存，在梯度累积完成的优化器边界写入；`trainer_state.json` 作为保存完成标记最后写入。它保存模型、优化器、调度器、随机数状态及训练进度。恢复时校验训练上下文和归一化统计，复用对应 run 目录。

对具备完整状态的目标 run，向启动脚本追加：

```text
training.resume_ckpt_path=/绝对路径/目标run目录
```

权重初始化使用 `training.finetune_ckpt_path`；完整进度恢复使用 `training.resume_ckpt_path`。后者要求存在可用的 `accel_state_step_*`。本文示例 run `2026-10-12_01-59-42` 当前包含推理权重，没有完整恢复状态。

## 4. 离线推理、动作评测与视频生成

离线脚本直接在 `cuda:0` 加载 checkpoint，通过 `JointInferenceEngine.generate()` 推理。每个窗口的输入为起始多视角图像、起始测量 q 和 prompt，输出为完整的 32 帧动作块；开启视频解码后同时获得 9 帧采样视频。

| 脚本 | 功能 |
| --- | --- |
| [eval_linglong_train.py](../../scripts/eval_linglong_train.py) | 指定训练 episode 的窗口推理、动作误差和视频输出 |
| [eval_linglong_test_all.py](../../scripts/eval_linglong_test_all.py) | 加载一次模型，评测测试集各 episode |
| [summarize_linglong_body_mae.py](../../scripts/summarize_linglong_body_mae.py) | 读取评测结果，按部位汇总 MAE |
| [prepare_linglong_foundation_eval.py](../../scripts/prepare_linglong_foundation_eval.py) | 配置官方 Foundation 权重的 Linglong 评测目录 |

在运行评测的终端先设置工作目录，并清除数据路径覆盖：

```bash
cd /iag_ad_01/ad/xiejinxia/code/test/OpenWAM
unset OPENWAM_LINGLONG_DATASET_DIR OPENWAM_LINGLONG_SOURCE_ROOT
```

### 4.1 单 episode 推理和视频

```bash
/iag_ad_01/ad/xiejinxia/envs/dreamzero/bin/python \
  scripts/eval_linglong_train.py \
  --checkpoint-dir outputs/linglong_0923_1010_32gpu_official_finetune/2026-10-12_01-59-42 \
  --dataset-dir data/linglong_0923_1010_openwam_train_15hz \
  --episode 0 --chunk-stride 32 --max-windows 1000 \
  --denoise-steps 10 --seed 42 \
  --decode-video \
  --output outputs/linglong_eval/train_episode_00/metrics.json
```

只运行一个窗口可设置 `--max-windows 1`。视频默认保存在输出 JSON 同级的 `video/`，也可通过 `--video-output` 指定目录。

### 4.2 全测试集推理和视频

```bash
/iag_ad_01/ad/xiejinxia/envs/dreamzero/bin/python \
  scripts/eval_linglong_test_all.py \
  --checkpoint-dir outputs/linglong_0923_1010_32gpu_official_finetune/2026-10-12_01-59-42 \
  --dataset-dir data/linglong_0923_1010_openwam_test_15hz \
  --chunk-stride 32 --max-windows 1000 \
  --denoise-steps 10 --seed 42 \
  --decode-video \
  --output-dir outputs/linglong_eval/test_all
```

`--max-windows` 是每个 episode 的窗口数量上限。只评测完整的 33 帧窗口，尾部未覆盖帧数写入视频 manifest。取消 `--decode-video` 可只保存动作评测结果。

```text
outputs/linglong_eval/test_all/
  aggregate_metrics.json
  episode_00/
    metrics.json
    metrics.npz
    video/
      generated_all_chunks.mp4
      gt_vs_generated_all_chunks.mp4
      manifest.json
  ...
```

`metrics.npz` 包含 `predictions`、`targets` 和 `starts`，动作数组形状为 `(窗口数, 32, 24)`。

视频为 3.75 fps 的多视角画面；并排视频左侧是真实采样帧、右侧是模型生成帧。步长 32 时，相邻块共享的边界视频帧会去重。每个块使用该处真实起始观测重新条件化，视频反映这种分块生成过程。

### 4.3 指标与部位统计

动作指标包括有效维度 `0..21` 的 MAE、RMSE、最大绝对误差和逐维 MAE。单 episode 脚本还输出平均/P50/P95 耗时，全测试脚本输出各 episode 平均耗时。两个脚本关闭 compile；开启视频时耗时包含视频解码，单 episode 脚本的计时还包含视频拼接/写入，耗时口径应随评测参数一同记录。

所有误差保留原始动作单位。夹爪和关节数值汇总时应同时参考逐维、分部位指标。分部位统计命令：

```bash
/iag_ad_01/ad/xiejinxia/envs/dreamzero/bin/python \
  scripts/summarize_linglong_body_mae.py \
  outputs/linglong_eval/test_all
```

输出为 `body_part_metrics/body_part_mae.json` 和 `body_part_metrics/body_part_mae.csv`，覆盖左臂、左夹爪、右臂、右夹爪、腰部和头部。各动作时刻等权，部位内各维 MAE 再取平均。

离线指标衡量记录观测下的动作预测误差；真机闭环任务成功率需要单独执行并统计。

### 4.4 Foundation 评测准备

`prepare_linglong_foundation_eval.py` 接受 `--foundation`、`--reference` 和 `--output`：引用 Foundation 权重与 tokenizer，使用 reference 的 Linglong 数据配置、归一化统计和本地模型资源路径，生成评测目录。输出目录须尚未存在，生成后交给上述评测脚本。

该工具当前固定引用 `checkpoint_step_154000.safetensors`，适用于具有该文件的 Foundation 包。权重通过符号链接引用，源目录需要保持可访问。

## 5. 开启云端 WebSocket 服务

### 5.1 启动命令

服务调用链为：

```text
scripts/deploy.sh → scripts/deploy.py → PolicyServer
  → checkpoint 配置、权重、归一化统计
  → 观测预处理 → WAMPolicy → 推理引擎
  → 24 维绝对关节目标
```

在云端终端执行以下命令，服务以前台方式运行：

```bash
cd /iag_ad_01/ad/xiejinxia/code/test/OpenWAM && \
PATH=/iag_ad_01/ad/xiejinxia/envs/dreamzero/bin:$PATH \
NUM_GPUS=1 \
bash scripts/deploy.sh \
  outputs/linglong_0923_1010_32gpu_official_finetune/2026-10-12_01-59-42 \
  --ckpt-name checkpoint_step_400.safetensors \
  --device cuda:0 \
  --host 127.0.0.1 --port 8848 \
  --denoise-steps 10 \
  --inference-mode sync \
  --inference-horizon 32 \
  --compile-enabled false
```

这里通过 PATH 选择 Python 环境，因为 `deploy.sh` 内部调用 `python`。checkpoint 目录须包含配置、权重、归一化统计和 tokenizer，并可访问配置所需的模型资源。

命令使用本机监听地址，供 SSH 隧道接入；显式关闭 compile 以便初次联调。需要编译加速时设置 `--compile-enabled true`，并给首次请求预留编译时间。停止前台服务使用 Ctrl-C。

### 5.2 本地转发端口

在 Mac 上运行，将 SSH 用户和地址替换为实际云端连接信息：

```bash
ssh -o ExitOnForwardFailure=yes -N \
  -L 8848:127.0.0.1:8848 <cloud-user>@<cloud-host>
```

客户端连接 `ws://127.0.0.1:8848`。云端 PolicyServer 持有动作缓存和 reset 状态，机器人联调时使用独占的服务实例。

### 5.3 请求和返回

观测消息的 `type` 为 `obs`，包含以下字段：

| 字段 | Linglong 内容 |
| --- | --- |
| `images.head_camera` | 头部 RGB 图像的 base64 PNG/JPEG |
| `images.left_wrist_camera` | 左腕 RGB 图像的 base64 PNG/JPEG |
| `images.right_wrist_camera` | 右腕 RGB 图像的 base64 PNG/JPEG |
| `state` | 24 个原始测量 q 值 |
| `prompt` | 与任务相符的训练指令文本 |

客户端提供各相机图像，由服务端按 checkpoint 完成拼接、resize 和状态归一化。prompt 原样传入模型。

响应字段为 `type=action`、`action`、`step`、`latency_ms`。`action` 为形状 `(24,)` 的反归一化绝对目标，客户端按目标关节位置解释。模型加载中的占位维度处理会将 `action[22:24]` 置零；云端没有针对 Linglong 全关节限位或夹爪 [0,1] 的统一裁剪。

`{"type":"reset"}` 返回 `reset_ack`，清空动作缓存并归零 step；`{"type":"ping"}` 返回 `pong`。

### 5.4 动作生成与缓存

上述同步配置下，首次请求生成 `(32, 24)` 动作块，返回第一帧；接下来 31 次请求依次取缓存。缓存用尽后，使用下一次请求中的观测生成动作块。每条 WebSocket 响应携带一帧动作。

`--inference-horizon 1` 表示每次请求重新生成动作块，只使用第一帧。`--inference-horizon 32` 表示消费完整块。这一参数控制重规划间隔。

训练动作时间间隔为 1/15 秒。机器人执行节奏由本地控制端负责，服务端按请求响应；网络耗时和生成动作块的耗时都需要纳入本地调度。服务端 `latency_ms` 包含当前请求的预处理与取动作过程，取缓存请求和实际生成请求的耗时含义不同。

### 5.5 连通性检查

在云端另开终端运行：

```bash
cd /iag_ad_01/ad/xiejinxia/code/test/OpenWAM
/iag_ad_01/ad/xiejinxia/envs/dreamzero/bin/python \
  scripts/inference_test/inference_single_test.py \
  --server ws://127.0.0.1:8848 \
  --test --state-dim 24
```

该命令使用随机图像和零 state，执行 ping、predict、reset，用于确认通信和推理入口可用。真实输入验证应提供三路真实图像、24 维测量 state 及匹配的任务 prompt。

实际机器人客户端可使用 [benchmarks/utils/client.py](../../benchmarks/utils/client.py) 构造 PNG payload，并通过 [WSPolicyClient](../../benchmarks/utils/transport.py) 的 `predict_once()` 发送单次请求。超时或断线后关闭本轮会话，重新连接并 reset 后开始下一轮，避免不确定的动作请求被自动重发。

