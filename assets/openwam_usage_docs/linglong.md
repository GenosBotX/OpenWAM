# Linglong 训练、评测与云端推理

本文统一维护 Linglong 数据、训练、火山云任务及评测部署说明。当前 vePFS 训练入口使用 0923 + 1010 + 1012 + 1017 + fittings 的实测 state EEF → action EEF 数据，单机 8 卡、每卡 bs4、梯度累积 24。

## 1. vePFS 环境和目录

| 项目 | 路径 |
| --- | --- |
| 仓库 | `/vePFS/tangyuting/OpenWAM` |
| 公共 Python 环境 | `/vePFS/tangyuting/envs/openwam/bin/python` |
| Wan2.2 视频模型及 VAE | `/vePFS/tangyuting/models/Wan2.2-TI2V-5B` |
| Foundation 初始化权重 | `/vePFS/tangyuting/OpenWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model` |
| 全量 portable 数据根目录 | `/vePFS/data/linglong/linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/data` |
| 训练配置 | [train_linglong_0923_1010_1012_1017_fittings_eef_gtstate_8gpu.yaml](../../configs/train_linglong_0923_1010_1012_1017_fittings_eef_gtstate_8gpu.yaml) |
| 输出根目录 | `/vePFS/tangyuting/OpenWAM/outputs/linglong_0923_1010_1012_1017_fittings_eef_gtstate_8gpu_bs4_official_finetune` |
| 启动日志 | `/vePFS/tangyuting/OpenWAM/logs/linglong_1017_eef8/` |

启动脚本自动加载公共环境；手动使用环境时，在 Bash 中执行：

```bash
source /vePFS/tangyuting/OpenWAM/scripts/activate_vepfs_openwam.sh
```

激活脚本设置 Python、PYTHONPATH、公共 Hugging Face 缓存及离线 W&B，并切换至仓库。训练 worker 需要挂载上述相同的 vePFS 路径。依赖声明见 [pyproject.toml](../../pyproject.toml)。

本文第 2.1–2.4 节保留原有 0923+1010 关节版本的数据语义，第 2.5 节介绍当前 EEF 版本。第 4、5 节保留关节 checkpoint 的评测和服务示例，其历史 run 和对应数据需另行准备，不能套用其 24 维关节语义解释 EEF 输出。

## 2. 数据与训练输入

### 2.1 24 维状态和动作

原有 0923+1010 配置使用关节空间表示：

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

[scripts/convert_linglong.py](../../scripts/convert_linglong.py) 读取 ROS2 bag 原始 zip、tar 或文件夹，在仓库内解码 CDR，并按头部相机时间戳对齐到 15 Hz；输出 joint/action、FK `action_eef`、实测 `/end_pos` `observation.state_eef`、图像、元数据和归一化统计。任务文本通过 `--prompt` 显式指定；多任务数据可用 `--episode-config` 按原始 episode ID 指定 prompt 和 split。

```bash
python scripts/convert_linglong.py \
  --input /path/to/raw_linglong.zip \
  --output-root /path/to/openwam_linglong \
  --prompt "Use the right arm to pick up the blue block from the table and place it in the box." \
  --test-fraction 0.1 --seed 0
```

输入可为 zip（内含每 episode tar）、原始文件夹、tar(.gz)、单个 episode 目录或 `.db3`。生成 `linglong_openwam_train_15hz/` 和 `linglong_openwam_test_15hz/`；两者均自带图像文件，训练统计由 train 计算并由 test 复用。FK 默认使用仓库内 `assets/linglong/kinematics.xml`，不需要外部模型代码或模型文件；`--model` 仅用于显式覆盖 FK 资产。若需固定 episode 划分或每集不同任务，可传 JSON，例如 `{"2026-10-10T1206.48/采集任务-test_88.tar!bag/bag_0.db3":{"prompt":"...","split":"test"}}`。不提供 prompt 的 episode 必须在该 JSON 中有自己的 `prompt`；文件名不会被当作任务描述。

仅转换数据需要 NumPy、SciPy、Pillow、PyArrow，无需 ROS 或 GPU。原始 bag 必须包含 `/driver_pvt`、`/end_pos` 和 head/left/right 三路压缩图像；输出同时保留 joint 与 EEF 两组列。夹爪直接保留原始 `q/q_exp`，不做反馈修复。相同 bag 按 SHA256 去重；时间戳不连续、缺少必需流或不足 33 个采样帧时会报错。输出目录须尚未存在；只有一个 episode 或 `--test-fraction 0` 时默认仅生成 train 桶。

训练关节表示时使用 `action_key: action`、`state_key: observation.state`、`action_mode: joint` 及 24 维映射。训练 EEF 表示时配置如下（训练和测试的归一化均指向 train 桶）：

```yaml
dataloader:
  dataset_dir: /path/to/openwam_linglong/linglong_openwam_train_15hz
  normalization_dataset_dir: /path/to/openwam_linglong/linglong_openwam_train_15hz
  action_key: action_eef
  state_key: observation.state_eef
  action_mode: eef_quat
  unify_action: true
  unify_action_map: ["0-6", 9, "34-40", 43]
  unify_state_map: ["0-6", 9, "34-40", 43]
  active_raw_action_dims: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  active_raw_state_dims: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

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

映射、统计和图像配置以所加载 checkpoint 的 `config.yaml` 为准。`configs/dataloader/linglong.yaml` 是独立的数据配置入口；原有关节完整配置为 `train_linglong_0923_1010`。当前第 3 节启动命令使用第 2.5 节的 EEF 配置。

### 2.5 全量实测 state EEF → action EEF

当前使用上传的 portable 数据，覆盖 0923、1010、1012、1017、1017_fittings，保留各 episode 的具体物体指令。

| 数据集 | 目录（相对于第 1 节数据根目录） | episode 数 | 窗口数 |
| --- | --- | --- | --- |
| 训练 | `linglong_0923_1010_1012_1017_fittings_eef_gtstate_train_15hz` | 721 | 116,373 |
| 测试 | `linglong_0923_1010_1012_1017_fittings_eef_gtstate_test_15hz` | 82 | 14,553 |

输入 `observation.state_eef` 为记录的双臂实测末端位姿；目标 `action_eef` 为数据中已有的绝对末端目标。两者均为 16 维：左臂 xyz、wxyz 四元数、夹爪，再接右臂对应字段。坐标系为 `base_link`，位置单位为米，`action_mode=eef_quat`。这是绝对 EEF 目标训练。

状态和动作分别使用训练集 `meta/state_eef_stats.json`、`meta/action_eef_stats.json` 归一化，测试复用训练统计。映射均为 `["0-6", 9, "34-40", 43]`，原始 16 维均有效。模型接收 `(32, 80)` 动作和 `(1, 80)` proprio，视频窗口仍为 9 帧。

### 2.6 allstuff 与原始 prompt 版本

上游提交 `817b76e968ac5584ac9e07d28effe66e1acdc2c4`（2026-10-20 16:22，北京时间）引入 [allstuff 配置](../../configs/train_linglong_0923_1010_1012_1017_fittings_eef_gtstate_allstuff_8gpu.yaml) 和对应脚本。

allstuff 对同一批采集数据扩充语言指令：保留 714 条具体物体指令，复制这 714 条轨迹并改用 “pick up all stuff” 指令；另外 7 条原有 “all blocks” 指令直接改为 “all stuff”。训练条目共 1,435 条、227,815 个窗口，没有增加采集轨迹。state/action 仍为 16 维 EEF。

当前 portable 训练配置继承 allstuff 的训练参数，但覆盖了数据目录，实际使用原始 prompt 的 721 条训练集。第 3 节命令启动的是这个原始 prompt 版本。

### 2.7 原有配置的数据路径和可用状态

原有配置的模型地址已改为第 1 节公共模型路径，数据根目录已改为 `/vePFS/data/linglong/`，保留各版本名：

| 配置 | 相对于 `/vePFS/data/linglong/` 的数据目录 | 2026-10-20 检查状态 |
| --- | --- | --- |
| `dataloader/linglong` | `linglong_0923_openwam_train_15hz` | 配置路径下缺少数据 |
| `train_linglong_0923_1010` | `linglong_0923_1010_openwam_train_15hz` | 配置路径下缺少数据 |
| `…eef_gtstate_allstuff_8gpu` | `linglong_0923_1010_1012_1017_fittings_eef_gtstate_allstuff_train_15hz` | 配置路径下缺少数据；portable 内已有同名子目录 |

allstuff 数据实际位于 `/vePFS/data/linglong/linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/data/linglong_0923_1010_1012_1017_fittings_eef_gtstate_allstuff_train_15hz`。这里只记录其存在及配置路径差异；当前验证通过的训练入口是第 3 节的原始 prompt portable 版本。上游原有 32 卡和 allstuff 启动脚本仍含原环境路径，使用那些入口前需要另行适配脚本，更新 YAML 路径不会自动更新脚本。

## 3. 火山云训练启动与 checkpoint

### 3.1 当前训练配置

以第 1 节 Foundation 权重开始新一轮微调，默认 `resume_ckpt_path=null`。模型为 dual_system / joint_self_attn，使用 Wan2.2 TI2V-5B，统一 state/action 宽度为 80。

| 参数 | 配置值 |
| --- | --- |
| GPU 数 | 1 worker × 8 GPU |
| 每 GPU batch size | 4 |
| 梯度累积 | 24 |
| 有效 batch size | `4 × 8 × 24 = 768` |
| 学习率 / epoch | `1e-4` / 5 |
| scheduler | cosine，warmup ratio 0.05 |
| 精度 / 分片 | bf16 / DeepSpeed ZeRO-2 |
| 梯度检查点 | 开启 |
| 保存间隔 / 保留数 | `save_steps=200` / `keep_last_k_ckpts=2` |
| 完整训练状态 | `save_full_states_for_resume=true` |

### 3.2 火山云任务与入口命令

在火山云训练控制台复用已有可用任务的镜像、资源队列及 vePFS 挂载，选择 **1 个 worker，每个 worker 8 张 GPU**。将启动命令设置为：

```bash
cd /vePFS/tangyuting/OpenWAM && \
bash scripts/launch_volcano_linglong_0923_1010_1012_1017_fittings_eef_gtstate_8gpu.sh
```

这是任务容器内的训练入口；在开发机执行它不会向平台申请 GPU。平台任务提交后，训练在 worker 内运行，不依赖本地 SSH 窗口存活。

[启动脚本](../../scripts/launch_volcano_linglong_0923_1010_1012_1017_fittings_eef_gtstate_8gpu.sh) 自动激活公共环境、检查模型和数据文件，并通过 `python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train.py` 启动训练。脚本清除旧的数据环境变量覆盖；收到多 worker 配置时退出并报错。无需手动填写 `MLP_WORKER_ALL_HOSTS`，不要复制 `worker-0,worker-1,…` 示例占位主机名。

[任务 YAML 模板](../../volcano_openwam_linglong_0923_1010_1012_1017_fittings_eef_gtstate_8gpu.yaml) 与以上入口一致。控制台启动时镜像、队列和 GPU 规格由任务表单配置，不需要在训练命令中写 `--set`。如果使用 CLI 提交 YAML，模板中的 `imageUrl`、`resourceQueueName`、`gpuFlavor` 仍需要由已有提交流程替换；该模板本身不自动发现资源。

开发机上可以先执行只检查资产、不启动训练的命令：

```bash
cd /vePFS/tangyuting/OpenWAM && \
OPENWAM_LAUNCH_DRYRUN=1 \
bash scripts/launch_volcano_linglong_0923_1010_1012_1017_fittings_eef_gtstate_8gpu.sh
```

### 3.3 日志、输出和恢复训练

真实启动日志写入 `/vePFS/tangyuting/OpenWAM/logs/linglong_1017_eef8/launch_*.log`，同时输出到平台 worker 日志。若任务立即结束且没有共享日志，应检查平台实例日志、入口命令、镜像和挂载；共享环境加载失败可能发生在日志文件创建之前。

输出根目录为：

```text
/vePFS/tangyuting/OpenWAM/outputs/linglong_0923_1010_1012_1017_fittings_eef_gtstate_8gpu_bs4_official_finetune/<run>/
  config.yaml
  checkpoint_step_<N>.safetensors
  normalization_stats.npy
  tokenizer/
  accel_state_step_<N>/
    trainer_state.json
    ... Accelerate / DeepSpeed 状态文件 ...
```

完整状态由全部 rank 在优化器更新边界参与保存，`trainer_state.json` 为保存完成标记。恢复已有训练时，向启动脚本追加以下两项：

```text
training.finetune_ckpt_path=null
training.resume_ckpt_path=/绝对路径/目标run目录
```

恢复目录必须具有可用的 `accel_state_step_*`；只有 safetensors 权重时应使用 `finetune_ckpt_path` 开始新 run。

### 3.4 本地适配范围与验证

本地适配基于上游 `817b76e968ac5584ac9e07d28effe66e1acdc2c4`。新增一套 portable EEF 配置、单机 8 卡启动脚本、任务模板和公共环境激活脚本；原有 Linglong 配置仅改模型、数据及输出路径。本说明统一维护于 `assets/openwam_usage_docs/linglong.md`。

当前 codediff：`/vePFS/tangyuting/codediff/openwam_current/`，包含 `BASE_COMMIT`、`adaptation.patch`、文件清单和验证记录。旧适配已备份到 `/vePFS/tangyuting/backups/`。

已验证配置解析、模型文件、五批数据的 train/test 样本读取、启动脚本语法及 dry-run。文档和路径同步本身不代表新提交的 GPU 训练已成功运行。

## 4. 关节版本的离线推理、动作评测与视频生成

本节为原有 24 维关节版本示例。历史 run `2026-10-12_01-59-42` 和 0923+1010 数据需先放到对应位置；示例不表示这些资源已存在于 vePFS。EEF 评测应按 checkpoint 配置选择 16 维 EEF 字段和对应统计，不能直接采用本节关节部位指标。

离线脚本直接在 `cuda:0` 加载 checkpoint，通过 `JointInferenceEngine.generate()` 推理。每个窗口的输入为起始多视角图像、起始测量 q 和 prompt，输出为完整的 32 帧动作块；开启视频解码后同时获得 9 帧采样视频。

| 脚本 | 功能 |
| --- | --- |
| [eval_linglong_train.py](../../scripts/eval_linglong_train.py) | 指定训练 episode 的窗口推理、动作误差和视频输出 |
| [eval_linglong_test_all.py](../../scripts/eval_linglong_test_all.py) | 加载一次模型，评测测试集各 episode |
| [summarize_linglong_body_mae.py](../../scripts/summarize_linglong_body_mae.py) | 读取评测结果，按部位汇总 MAE |
| [prepare_linglong_foundation_eval.py](../../scripts/prepare_linglong_foundation_eval.py) | 配置官方 Foundation 权重的 Linglong 评测目录 |

在运行评测的终端先设置工作目录，并清除数据路径覆盖：

```bash
cd /vePFS/tangyuting/OpenWAM
unset OPENWAM_LINGLONG_DATASET_DIR OPENWAM_LINGLONG_SOURCE_ROOT
```

### 4.1 单 episode 推理和视频

```bash
/vePFS/tangyuting/envs/openwam/bin/python \
  scripts/eval_linglong_train.py \
  --checkpoint-dir outputs/linglong_0923_1010_32gpu_official_finetune/2026-10-12_01-59-42 \
  --dataset-dir /vePFS/data/linglong/linglong_0923_1010_openwam_train_15hz \
  --episode 0 --chunk-stride 32 --max-windows 1000 \
  --denoise-steps 10 --seed 42 \
  --decode-video \
  --output outputs/linglong_eval/train_episode_00/metrics.json
```

只运行一个窗口可设置 `--max-windows 1`。视频默认保存在输出 JSON 同级的 `video/`，也可通过 `--video-output` 指定目录。

### 4.2 全测试集推理和视频

```bash
/vePFS/tangyuting/envs/openwam/bin/python \
  scripts/eval_linglong_test_all.py \
  --checkpoint-dir outputs/linglong_0923_1010_32gpu_official_finetune/2026-10-12_01-59-42 \
  --dataset-dir /vePFS/data/linglong/linglong_0923_1010_openwam_test_15hz \
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
/vePFS/tangyuting/envs/openwam/bin/python \
  scripts/summarize_linglong_body_mae.py \
  outputs/linglong_eval/test_all
```

输出为 `body_part_metrics/body_part_mae.json` 和 `body_part_metrics/body_part_mae.csv`，覆盖左臂、左夹爪、右臂、右夹爪、腰部和头部。各动作时刻等权，部位内各维 MAE 再取平均。

离线指标衡量记录观测下的动作预测误差；真机闭环任务成功率需要单独执行并统计。

### 4.4 Foundation 评测准备

`prepare_linglong_foundation_eval.py` 接受 `--foundation`、`--reference` 和 `--output`：引用 Foundation 权重与 tokenizer，使用 reference 的 Linglong 数据配置、归一化统计和本地模型资源路径，生成评测目录。输出目录须尚未存在，生成后交给上述评测脚本。

该工具当前固定引用 `checkpoint_step_154000.safetensors`，适用于具有该文件的 Foundation 包。权重通过符号链接引用，源目录需要保持可访问。

## 5. 关节版本的云端 WebSocket 服务

本节描述 24 维关节 checkpoint 的消息语义。使用 EEF checkpoint 时，需要客户端按其 16 维末端位姿布局提供实测 state 并解释输出；本节关节目标的执行方式不适用于 EEF。

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
cd /vePFS/tangyuting/OpenWAM && \
PATH=/vePFS/tangyuting/envs/openwam/bin:$PATH \
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
  -L 8848:127.0.0.1:8848 'cloud-user@cloud-host'
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
cd /vePFS/tangyuting/OpenWAM
/vePFS/tangyuting/envs/openwam/bin/python \
  scripts/inference_test/inference_single_test.py \
  --server ws://127.0.0.1:8848 \
  --test --state-dim 24
```

该命令使用随机图像和零 state，执行 ping、predict、reset，用于确认通信和推理入口可用。真实输入验证应提供三路真实图像、24 维测量 state 及匹配的任务 prompt。

实际机器人客户端可使用 [benchmarks/utils/client.py](../../benchmarks/utils/client.py) 构造 PNG payload，并通过 [WSPolicyClient](../../benchmarks/utils/transport.py) 的 `predict_once()` 发送单次请求。超时或断线后关闭本轮会话，重新连接并 reset 后开始下一轮，避免不确定的动作请求被自动重发。

