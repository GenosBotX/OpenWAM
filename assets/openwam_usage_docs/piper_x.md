# Piper-X 华为云 OBS 训练与推理

## 数据

当前训练配置为 `configs/train_piper_x_ascend.yaml`，数据根路径为：

```text
obs://genosbot-shanghai/openwam/data/piper-x
```

桶所在区域为 `cn-southwest-2`，访问端点为 `https://obs.cn-southwest-2.myhuaweicloud.com`。
数据由 parquet、独立 JPEG 和元数据组成，训练按对象读取并缓存到节点本地。

一次训练同时使用以下 7 个数据目录，每个目录分别包含 `train/` 和 `test/`：

| 数据目录 | 样本类型 | 训练 episode 数 | 测试 episode 数 |
| --- | --- | ---: | ---: |
| `20261022_eef_v1` | 时序 | 49 | 6 |
| `20261023_eef_v1` | 时序 | 47 | 5 |
| `20261024_eef_v1` | 时序 | 211 | 23 |
| `20261026_eef_v2` | 时序 | 19 | 4 |
| `20261026_static_eef_v2` | 静态保持 | 13 | 2 |
| `20261027_eef_v1` | 时序 | 54 | 6 |
| `20261027_static_eef_v1` | 静态保持 | 7 | 1 |

目录名中的 `v1/v2` 是各数据目录的发布标识。训练与测试路径由配置中的
`dataloader.dataset_dir` 和 `dataloader.eval_dataset_dir` 指定，按完整 episode 划分。

所有数据源使用共享归一化目录：

```text
obs://genosbot-shanghai/openwam/data/piper-x/20261022_20261027_normalization_v1
```

`dataloader.normalization_dataset_dir` 指向该目录。统计仅使用训练数据，测试读取相同统计。
静态保持样本将同一帧的三路图像重复到 9 个视频位置，并以该帧实测 EEF state
作为全部 32 步 action，学习保持当前位置。

## 凭证与缓存

OBS 客户端通过 ModelArts 环境中的 MoXing 凭证提供器获取并刷新凭证。
每个训练节点的任务身份都需要具备上述 OBS 前缀的读取权限。
当前 `openwam` 环境包含所需 MoXing 和 OBS SDK。

各 rank 和 DataLoader worker 共用节点本地的 LRU 缓存。对象内容默认上限为 4 GiB，
并预留 1 GiB 可用空间；数据库和锁文件另占少量空间。缓存目录需支持 POSIX 文件锁和 SQLite，
同一目录下的所有进程必须使用一致的缓存限额。调整限额时使用新的缓存目录。

| 环境变量 | 默认值 |
| --- | --- |
| `OPENWAM_DATA_ROOT` | `obs://genosbot-shanghai/openwam/data/piper-x` |
| `OPENWAM_OBS_ENDPOINT` | `https://obs.cn-southwest-2.myhuaweicloud.com` |
| `OPENWAM_CACHE_DIR` | `/tmp/openwam-piper-obs-cache` |
| `OPENWAM_CACHE_MAX_BYTES` | `4294967296` |
| `OPENWAM_CACHE_MIN_FREE_BYTES` | `1073741824` |

训练期间数据前缀内容必须保持固定。

## EEF 与图像约定

- 采样频率为 15 Hz；窗口 33 帧，前 32 帧作为动作，视频每 4 帧取一帧。
- parquet 保存双臂 EEF16：每臂为 `xyz + quat(wxyz) + gripper`，坐标系为对应 arm 的 `base_link`，位置单位为米。
- reader 将每臂转换为 EEF20：`xyz + rot6d + gripper`，左臂映射到统一 80 维的 `0–9`，右臂映射到 `34–43`。
- action 为绝对 EEF；state 和 action 使用同一套 EEF20 约定。位置与夹爪使用训练统计归一化，rot6d 使用 identity 约定。
- 夹爪为 0 到 1，0 表示闭合，1 表示打开。
- 相机顺序为 `head`、`left`、`right`，拼接尺寸为高 384、宽 320。

## 训练入口

在 ModelArts 任务启动命令中使用统一入口：

```bash
cd /home/ma-user/work/tangyuting/OpenWAM-CUDAGraph
bash scripts/piper/train_ascend.sh
```

各节点执行同一命令。脚本读取 `MA_NUM_HOSTS`、`VC_TASK_INDEX` 和 `VC_WORKER_HOSTS`
确定节点数、节点序号及主节点地址，也支持显式设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR`
和 `MASTER_PORT`。每节点默认 8 张 NPU；可通过 `NPROC_PER_NODE` 指定设备进程数。

每卡 batch 默认 6，有效 batch 为 768，梯度累积随节点数计算：

| 资源 | 每卡 batch | 梯度累积 | 有效 batch |
| --- | ---: | ---: | ---: |
| 2 机 × 8 NPU | 6 | 8 | 768 |
| 4 机 × 8 NPU | 6 | 4 | 768 |

配置使用 5 个 epoch、bf16 和 ZeRO-2。输出目录为仓库下的 `outputs/piper_x_ascend_1027`。
默认模型与 Python 路径如下，均可通过对应环境变量覆盖：

| 环境变量 | 默认路径 |
| --- | --- |
| `OPENWAM_PYTHON` | `/home/ma-user/anaconda3/envs/openwam/bin/python` |
| `OPENWAM_BASE_CHECKPOINT` | `/home/ma-user/work/tangyuting/models/OpenWAM-Alpha-Real-RoboDojo-Piper-X/checkpoint_step_30000.safetensors` |
| `OPENWAM_WAN_MODEL_DIR` | `/home/ma-user/work/customer-test/OpenWAM-main/assets/video_backbone_ckpt/Wan2.2-TI2V-5B` |

启动前检查本地模型文件，并通过训练的数据读取接口读取全部 14 个 split 的 manifest、
episode 清单和共享归一化统计。仅执行预检：

```bash
OPENWAM_PREFLIGHT_ONLY=1 bash scripts/piper/train_ascend.sh
```

仅输出分布式启动命令：

```bash
NNODES=4 NODE_RANK=0 MASTER_ADDR=192.0.2.10 \
OPENWAM_LAUNCH_DRYRUN=1 bash scripts/piper/train_ascend.sh
```

该示例地址用于 dry-run；实际多机任务使用 ModelArts 提供的主节点地址。

## Checkpoint 与续训

当前配置开启：

```yaml
training:
  save_trainable_only: true
  save_full_states_for_resume: true
```

每个 `checkpoint_step_N.safetensors` 保存可训练参数与持久 buffer。
checkpoint 清单记录基础权重路径、公共地址、SHA-256、参数布局及归一化文件校验值。
同一步的 `accel_state_step_N/` 保存优化器、调度器、混合精度、随机数和数据读取状态。
精简续训支持 DDP、ZeRO-1 和 ZeRO-2。

续训时指定具体运行目录，并清空预训练入口：

```bash
bash scripts/piper/train_ascend.sh \
  training.finetune_ckpt_path=null \
  training.resume_ckpt_path=/path/to/run
```

续训应保持数据、归一化统计、batch、梯度累积、进程数和随机种子一致。

## 推理

离线推理和 WebSocket 服务共用 `openwam.deploy.model_loader.load_from_checkpoint_dir`。
加载精简 checkpoint 时，服务读取运行目录中的配置、归一化统计和训练权重，从本机基础权重、
共享缓存或 checkpoint 清单中的 HTTPS 地址取得冻结参数，并按 SHA-256 校验后合并。

运行目录需要保留 `config.yaml`、`normalization_stats.npy`、tokenizer 资源和至少一个
`checkpoint_step_N.safetensors`。异地部署可通过 `OPENWAM_BASE_CHECKPOINT` 和
`OPENWAM_CHECKPOINT_CACHE` 指定基础权重及缓存目录。
