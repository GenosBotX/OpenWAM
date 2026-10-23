# Linglong EEF 数据读取与训练

## 数据与输入

当前训练使用 0923、1010、1012、1017 和 fittings 的自采数据，保留各 episode 的任务指令。训练集 721 个 episode，划窗间隔 1 时为 116,373 个窗口；本地测试集 82 个 episode、14,553 个窗口。

| 字段 | 含义 |
| --- | --- |
| `observation.state_eef` | 双臂实测末端状态，16 维 |
| `action_eef` | 双臂绝对末端目标，16 维 |
| 单臂排列 | xyz、wxyz 四元数、夹爪；左臂在前、右臂在后 |
| 坐标与单位 | `base_link`，位置为米 |
| 图像 | head、left、right 三路 RGB |
| prompt | 优先使用 `meta/episodes.jsonl` 中的 `tasks[0]`，缺失时使用 manifest 的 prompt |

状态和动作分别使用训练集 `meta/state_eef_stats.json`、`meta/action_eef_stats.json` 做 min-max 归一化；测试复用训练统计。两者均使用 `["0-6", 9, "34-40", 43]` 映射到 80 维，原始 16 维有效，其余槽位由 mask 排除。模型接收 `(32, 80)` 动作和 `(1, 80)` proprio。

窗口包含 33 个原始时刻：首帧状态作为 proprio，前 32 帧作为动作监督。动作频率为 15 Hz；`video_stride=4` 取相对帧 `0,4,...,32`，共 9 帧。`window_stride=1` 控制相邻训练窗口的起点间隔，与视频抽帧间隔是两个参数。

三路图像经 BILINEAR resize 后拼接，尺寸均按高×宽表示：默认画布 384×320，上方 head 为 256×320，下方左右腕各为 128×160。288×256 入口使用相同布局比例，上方为 192×256，下方各为 96×128。

## 本地与 TOS 读取

[LinglongDataset](../../openwam/dataloader/linglong.py) 使用统一的数据处理流程，`dataset_dir` 可为 vePFS 本地目录或 `tos://` 前缀。两种来源使用相同的字段、归一化、映射、掩码、图像布局和窗口采样。

本地训练桶：

```text
/vePFS/data/linglong/linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/data/linglong_0923_1010_1012_1017_fittings_eef_gtstate_train_15hz
```

对应测试桶将末尾 `_train_15hz` 替换为 `_test_15hz`。本地配置中 `normalization_dataset_dir` 始终指向训练桶；本地读取不启用 TOS 缓存。

已发布的完整 TOS 训练数据：

```text
tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/20261022_0923_1017_eef_v1
```

桶内保留 portable 数据结构：

```text
<dataset-prefix>/
  manifest.json
  meta/
    episodes.jsonl
    normalization_stats.json
    action_eef_stats.json
    state_eef_stats.json
    ...
  data/chunk-000/episode_000000.parquet
  ... parquet 中引用的相对 JPEG 路径 ...
```

TOS manifest 必须标记 `status=complete`，`source_root` 为 `.`。发布器上传图像、parquet 和元数据后最后写入 manifest；已发布前缀不可覆盖，更新数据需要使用新前缀。TOS 训练配置的 `normalization_dataset_dir` 指向同一训练前缀。TOS 测试需单独发布 test 桶并设置 `dataloader.eval_dataset_dir`，归一化继续指向 train；不会自动猜测 TOS 测试地址。

## 限额缓存与提前读取

[DatasetStorage / BoundedTosCache](../../openwam/dataloader/tos_cache.py) 将 TOS 对象按需下载到节点本地磁盘，缓存命中直接读本地文件。启动时读取 manifest、episode 索引和统计；采样时按需读取 parquet、JPEG。解析后的 parquet 另有每进程最多 128 张表的内存 LRU，此内存不计入磁盘缓存限额。

| 配置 | 默认值 |
| --- | --- |
| `dataloader.cache_dir` | `${OPENWAM_TOS_CACHE_DIR:-/tmp/openwam-tos-cache}` |
| `dataloader.cache_max_bytes` | 4,294,967,296，即每节点 4 GiB |
| `dataloader.cache_min_free_bytes` | 1,073,741,824，即保留 1 GiB 空闲空间 |
| `training.dataset_num_workers` | 每 rank 8 个 worker |
| DataLoader prefetch | PyTorch 默认 `prefetch_factor=2`，每 worker 提前准备 2 个 batch |
| `pin_memory` | `true` |

同节点全部 rank/worker 共享一个缓存目录，容量和空闲阈值必须一致。使用节点本地 Linux 文件系统，避免将 SQLite 和文件锁放在共享 vePFS 上；不同节点分别缓存。调整限额时用新目录，或在所有使用者退出后清理旧缓存再启动。

文件锁防止重复下载和读取期间被淘汰；SQLite 索引以事务维护容量计数和 LRU 淘汰。缓存限额覆盖对象内容和下载中的预留容量，索引及锁文件有额外磁盘开销。下载使用 ETag 条件校验并检查长度；过大对象、空间不足且文件仍被占用、下载失败会报错。

DataLoader worker 在模型计算期间准备后续 batch，包括 TOS 下载、解码和拼接。读取属于按需预取：没有独立全量预下载服务，也不会在训练前复制完整数据集；缓存未命中且预取队列不足时，训练仍会等待 I/O。4 GiB 是节点磁盘缓存上限，worker 的解码结果、prefetch 队列和 parquet 内存缓存需另外预留主机内存。

## 环境与凭证

仓库路径为 `/vePFS/tangyuting/OpenWAM-tos`，分支为 `feat/linglong-tos`。公共 Python 为 `/vePFS/tangyuting/envs/openwam/bin/python`，Wan 模型为 `/vePFS/tangyuting/models/Wan2.2-TI2V-5B`。Foundation 初始化权重使用配置中的 `/vePFS/tangyuting/OpenWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model`。

安装可选 TOS 依赖：

```bash
/vePFS/tangyuting/envs/openwam/bin/python -m pip install -e '.[tos]'
```

凭证由任务环境中的 `TOS_ACCESS_KEY`、`TOS_SECRET_KEY` 提供，或通过外部 JSON 文件加载。生产 launcher 默认使用 `/vePFS/tangyuting/.config/openwam/tos-linglong.json`；文件需归任务用户所有，权限仅允许用户读写（0600），键名与环境变量相同。可选键为 `TOS_SECURITY_TOKEN`、`TOS_ENDPOINT`、`TOS_REGION`，默认 endpoint 为 `tos-cn-beijing.volces.com`、region 为 `cn-beijing`。凭证不放入仓库、patch 或训练 YAML。

## 8 卡训练

| 参数 | 值 |
| --- | --- |
| GPU / batch / 累积 | 1 worker × 8 GPU，每卡 bs6，accum24 |
| 有效 batch | `8 × 6 × 24 = 1152` |
| 窗口 / 视频抽帧间隔 | 1 / 4 |
| epoch / 学习率 | 5 / `1e-4` |
| 精度 / 分片 | bf16 / DeepSpeed ZeRO-2 |
| 保存间隔 | `save_steps=200`，按 micro-step 计数 |
| TOS 权重保留数 | `keep_last_k_ckpts=1` |
| 完整恢复状态 | `save_full_states_for_resume=false` |

在 Volcano 任务中配置 1 worker、每 worker 8 GPU，复用公共环境和 vePFS 挂载。任务容器启动命令：

```bash
cd /vePFS/tangyuting/OpenWAM-tos && \
bash scripts/launch_volcano_linglong_tos_eef_8gpu_bs6.sh
```

288×256 输入使用：

```bash
cd /vePFS/tangyuting/OpenWAM-tos && \
bash scripts/launch_volcano_linglong_tos_eef_8gpu_bs6_h288_w256.sh
```

288×256 入口仅调整画布尺寸及输出目录，数据、batch、累积、窗口采样等参数相同。上述为任务容器入口；资源申请由 Volcano 控制台或现有提交工具完成。任务模板见 [volcano_openwam_linglong_tos_eef_8gpu_bs6.yaml](../../volcano_openwam_linglong_tos_eef_8gpu_bs6.yaml)。

启动前检查展开的命令（不启动 GPU 训练，也不验证 TOS 连通性）：

```bash
OPENWAM_LAUNCH_DRYRUN=1 bash scripts/launch_volcano_linglong_tos_eef_8gpu_bs6.sh
```

可通过 `LINGLONG_TOS_DATASET`、`TOS_CREDENTIAL_FILE`、`OPENWAM_TOS_CACHE_DIR`、`LINGLONG_TOS_OUTPUT_PATH` 覆盖地址；通过脚本附加的 Hydra 参数覆盖配置，例如 `dataloader.cache_max_bytes=8589934592`。通用入口 [train_linglong_tos.sh](../../scripts/train_linglong_tos.sh) 也支持直接使用任务注入的凭证环境变量，需显式设置 `LINGLONG_TOS_DATASET`。

## 输出与 checkpoint

默认输出仍保存在 vePFS：

```text
/vePFS/tangyuting/OpenWAM/outputs/linglong_tos_0923_1017_eef_8gpu_bs6_acc24/<run>/
  config.yaml
  checkpoint_step_<N>.safetensors
  normalization_stats.npy
  tokenizer/
```

288×256 入口对应输出根目录末尾追加 `_h288_w256`。每个 run 保存实际解析配置，loss 等日志沿用标准 trainer。`global_step` / checkpoint 文件名中的 step 是 micro-step；优化器一般每 24 个 micro-step 更新一次，每个 epoch 末会处理剩余累积。

`save_steps` 为正时，正常结束会额外保存最终权重，即使总步数未到保存间隔。当前 TOS 配置只保存权重，不生成完整的 `accel_state_step_*`；用 `training.finetune_ckpt_path=/path/to/run` 可从权重开始新 run。需要完整断点恢复时，在首次训练时启用 `training.save_full_states_for_resume=true`；恢复时设置 `training.finetune_ckpt_path=null training.resume_ckpt_path=/path/to/run`，且该 run 须有完整状态。checkpoint 不自动上传到 TOS。

发布新数据及 TOS 的独立使用说明见 [linglong_tos.md](linglong_tos.md)。
