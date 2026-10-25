# Linglong EEF 数据与训练

## 数据约定

训练数据由 `LinglongDataset` 读取，支持 vePFS 本地路径和 TOS 前缀。每条样本包含 head、left、right 三路 RGB，episode 指令、双臂 state 和 action。

| 字段 | 约定 |
| --- | --- |
| `observation.state_eef` | 双臂实测末端状态，16 维 |
| `action_eef` | 双臂绝对末端目标，16 维 |
| 单臂排列 | `xyz + quat(wxyz) + gripper`，左臂在前、右臂在后 |
| 坐标与单位 | 各 arm 的 `base_link`，位置单位为米 |
| 图像 | `head`、`left`、`right` 三路 RGB |
| prompt | episode 的 `tasks[0]`，缺失时使用 manifest prompt |

state 和 action 使用训练集统计分别进行 min-max 归一化。原始 16 维映射到统一 80 维的 `0–6, 9, 34–40, 43`，未映射维度由 mask 排除。模型输入为 `(1, 80)` proprio 和 `(32, 80)` action。

每个窗口包含 33 帧：首帧 state 作为 proprio，前 32 帧作为动作监督。数据频率 15 Hz，视频按 `video_stride=4` 取 9 帧。默认图像画布为 384×320，上方是 256×320 head 视图，下方为两个 128×160 腕部视图。

## TOS 数据

当前 TOS 数据前缀：

```text
tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/20261022_0923_1017_eef_v1
```

数据以 parquet、JPEG 和元数据形式存储。`LinglongDataset` 按需读取对象，并使用节点本地限额缓存；训练不需要先将完整数据集复制到 vePFS。缓存默认目录为 `/tmp/openwam-tos-cache`，每节点上限 4 GiB，保留 1 GiB 空闲空间。

TOS 训练使用同一训练桶计算并加载归一化统计。若配置单独测试桶，需将 `dataloader.eval_dataset_dir` 指向测试前缀，`normalization_dataset_dir` 仍指向训练前缀。

## 训练入口

当前训练配置为 `configs/train_linglong_tos.yaml`，启动脚本为：

```bash
cd /vePFS/tangyuting/OpenWAM-tos
bash scripts/launch_volcano_linglong_tos_eef_8gpu_bs6.sh
```

默认单节点 8 卡、每卡 batch 6、梯度累积 24、5 个 epoch，使用 bf16 和 ZeRO-2。TOS 凭证通过任务环境变量或外部 JSON 文件提供；生产入口默认读取 `/vePFS/tangyuting/.config/openwam/tos-linglong.json`。凭证不应写入仓库或训练配置。

数据入口、凭证、缓存和任务启动的完整说明见 [Linglong TOS 训练与推理](linglong_tos.md)。

## checkpoint、续训与推理

Linglong 与 Piper-X 共用精简 checkpoint 保存和加载实现。当前 TOS 训练只保存可训练参数与持久 buffer；冻结参数从 Foundation 基础权重恢复。每个 checkpoint 清单记录基础权重 SHA-256，并在训练和推理加载时校验。

配置同时开启 `save_full_states_for_resume=true`。每个完整保存点包含同一步的 `checkpoint_step_<N>.safetensors` 与 `accel_state_step_<N>/`；后者保存优化器、调度器、混合精度、随机数和数据读取状态，可用于继续原训练。

续训时指定具体运行目录，并清空预训练入口：

```bash
bash scripts/launch_volcano_linglong_tos_eef_8gpu_bs6.sh \
  training.finetune_ckpt_path=null \
  training.resume_ckpt_path=/path/to/run
```

离线推理和 WebSocket 服务使用相同的模型加载入口：

```bash
bash scripts/deploy.sh \
  --ckpt-dir /path/to/run \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8848
```

推理目录需保留 `config.yaml`、`normalization_stats.npy`、tokenizer 和训练 checkpoint。基础权重可通过本机文件、共享缓存或 checkpoint 清单中的 HTTPS 地址取得；下载后会验证 SHA-256。精简 checkpoint 支持 DDP 和 ZeRO-1/2。
