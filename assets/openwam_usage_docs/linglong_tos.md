# Linglong TOS 训练与推理

## 数据

当前 Linglong 训练数据前缀为：

```text
tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/20261022_0923_1017_eef_v1
```

`LinglongDataset` 直接读取 TOS 对象，训练不需要先把数据集复制到 vePFS。数据字段、EEF16 约定、80 维统一映射、归一化、三路相机布局和窗口采样见 [Linglong 数据说明](linglong.md)。

TOS 对象按需缓存到节点本地目录 `${OPENWAM_TOS_CACHE_DIR:-/tmp/openwam-tos-cache}`。默认每节点缓存上限为 4 GiB，并保留 1 GiB 空闲空间；各 rank 和 DataLoader worker 共用该缓存。凭证通过任务环境变量或外部 JSON 文件提供，不写入仓库。

## 训练

使用 `configs/train_linglong_tos.yaml` 和以下入口：

```bash
cd /vePFS/tangyuting/OpenWAM-tos
bash scripts/launch_volcano_linglong_tos_eef_8gpu_bs6.sh
```

当前默认参数为单节点 8 卡、每卡 batch 6、梯度累积 24、窗口步长 1、视频步长 4、5 个 epoch、bf16 和 ZeRO-2。训练数据、图像和统计量均从 TOS 读取，模型 checkpoint 保存到 `training.output_path` 指定的 vePFS 目录。

启动脚本默认使用 `/vePFS/tangyuting/.config/openwam/tos-linglong.json`。也可以在任务环境中设置：

```bash
export LINGLONG_TOS_DATASET=tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/20261022_0923_1017_eef_v1
export TOS_CREDENTIAL_FILE=/path/to/tos-credentials.json
bash scripts/train_linglong_tos.sh
```

## 精简 checkpoint

Linglong 和 Piper-X 使用同一套 checkpoint 实现。当前 Linglong TOS 配置开启：

```yaml
training:
  save_trainable_only: true
  save_full_states_for_resume: true
```

每个 `checkpoint_step_N.safetensors` 保存当前实际参与训练的参数和持久 buffer，冻结的基础参数不重复保存。清单内记录基础权重的 SHA-256 和公共地址；训练开始时会校验当前模型的冻结参数与基础权重一致。

同一步的 `accel_state_step_N/` 保存优化器、调度器、混合精度、随机数和数据读取状态，用于继续训练。只有权重与该目录配对完整时，保存点才会被续训逻辑选中。

## 续训

续训时指向具体的运行目录，并清空预训练入口：

```bash
bash scripts/launch_volcano_linglong_tos_eef_8gpu_bs6.sh \
  training.finetune_ckpt_path=null \
  training.resume_ckpt_path=/path/to/run
```

续训应保持数据、归一化统计、batch、梯度累积、进程数和随机种子一致。

## 推理与 WebSocket

`scripts/deploy.py` 和 WebSocket 服务共用同一个 checkpoint 加载入口。加载精简 checkpoint 时，程序读取运行目录中的 `config.yaml`、归一化统计和训练权重，从本机基础权重、共享缓存或清单记录的 HTTPS 地址取得冻结参数，并按 SHA-256 校验后合并。client 协议不需要改变。

启动服务：

```bash
cd /vePFS/tangyuting/OpenWAM-tos
bash scripts/deploy.sh \
  --ckpt-dir /path/to/run \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8848
```

异地部署时可以显式指定完整基础权重和缓存目录：

```bash
export OPENWAM_BASE_CHECKPOINT=/shared/models/checkpoint.safetensors
export OPENWAM_CHECKPOINT_CACHE=/shared/cache/openwam-checkpoints
```

运行目录需要保留 `config.yaml`、`normalization_stats.npy`、tokenizer 资源和至少一个 `checkpoint_step_N.safetensors`。精简 checkpoint 支持 DDP、ZeRO-1 和 ZeRO-2；不支持 FSDP、ZeRO-3。

Piper-X 的数据和训练入口见 [Piper-X TOS 说明](../../scripts/piper/README.md)。
