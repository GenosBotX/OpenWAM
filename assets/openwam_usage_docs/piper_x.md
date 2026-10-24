# Piper-X TOS 训练与推理

## 数据读取

Piper-X 使用 TOS 中按 episode 划分的数据，不需要把数据集复制到 vePFS：

```text
tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261022_eef_v1/train
tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261023_eef_v1/train
```

测试路径由配置中的 `eval_dataset_dir` 指定，归一化统计由 `normalization_dataset_dir` 指定。训练和测试复用训练集统计。数据按需缓存到 `/tmp/openwam-piper-tos-cache`，默认上限为 4 GiB。

## EEF 与图像约定

- 采样频率为 15 Hz；窗口 33 帧，前 32 帧作为动作，视频每 4 帧取一帧。
- TOS parquet 保存双臂 EEF16：每臂为 `xyz + quat(wxyz) + gripper`，坐标系为对应 arm 的 `base_link`，位置单位为米。
- reader 将每臂转换为 EEF20：`xyz + rot6d + gripper`，左臂映射到统一 80 维的 `0–9`，右臂映射到 `34–43`。
- state 和 action 使用同一套 EEF20 约定；位置与夹爪使用训练统计归一化，rot6d 使用 identity 约定。
- 夹爪为 0 到 1，0 表示闭合，1 表示打开。
- 相机顺序为 `head`、`left`、`right`，拼接尺寸为高 384、宽 320。

## 训练

当前训练配置为 `configs/train_piper_x_20261022_20261023_tos_eef_8gpu_e10.yaml`，入口为：

```bash
cd /vePFS/tangyuting/OpenWAM-tos
bash scripts/piper/launch_volcano_piper_x_20261022_20261023_tos_eef_8gpu_e10.sh
```

默认使用单节点 8 卡、每卡 batch 4、梯度累积 24、10 个 epoch、bf16 和 ZeRO-2，从 Piper-X 基础权重初始化。TOS 凭证通过 `OPENWAM_TOS_CREDENTIALS` 或 `TOS_CREDENTIAL_FILE` 指定；也可以由任务环境注入 TOS 凭证变量。凭证不写入仓库或启动参数。

数据读取、归一化、checkpoint 保存和训练状态均由通用 OpenWAM 训练入口处理；Piper-X 专属代码只负责 EEF 数据格式、TOS 数据源和图像布局。

## 精简 checkpoint

Piper-X 和 Linglong 使用同一套精简 checkpoint 实现。当前配置开启：

```yaml
training:
  save_trainable_only: true
  save_full_states_for_resume: true
```

每个 `checkpoint_step_N.safetensors` 保存当前实际参与训练的参数和持久 buffer，冻结的基础权重不重复保存。checkpoint 清单记录基础权重路径、公共地址、SHA-256、参数布局以及训练配置和归一化文件校验值。

同一步的 `accel_state_step_N/` 保存优化器、调度器、混合精度、随机数和数据读取状态。精简续训支持 DDP、ZeRO-1 和 ZeRO-2，不支持 FSDP、ZeRO-3。

## 续训

续训时指定具体运行目录，并清空预训练入口：

```bash
bash scripts/piper/launch_volcano_piper_x_20261022_20261023_tos_eef_8gpu_e10.sh \
  training.finetune_ckpt_path=null \
  training.resume_ckpt_path=/path/to/run
```

续训应使用与原运行一致的数据、归一化统计、batch、梯度累积、进程数和随机种子。

## 推理与 WebSocket

离线推理和 WebSocket 服务共用 `openwam.deploy.model_loader.load_from_checkpoint_dir`。服务启动入口为：

```bash
cd /vePFS/tangyuting/OpenWAM-tos
bash scripts/deploy.sh \
  --ckpt-dir /path/to/run \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8848
```

加载精简 checkpoint 时，服务读取运行目录中的配置、归一化统计和训练权重，从本机基础权重、共享缓存或 checkpoint 清单中的 HTTPS 地址取得冻结参数，并校验 SHA-256 后完成合并。WebSocket client 的请求格式不需要修改。

异地部署可指定基础权重和缓存：

```bash
export OPENWAM_BASE_CHECKPOINT=/shared/models/piper-x-base.safetensors
export OPENWAM_CHECKPOINT_CACHE=/shared/cache/openwam-checkpoints
```

运行目录需要保留 `config.yaml`、`normalization_stats.npy`、tokenizer 资源和至少一个 `checkpoint_step_N.safetensors`。
