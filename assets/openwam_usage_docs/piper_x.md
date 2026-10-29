# Piper-X TOS 训练与推理

## 数据读取

Piper-X 使用 TOS 中按 episode 划分的数据，不需要把数据集复制到 vePFS：

```text
tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261022_eef_v1/train
tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261023_eef_v1/train
tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261024_eef_v1/train
tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261026_eef_v2/train
tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261026_static_eef_v2/train
tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261027_eef_v1/train
tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261027_static_eef_v1/train
tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261028_eef_v1/train
```

测试路径由配置中的 `eval_dataset_dir` 指定，归一化统计由 `normalization_dataset_dir` 指定。训练和测试复用训练集统计。数据按需缓存到 `/tmp/openwam-piper-tos-cache`，默认上限为 4 GiB。

| 数据源 | 训练 episode | 训练原始帧 | 训练窗口 | 测试 episode | 测试原始帧 | 测试窗口 |
|---|---:|---:|---:|---:|---:|---:|
| `20261022_eef_v1` | 49 | 60,931 | 59,363 | 6 | 7,589 | 7,397 |
| `20261023_eef_v1` | 47 | 55,918 | 54,414 | 5 | 5,749 | 5,589 |
| `20261024_eef_v1` | 211 | 71,916 | 65,164 | 23 | 8,100 | 7,364 |
| `20261026_eef_v2` | 19 | 9,983 | 9,375 | 4 | 1,913 | 1,785 |
| `20261026_static_eef_v2` | 13 | 9,512 | 9,512 | 2 | 1,467 | 1,467 |
| `20261027_eef_v1` | 54 | 28,439 | 26,711 | 6 | 2,997 | 2,805 |
| `20261027_static_eef_v1` | 7 | 8,623 | 8,623 | 1 | 894 | 894 |
| `20261028_eef_v1` | 106 | 49,027 | 45,635 | 12 | 5,659 | 5,275 |
| **合计** | **506** | **294,349** | **278,797** | **59** | **34,368** | **32,576** |

训练和测试按完整 episode 划分。新增数据在各日期的抓放和空景类别内使用随机种子 0、约 10% 的测试比例；已划分的抓放序列保留其 train/test 归属。ready/review 标签不参与筛选。

抓放序列保留原时序和真实动作目标，以 stride 1 在线构造 33 帧窗口，每个 episode 产生「帧数 - 32」个窗口。10 月 26 日的抓放有 23 个 episode，10 月 27 日的抓放有 60 个 episode，10 月 28 日的抓放有 118 个 episode。

10 月 26 日和 27 日的空景分别为 15 个和 8 个 episode，共 20,496 个原始帧。每帧生成一个静止样本，其中 18,135 个用于训练、2,361 个用于测试。每个样本将同一帧的三路图像重复到视频的 9 个采样位置，以该帧实测 EEF state 作为全部 32 步 action。图像在 TOS 上只保存一份，由 reader 构造重复序列。

全部数据使用同一条 prompt：

```text
Use the right hand to continuously place the objects on the table into the container on the table.
```

绝对 EEF 版本的 8 个训练源共享 `20261022_20261028_normalization_v1` 中的统计。min/max 等统计覆盖 506 个训练 episode 的 294,349 个原始行，每行仅计一次；测试数据不参与统计，rot6d 使用 identity 约定。

## EEF 与图像约定

- 采样频率为 15 Hz；窗口 33 帧，前 32 帧作为动作，视频每 4 帧取一帧。
- TOS parquet 保存双臂 EEF16：每臂为 `xyz + quat(wxyz) + gripper`，坐标系为对应 arm 的 `base_link`，位置单位为米。
- reader 将双臂转换为 EEF20，每臂 10 维：`xyz + rot6d + gripper`，左臂映射到统一 80 维的 `0–9`，右臂映射到 `34–43`。
- state 和 action 使用同一套 EEF20 约定；位置与夹爪使用训练统计归一化，rot6d 使用 identity 约定。
- 夹爪为 0 到 1，0 表示闭合，1 表示打开。
- 相机顺序为 `head`、`left`、`right`，拼接尺寸为高 384、宽 320。

## 训练

当前训练配置为 `configs/train_piper_x_20261022_20261028_tos_eef_16gpu_e5.yaml`，在两个 Volcano worker 上使用同一入口：

```bash
cd /vePFS/tangyuting/OpenWAM-CUDAGraph
OPENWAM_TOS_CREDENTIALS=/vePFS/tangyuting/.config/openwam/tos-linglong.json \
bash scripts/piper/launch_volcano_piper_x_20261022_20261028_tos_eef_16gpu_e5.sh
```

默认使用 2 节点共 16 卡、每卡 batch 6、梯度累积 8、有效 batch 768、5 个 epoch、bf16 和 ZeRO-2，从 Piper-X 基础权重初始化。脚本从 Volcano worker hosts 解析主节点与节点序号，也支持显式设置 `MASTER_ADDR` 和 `NODE_RANK`。

输入分辨率默认高 384、宽 320；使用高 288、宽 256 时在命令末尾加 `dataloader.height=288 dataloader.width=256`。TOS 凭证通过 `OPENWAM_TOS_CREDENTIALS` 或 `TOS_CREDENTIAL_FILE` 指定；也可以由任务环境注入 TOS 凭证变量。凭证不写入仓库或启动参数。

数据读取、归一化、checkpoint 保存和训练状态均由通用 OpenWAM 训练入口处理；Piper-X 专属代码只负责 EEF 数据格式、TOS 数据源和图像布局。

### 相对 EEF 训练

`configs/train_piper_x_20261022_20261028_relative_eef_16gpu_e5.yaml` 训练独立的相对动作模型，绝对版配置与 checkpoint 不变。state 仍是每臂 `base_link` 下的绝对 EEF20；每个 33 帧训练窗口使用首帧 state 作为固定参考，32 步 action 在线转为相对目标：位置为 `p_action - p_state`，旋转为 `R_action @ R_state.T`，左右臂各自在自己的基座轴中表达；夹爪仍是绝对开合比例。窗口内不逐步累加参考 state。

相对 action 的统计与绝对版不同，必须从相同 8 个训练源的全部 stride-1 窗口重新计算。脚本仅读取 parquet 的 state/action 列，min/max 精确覆盖所有相对动作目标；均值、标准差覆盖所有目标，分位数由固定种子的 200,000 行蓄水池样本估算。state 统计仍按训练源的全部绝对 state 帧计算，rot6d 保持 identity。测试集不参与统计。

在开发机使用既有凭证生成并验证 TOS 统计文件：

```bash
cd /vePFS/tangyuting/OpenWAM-CUDAGraph
OPENWAM_TOS_CREDENTIALS=/vePFS/tangyuting/.config/openwam/tos-linglong.json \
/vePFS/tangyuting/envs/openwam/bin/python scripts/run_with_tos_credentials.py \
  /vePFS/tangyuting/.config/openwam/tos-linglong.json \
  /vePFS/tangyuting/envs/openwam/bin/python scripts/piper/compute_relative_eef_stats.py \
  --config-name train_piper_x_20261022_20261028_tos_eef_16gpu_e5 \
  --output /tmp/piper_x_relative_eef20_normalization_stats.json \
  --upload tos://ml-platform-auto-created-required-2109640594-cn-beijing/piperx/openwam/20261022_20261028_relative_normalization_v1/meta/relative_eef20_normalization_stats.json
```

相对统计覆盖 506 个训练 episode、294,349 帧、278,797 个窗口和 8,921,504 个 action 目标。

使用专用 16 卡入口启动：

```bash
OPENWAM_TOS_CREDENTIALS=/vePFS/tangyuting/.config/openwam/tos-linglong.json \
bash scripts/piper/launch_volcano_piper_x_20261022_20261028_relative_eef_16gpu_e5.sh
```

配置为两节点共 16 卡、每卡 batch 6、梯度累积 8、有效 batch 768、5 个 epoch、384×320。它从相同 Piper-X 基础权重开始新训练，不从绝对版优化器状态 resume。

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
bash scripts/piper/launch_volcano_piper_x_20261022_20261028_tos_eef_16gpu_e5.sh \
  training.finetune_ckpt_path=null \
  training.resume_ckpt_path=/path/to/run
```

续训应使用与原运行一致的数据、归一化统计、batch、梯度累积、进程数和随机种子。

## 推理与 WebSocket

离线推理和 WebSocket 服务共用 `openwam.deploy.model_loader.load_from_checkpoint_dir`。服务启动入口为：

```bash
cd /vePFS/tangyuting/OpenWAM-CUDAGraph
bash scripts/deploy.sh \
  --ckpt-dir /path/to/run \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8848
```

加载精简 checkpoint 时，服务读取运行目录中的配置、归一化统计和训练权重，从本机基础权重、共享缓存或 checkpoint 清单中的 HTTPS 地址取得冻结参数，并校验 SHA-256 后完成合并。WebSocket client 的请求格式不需要修改。

相对 EEF checkpoint 始终在服务端做动作反归一化。默认 `inference.relative_action_output=absolute`：PONG 标明训练目标是相对位姿、相对参考为该请求输入 state，服务端在进入同步/异步动作缓存前把整段相对预测恢复成绝对 EEF20，WebSocket 返回绝对位姿。设为 `relative`（或 `--relative-action-output relative`）时跳过还原，下发物理单位的相对 EEF20，PONG 的 `action_target` 改为 `relative_pose`、`relative_actions_decoded_by=client`。两种模式都要求 obs 带正确的 20D 绝对 state。绝对版 checkpoint 拒绝加载相对统计，反向也一样；绝对模型不能开相对下发。

异地部署可指定基础权重和缓存：

```bash
export OPENWAM_BASE_CHECKPOINT=/shared/models/piper-x-base.safetensors
export OPENWAM_CHECKPOINT_CACHE=/shared/cache/openwam-checkpoints
```

运行目录需要保留 `config.yaml`、`normalization_stats.npy`、tokenizer 资源和至少一个 `checkpoint_step_N.safetensors`。
