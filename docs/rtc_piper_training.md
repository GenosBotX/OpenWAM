# Piper-X RTC 训练启动说明

更新：2026-11-06。代码位于 lq/training-rtc 分支，远端 `/vePFS/liqi/OpenWAM-CUDAGraph`；本轮训练已产出 checkpoint_step_14525.safetensors（2026-11-06 日志确认最终权重复制完成）。

## 启动文件

- 脚本：`scripts/piper/launch_volcano_piper_x_20261022_20261028_relative_eef_rtc_16gpu_e5.sh`
- 配置：`configs/train_piper_x_20261022_20261028_relative_eef_rtc_16gpu_e5.yaml`
- 配置继承原相对 EEF 训练配置，沿用原数据、初始化 checkpoint 和优化器设置。
- 2 节点，每节点 8 GPU；每卡 batch 6，梯度累积 8，全局有效 batch 768；训练 5 epochs。
- 原始窗口 33 帧、32 个动作，video stride 4 对应 9 个视频帧。
- 每条样本独立从 `{0,6,7,8,9,10,11,12}` 等概率抽取前缀长度，每个值占 12.5%。单位是动作 tick。
- 输出：`/vePFS/liqi/OpenWAM-CUDAGraph/outputs/train_piper_x_20261022_20261028_relative_eef_rtc_16gpu_e5`。

## 两机手动启动

将下面 `10.0.0.1` 替换为节点 0 对节点 1 可达的内网 IP。两机共享同一仓库和环境，使用同一地址、端口；端口应可互通且未被其他训练占用。

节点 0：

```bash
cd /vePFS/liqi/OpenWAM-CUDAGraph
NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  bash scripts/piper/launch_volcano_piper_x_20261022_20261028_relative_eef_rtc_16gpu_e5.sh
```

节点 1：

```bash
cd /vePFS/liqi/OpenWAM-CUDAGraph
NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  bash scripts/piper/launch_volcano_piper_x_20261022_20261028_relative_eef_rtc_16gpu_e5.sh
```

Volcano 若已提供 worker hosts 和可识别的 worker 编号，则两个节点执行相同脚本即可自动确定节点参数：

```bash
cd /vePFS/liqi/OpenWAM-CUDAGraph
bash scripts/piper/launch_volcano_piper_x_20261022_20261028_relative_eef_rtc_16gpu_e5.sh
```

在对应命令前添加 `OPENWAM_LAUNCH_DRYRUN=1` 只打印命令。Hydra 覆盖项可加在脚本参数末尾。

脚本继承原脚本的 Python 环境和 TOS 凭据路径。若节点路径不同，设置 `OPENWAM_PYTHON`、`OPENWAM_TOS_CREDENTIALS`；两机均需可读。输出目录应供两机一致访问。

## 实现语义

训练中的前 d 个动作保持干净，其动作时间步设为 0；其余动作按原 flow matching 加噪。前缀仍参与联合注意力，动作损失仅监督有效后缀，继续使用原有 padding、有效维度和 scheduler 权重。视频分支沿用原训练流程。动作时间嵌入支持逐 token 时间步，模型参数名称及形状保持兼容。

相对 EEF 语义沿用现有设置：整个动作窗口均相对于本次输入的第一个原始绝对 state。RTC 前缀也使用同一参考系和动作归一化统计，输入 state 本身保持绝对状态后再归一化。

## 部署接口边界

RTC checkpoint 在加载时从保存的训练配置恢复可用前缀长度。服务端 PONG 提供 `rtc_prefix_lengths` 和 `rtc_prefix_format=absolute_eef20`。

使用整段动作响应模式，在 observation 中传入 `action_prefix`，格式为绝对物理 EEF20 的 `[d,20]` 数组；冷启动可以不传或传空列表。服务端根据该次原始 state 重新相对化，并使用动作统计归一化、映射到模型动作维度。模型每步去噪均固定前缀；输出恢复到绝对 EEF20，已承诺前缀按输入原值返回。旧 checkpoint 或未训练长度会明确拒绝前缀请求。

Piper 外部异步执行器的前缀从该次观察 action[0] 对齐，包含观察到提交期间的命令历史及后续已承诺命令；state 必须来自同一观察快照。当前要求 playback=4、replan=0、alpha=.55、真实相位补偿及三种执行模式。d 与实际接管索引 k 分离，不能假设总从 d 接管或超过 d 就退出。详见 rtc_piper_inference.md。本次没有完成机器人客户端接线或实机验证。

RTC 生成禁用 DiT velocity cache。GPU、CUDA Graph 的变长前缀部署仍需验证；训练与云端推理是不同验收环节。

## 已完成验证

- RTC 核心与动作训练相关测试：65 passed。
- 服务端/Policy/推理引擎/归一化边界新增测试：10 passed，覆盖所有 8 个长度、统计量裁剪后的物理前缀恢复，以及旧 checkpoint 和未知长度拒绝。
- 兼容性回归：290 passed、3 skipped、1 failed。唯一失败为 `test_dual_system_joint_self_attn_compile_auto_sets_mot_loop`，同样可在未修改的原始 git HEAD 复现，属于原有 compile 调用次数断言问题。
- 启动脚本 `bash -n` 和 rank 0/1 dry-run 通过；`git diff --check` 通过。
- SSH 当前主机没有可用 GPU，未执行真实 GPU forward 或两机训练。

本次云端协议补充和回归：96 passed（RTC、部署边界、CLI、观测预处理与 request projection）；加入请求上下文回传、协议声明及非法 prefix 校验。
