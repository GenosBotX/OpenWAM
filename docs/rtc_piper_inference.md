# RTC 接入修订方案：按 playback=4 的现场日志确定边界

更新：2026-11-06。本版替代 V1/V2 的倍速和 d 起点建议。固定目标为 `replan=0, alpha=0.55, phase_align=true, phase_steps=0, playback=4`，支持 direct/substeps/segment。客户端已在 `lq/async_traj_smooth_rtc` 的 `b5bd998` 完成接入并通过离线测试。云端支持后缀感知 DiT cache；尚未加载完整 RTC checkpoint 做 GPU 精度/性能验收，未运行机械臂。

## 1. 前缀与 state 的准确含义

`action_prefix[d,20]` 是以本次 observation 的 action[0] 为起点、按同一动作索引网格排列的 d 条绝对 EEF20 命令。每臂 10D：xyz（米）、rot6d、绝对夹爪比例；左右臂位置/姿态各自使用训练时的基座坐标系。来源是已经 alpha 融合、且执行器承诺继续执行的命令计划，不能直接取上一次模型的原始预测。

若观察后立即提交请求，它通常就是当前块即将执行的 d 步。观察采集到提交之间已经执行的动作也属于该观察的前缀，必须从命令历史补齐，不能把整条前缀平移到请求发送时刻。prefix 是命令目标，不是未来实测 TCP。左臂即使不参与当前右臂任务，输入仍应维持完整双臂语义。

state 是同一个 observation 包的原始绝对状态快照；不能在云端生成或结果采用时另取“最新 state”作为基准。摄像头与反馈采样存在时间偏差，需保留各自时间戳；同一包不等于硬件严格同步。云端先保存原始 state，再将整条 prefix 相对此 state 重基准、用 action 统计归一化和映射到 80D；state 独立用 state 统计归一化。输出恢复绝对 EEF20。位置差在基座系表达，旋转 `R_delta = R_action @ R_state.T`，夹爪不作差。

训练使用统计范围裁剪。物理输出处恢复 prefix 原值只保证传输回显一致，不保证模型看到了超出统计范围的原始值；需要在客户端/联调中记录 prefix 裁剪频率，评估 OOD 命令。不要把精确回显误当作模型物理连续性证明。

## 2. 分支与本次云端工作

训练仓库：`115.190.155.12:/vePFS/liqi/OpenWAM-CUDAGraph`，新分支 `lq/training-rtc`，基于原 HEAD `4c9db99`。RTC 训练、八种长度均匀采样、逐动作 timestep、prefix loss mask、每轮去噪钳制及归一化适配全部保留。原 TOS 和其他实验改动不纳入 RTC 提交。

云端已有 `action_prefix` 输入贯通 server→policy→engine→generate。本次补充：

- PONG 声明 `rtc_protocol_version=1`、`rtc_conditioning=clean_prefix_zero_timestep`、`rtc_prefix_format=absolute_eef20`、`rtc_prefix_anchor=observation_state`、`rtc_prefix_index_origin=observation_action_zero`，以及八种训练长度。声明 action 统计含裁剪；chunk PONG 声明 horizon=32。
- 请求可携带 `request_id`、`plan_version`（字符串/整数）及 `observation_mono_ns`（非负整数），响应原样回传。服务端不拿客户端单调时钟和自身时钟相减。WebSocket 参数错误/推理异常附带已验证的请求上下文。
- 错误 prefix 类型、形状、非有限数、未训练长度/旧 checkpoint、错误前缀位姿作为 observation validation error 处理；拒绝给单步云端 action buffer 输入 prefix。
- 增加 `configs/deploy_piper_x_rtc.yaml` 功能参考配置：5 步去噪、chunk、云端 sync、action-only，DiT cache/FP8/compile/CUDA Graph 关闭。客户侧仍异步、playback=4。该配置用于正确性基线，不能视为满足下面已有优化服务的耗时。

请求示例：

```json
{
  "type": "obs",
  "request_id": 42,
  "plan_version": 17,
  "observation_mono_ns": 123456789,
  "images": {"head_camera": "<base64 PNG>"},
  "state": ["20 floats: observation absolute EEF20"],
  "action_prefix": ["d rows, each 20 floats: absolute EEF20"],
  "prompt": "<task>"
}
```

这是字段示意，state/prefix 必须替换为数值数组。d=0 用空列表或省略 prefix；已有不带上下文的客户端保持兼容。

验证：`tests/test_rtc.py tests/test_rtc_deploy_boundary.py tests/test_policy_server_cli.py tests/test_obs_preprocess.py tests/test_request_projection.py` 共 **96 passed**。覆盖所有 d、真实 server/policy/engine/normalizer 边界、连续请求更换 state/prefix/d、旧请求兼容及非法参数；模型测试使用小型真实网络/CPU，并非加载完整 checkpoint 的 GPU 验收。更改没有部署到正在运行的云端服务。优化路径需分别验证不同 d、prefix 值、state、prompt 不复用陈旧条件，再重测 RTC 延迟。

## 3. 现场日志结果

来源为机器人服务器 `/home/zhuoyu/liqi/piper-x-traj-smooth-worktree/openwam-piper-x/reports/`，2026-11-05 至 11-06。仅遍历 reports 顶层 timing JSONL，没有扫描图片/手动录制内容。以 `async_start` 内实际设置筛选，部分 `a0.99` 文件名实际是 alpha=.55。全部匹配 playback4、replan0、alpha.55、phase_align=true，phase_steps=0、horizon32。19 个运行、9,104 个非首请求；冷启动每个运行单独排除，缺失结果/未下发请求仍保留计数。

所有时间单位 ms，分位数按请求合并计算，并非控制变量实验。不是新 RTC 模型的性能测试。

|执行模式|运行数 / 非首请求|服务端 P50/P95|观测→结果 P50/P95|观测→记录接管 P50/P95|观测→首次成功 SDK 下发 P50/P95|
|---|---:|---:|---:|---:|---:|
|direct|2 / 621|101 / 189|205 / 283|217 / 297|231 / 310|
|substeps|6 / 3,482|99 / 159|195 / 255|211 / 296|225 / 307|
|segment|11 / 5,001|101 / 188|202 / 280|215 / 296|254 / 383|

`observation_mono_ns` 为客户端观察锚点，不是硬件曝光时间；观测→结果包含取图/状态准备、编码、worker、RPC/服务和主线程收到结果。服务 latency 与 RPC 差值不能简单当作纯网络耗时。各列 P95 不是同一批尾样本，不能相加。

|执行模式|现有接管索引 k P50/P95/P99|d12 覆盖观测→结果|d12 覆盖接管 k|未记录结果 / 未成功下发|
|---|---:|---:|---:|---:|
|direct|14 / 18 / 19|245/619 = 39.6%|97/619 = 15.7%|2 / 2|
|substeps|13 / 18 / 22|2,130/3,477 = 61.3%|926/3,467 = 26.7%|5 / 15|
|segment|13 / 18 / 19|2,326/4,992 = 46.6%|1,054/4,990 = 21.1%|9 / 16|

这些 k 是普通 async 已有真实相位补偿的采用索引，不是已经测出的 RTC prefix 长度。分母只包括有对应结果/成功采用的样本，因此不能把覆盖率等同于成功率。未下发可能来自结束中断、结果丢弃等，不统一归因于延迟。运行错误保存在 JSON：18 次 KeyboardInterrupt，1 次 TCP/FK consistency wait 超时，不能隐藏。

首请求服务耗时范围约 103ms–10.24s，编译/初始化长耗时应独立统计并先预热；不能混进稳态 d 估计，也不能删除这些启动记录后称模型无尾延迟。

segment 还有关键差异：配对计算“记录接管→首次实际下发”的中位数约 34ms、P95 127ms、最大445ms。代码先在 scheduler 采用结果，再在 sender 建立新 segment 并等待其发送时刻。因此现有日志的 takeover 并非命令已作用到机械臂的时刻。不能仅用 server_latency 或该 takeover 的 k 宣称覆盖了整个执行链路。RTC 要提前估计 sender 可下发边界、记录实际 activation；冻结不可打断的 segment/substep 后，在可行边界对齐，避免重复相位补偿和发送时临时换目标破坏限速。

可复现脚本 `latency/analyze_recent.py`，逐请求结果、来源绝对路径与 SHA256 在 `latency/recent_summary.json`；该 JSON 同时保留每个运行独立分位数和 bootstrap 样本。

## 4. playback=4 下的 d 与 k

训练动作网格原生 15Hz；playback4 的目标执行网格是60Hz，每个模型索引约16.67ms。sender 限速、反馈等待仍可能拉长端点完成时间，60Hz 不能当作机械臂实际固定完成频率。

|d|6|7|8|9|10|11|12|
|---|---:|---:|---:|---:|---:|---:|---:|
|四倍播放下名义覆盖 ms|100|116.7|133.3|150|166.7|183.3|200|

32步名义队列长度约533ms，索引31对应约517ms；索引原点和首次下发相位还影响边界。四倍播放不改变训练时数据的物理时间分布，动作压缩与原生15Hz训练之间的时序差异仍需实机验证。

首版建议稳态先设 **d=12**，d0用于冷启动及显式无前缀对照；同时保留每请求动态 d，集合仍为 `{0,6,...,12}`。在线使用当前运行内每个模式最近的完整“观测→可执行边界”样本估计 d（可配置 P80+一 tick、带滞回），受训练集合和可用旧命令数限制。当前 k 的 P80 为15–16，估计器大多数时候会饱和到12，必须记 `coverage_overflow`。不要承诺动态 d 能解决当前训练覆盖不足，也不要外推尚未存在的旧动作凑够12；不足最小非零6时使用明确的 d0重规划/恢复。

真实未来耗时事前未知：d 来自历史实测及当前 sender 状态的预测；结果到达后再用这次真实时间算 k。一条生成请求内不能事后把 d12 改成 d16；新请求才可改 d。本次先不扩大训练参数或发未经训练的 d13以上。

当前 checkpoint 可以先验证 `d12 + k动态接续`，但多数请求会是 k>d，属于有条件近似而非严格前缀覆盖。若需要大部分执行都满足严格条件续写，下一轮可考虑覆盖到22–24步（例如0和6..24）并评估剩余suffix只有8–10步、频繁重规划的代价；或加速完整推理/预处理、增加动作 horizon并重新训练。阈值应由新RTC优化服务的实际延迟重定，不直接把旧数据当RTC性能承诺。

## 5. 结果接续与现有设置

不启用旧 `rtc=True` 的 training-free VJP 或其 replan/alpha/segment 禁用规则。复用普通 async、单请求 worker、现有硬件检查和三类 sender。

- **k<d**：继续原承诺命令，suffix 安装在 d 后；不重复执行返回的 prefix。d之前的夹爪也不得提前替换。机器人可继续运动，但新观察的影响延后到 suffix。
- **k=d**：从 suffix 开始按同相位 alpha=.55 接续。
- **d<k<32**：按真实相位丢掉已过时生成点，检查剩余 horizon、接缝、姿态/夹爪、IK/限速后接续；不能仅因超过 d退出。实际[d,k)旧命令与模型生成[d,k)不同，连续性保证已减弱，须记录区间误差和接缝。
- **k>=32、剩余suffix不足或接缝不合格**：丢弃此结果，继续仍有效旧计划、尽快用新观察重发；旧计划耗尽才进入现有受控等待。不能为“永不停顿”构造无界外推。

采用 `alpha=.55` 的位置55%旧+45%新、旋转最短弧SLERP、suffix夹爪沿用同相位新目标；prefix和活动sender点冻结。下一请求从最终融合后队列取前缀。replan0继续新观察且worker空闲即提交。

direct 使用当前命令节拍；substeps保存 joint 子命令及夹爪阶段、FK得到命令 EEF；segment提供计划发送边界。历史、当前活动点和未来队列统一成以 observation 为原点的命令视图，再采样 d 个动作网格点；不能把 d 个逻辑EEF端点、d个关节子命令与d个墙钟tick混为一谈。采样时不凭空假设实测 TCP 线性运动，记录执行时间预测误差。

prefix 不足、计划版本失效、重连、人工接管、SDK部分成功、超时等应有明确恢复分支。相位补偿只能做一次；延迟估计要剔除由选择 d 本身导致的承诺等待，防止自增反馈。有限 horizon 无法保证无限网络断流时继续有效运动。

## 6. 客户端实现状态和验收

1. 客户端已贯通新 PONG、prefix 请求、上下文检查、d/k 分离、三类 sender 的接续及诊断日志。
2. 实现传入的是 alpha 融合后的已承诺逻辑 EEF 端点，尚未实现严格按墙钟时间重采样的物理命令带。substeps/segment 会拉长时间，不能将端点前缀等同于均匀物理时间前缀；这是当前适配的已知限制。
3. 离线注入延迟测试k<d/k=d/d<k<32/k>=32、活动substep/segment、prefix不足、版本失效与中断。先核对前缀采样和限速，再做shadow。
4. 在GPU验证同一RTC权重d0与d12、变化state/prefix/d及各优化开关；重测完整链路后再决定是否补训更长prefix/horizon。
5. 三种模式保持playback4、alpha.55等现有设置对比接缝、夹爪事件、停顿和任务成功率；实机运行由用户现场执行。

## 功能验证启动命令

在具有模型依赖和 GPU 的服务节点，设置 checkpoint 目录后启动：

    python scripts/deploy.py --config configs/deploy_piper_x_rtc.yaml --ckpt-dir /path/to/rtc_checkpoint --port 8848

/path/to/rtc_checkpoint 是占位路径，须包含对应 config.yaml、normalization_stats.npy、权重及所需 tokenizer/base model。该功能配置尚未进行完整模型 GPU 性能验收。现有已优化服务不能只换权重后就假定支持 RTC。

日志审计文件同步到仓库 analysis/rtc_inference_20261106/latency/（本地分析产物，不纳入代码提交）。


## 7. RTC 与 DiT velocity cache 的兼容

此前 `openwam/model/architectures/base.py` 在 RTC 分支直接执行 `dit_cache = None` 是适配阶段的保守实现，并非训练 RTC 的理论要求。本版移除此限制。CFG scale>1 原有禁用规则仍在；当前 5 步配置使用默认 CFG scale=1。

- 每个请求开始清空 cache，不跨请求复用速度。自适应模式先计算第 1、2 步；固定模式按指定步号计算（见第 9 节）。
- 请求内部同时缓存 video/action 的 flow velocity；只在视频预测和有效动作后缀预测都满足余弦阈值时跳过联合 DiT forward。动作比较排除冻结 prefix 和 80D 中未使用的通道，避免这些通道掩盖后缀变化。
- 每个去噪步骤仍把 prefix 钳制回原值，prefix action timestep 仍为零。复用速度不会解除 RTC 条件。
- 这是近似加速。余弦相似度不约束幅值；即使 prefix 保持不变，suffix 也可能偏离 cache-off。0.99 阈值不代表误差低于 1%。5 步最多可由两次真实 forward 加三次复用完成，但命中率和总耗时需要实测，不能把理论上限当实测值。

优化配置 `configs/deploy_piper_x_rtc_fp8_5step.yaml` 保留 FP8、compile、CUDA Graph，画布 384×320，cache threshold=0.99、max_skips=3。旧 `deploy_piper_x_rtc.yaml` 保留作无优化正确性参考。

4090 独立目录完整后，使用原 FP8 环境启动：

```bash
cd /home/zhuoyu/liqi/OpenWAM-rtc
export OPENWAM_REPO_ROOT="$PWD"
export OPENWAM_BASE_CHECKPOINT=/home/zhuoyu/piper-x/OpenWAM-Alpha-Real-RoboDojo-Piper-X/checkpoint_step_30000.safetensors
/home/zhuoyu/tangyuting/envs/openwam-fp8/bin/python -u scripts/deploy.py \
  --config configs/deploy_piper_x_rtc_fp8_5step.yaml \
  --ckpt-dir /home/zhuoyu/liqi/ckpt/1106_rtc_based_0020 \
  --ckpt-name checkpoint_step_14525.safetensors \
  --device cuda:0 --host 0.0.0.0 --port 8848
```

也可继续用原 `configs/deploy_fp8_5step.yaml` 加 `inference.height=384 inference.width=320`，无需新增 cache 开关。客户端命令及 `--trained-rtc` 不变。

## 8. 测缓存造成的输出偏差

新增 `scripts/eval_rtc_cache.py` 离线加载一个模型，不连接机械臂或 WebSocket 服务。对于同一记录的 observation/state/prefix、同一 checkpoint、同一 seed，分别执行 cache-off、cache-off-repeat、cache-on；重复与交替测量顺序，用 CUDA synchronize 覆盖异步 GPU 执行。FP8/compile/CUDA Graph 在三组保持一致，因此隔离的是 DiT cache 的增量影响，不是 BF16 与 FP8 的差异。

```bash
/home/zhuoyu/tangyuting/envs/openwam-fp8/bin/python -u scripts/eval_rtc_cache.py \
  --config configs/deploy_piper_x_rtc_fp8_5step.yaml \
  --ckpt-dir /home/zhuoyu/liqi/ckpt/1106_rtc_based_0020 \
  --ckpt-name checkpoint_step_14525.safetensors \
  --request-json /path/to/rtc_inputs/000123/request.json \
  --prefix-lengths 0 6 7 8 9 10 11 12 \
  --seeds 42 43 44 --repeats 3 --warmup 1 \
  --cosine-threshold 0.99 --max-skips 3 \
  --output-dir analysis/rtc_cache_replay_001
```

请求路径应替换为客户端 `--record-dir` 保存的真实 `request.json`；遍历全部八种 d 时，所选请求必须已记录至少 12 条合法承诺 prefix。程序只能截短，不补造未来命令。不指定 `--prefix-lengths` 则保持记录的 d；旧非 RTC 记录只可测 d=0。应选不同任务阶段/运动幅度/夹爪事件的多个请求（`--request-json` 支持多个文件），不可只用一个静止样本推断整体精度。

输出 `manifest.json` 记录代码版本、配置、硬件；`results.jsonl` 和 `summary.json` 报告：

- 同种子缓存开关的双臂 suffix TCP 位置误差（mm）、旋转测地误差（度）、夹爪比例误差和 0.5 阈值翻转率；单独报告首个 suffix 点和逐步位置增量偏差。
- cache-off 重复执行的偏差，用以识别计算非确定性。
- 推理引擎端 P50/P95、配对加速比、跳步率；不含网络、客户端和执行器等待，不可当作端到端延迟。
- 每组 NPZ 保存 off/off_repeat/on 的完整动作；物理 prefix 必须精确一致，否则中止报告。

这里测到的是相对无缓存输出的数值变化，不是有真值的预测 loss，也不是任务成功率。后者应保持 playback4、alpha.55、phase-align 等参数一致，现场比较回撤/抖动、接缝、任务成功率和完整延迟。若 0.99/3 偏差过大，测 0.995/2、0.999/1 等更保守的缓存配置并比较实测速度，再选可接受的折中，不默认牺牲全部缓存加速。

客户端 P80 从当前进程当前执行阶段在线累计的最近 64 个有效延迟样本计算，重启清空；不自动读取以前日志。历史日志只用于设计参数起点。前缀长度仍受训练集合及现有队列约束。

本次 CPU 回归：185 passed、1 skipped（需要 CUDA）、1 deselected。排除的 `test_dual_system_joint_self_attn_compile_auto_sets_mot_loop` 在未修改的 `8bdb571` 上也失败：它断言 torch.compile 调用一次，而已有代码对联合循环和 request-context KV 分别编译，共两次；不是此次 cache 改动引入。缓存测试覆盖全部八种 d、真实跳过 forward、每步 prefix 固定、跨请求清空，以及物理误差单位和配对测量隔离。这些检查不替代完整 checkpoint 的 GPU 验收。

## 9. 固定重算与自适应缓存切换（2026-11-10）

`optimization.dit_cache` 支持以下启动命令覆盖，均为云端配置；客户端无需增加参数。

```bash
# 固定重算：5 步去噪中仅第 1、3 步运行联合 DiT。
optimization.dit_cache.enabled=true \
optimization.dit_cache.mode=fixed \
'optimization.dit_cache.recompute_steps=[1,3]'

# 自适应：保留原来的余弦判据和连续跳步上限。
optimization.dit_cache.enabled=true \
optimization.dit_cache.mode=adaptive \
optimization.dit_cache.cosine_threshold=0.99 \
optimization.dit_cache.max_skips=3

# 关闭两种速度缓存；FP8、compile、CUDA Graph 由各自开关控制。
optimization.dit_cache.enabled=false
```

省略 `mode` 保持 `adaptive`，现有配置和命令兼容。优化 preset 也保留自适应默认值，需显式指定 `mode=fixed` 才切换。步号从 1 开始，与客户端执行的动作步和前缀 d 无关；`[1,3]` 的复用来源为 `[1,1,3,3,3]`。每步仍积分更新 video/action latent，使用各步自身的 sigma 间隔，RTC 前缀仍逐步钳制且 timestep 为零。固定模式忽略 `cosine_threshold`、`max_skips`，缺少必要缓存时仍强制 forward。

固定模式当前支持云端 `inference.denoise_mode=sync`。这不影响客户端 `--async`、三种执行模式或在线动态选择 d。步号必须为严格递增的正整数、含第 1 步、且不超过 `inference.denoise_steps`；启动和引擎请求覆盖均校验。每请求重置缓存及计数。启动日志输出 mode/步号；`WAM_PROFILE=1` 时引擎返回的 `dit_cache_stats` 也包含 mode、步号和实际跳步数（当前 WebSocket 响应未透传此字段）。

先前 A800 BF16 p16/d16 的 7 个请求、4 个种子配对回放：固定 `[1,3]` 相对 cache-off 的右臂平均位置偏差 9.27 mm，自适应 0.99/3 为 13.79 mm；平均引擎耗时分别 116.9/120.4 ms。固定方案仍有误差，也未证明解决衔接反转。该结果不等于 4090 FP8/compile 的端到端精度和时延验收。

新 p16 权重在 4090 的目录为 `/home/zhuoyu/liqi/ckpt/1106_rtc_p16_based_0020`，文件 `checkpoint_step_14525.safetensors`；与旧 p12 的 `1106_rtc_based_0020` 区分。训练集合为 `{0,6,...,16}`，客户端仍从服务端 PONG 读取支持长度，根据当前运行的实测延迟及可用队列动态选择 d，无需固定为 16。

本次接入验证：云端缓存/RTC/部署边界及 CLI/归一化/去噪调度共 269 项回归通过。A800 使用 p16 最终权重、真实记录请求及 seed=42，在 5 组输入（real_000001 d0/d6/d16、real_000002 d6、real_000311 d16）比较配置驱动的正式 fixed 模式、原离线 Fixed13 实现及已归档 fixed_13 输出，全部逐元素一致；每次 5 步中跳过 3 步，物理前缀精确保持。此验证使用 BF16、FP8/compile 关闭，未占用 4090 或替换其运行服务。客户端 `b5215ff` 的 RTC/异步调度回归 75 项通过，4090 已同步该版本，长度专测 24 项通过。
