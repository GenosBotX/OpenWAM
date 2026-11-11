# OpenWAM 0050 RTC

模型代码基于 `ddea7df86c11e0f6f3f40e7d2b9fccda4451a388`；本发版增加部署 YAML、独立启动脚本及页面配置说明，不更改模型算法或训练设置。客户端配套 Piper `release/0050`。

## 对应 OpenWAM 模型设置

配套使用 OpenWAM `release/0050` 的完整 commit；源码基于 `ddea7df86c11e0f6f3f40e7d2b9fccda4451a388`。模型页面目前使用平台固定参数，**填写自定义模型 commit 不会加载仓库的模型 JSON 默认值**，更新 commit 后须填写下表。

| 模型页面字段 | 值 |
| --- | --- |
| Python | `/home/zhuoyu/tangyuting/envs/openwam-fp8/bin/python` |
| 部署配置 | `configs/deploy_release_0050_rtc_fp8_5step.yaml` |
| Checkpoint 目录 | `/home/zhuoyu/liqi/ckpt/1111_rtc_p16_based_0030` |
| Checkpoint 文件 | `checkpoint_step_18495.safetensors` |
| 基础权重 | `/home/zhuoyu/piper-x/OpenWAM-Alpha-Real-RoboDojo-Piper-X/checkpoint_step_30000.safetensors` |
| device / host / port | cuda:0 / 0.0.0.0 / 8848 |
| height / width / denoise_steps | 384 / 320 / 5 |
| action_response_mode / denoise_mode / inference_mode | chunk / sync / sync |
| num_frames / video_num_frames | 33 / 9 |
| FP8 / DiT cache / compile / CUDA Graph | 全部开启 |
| 用表单覆盖全部模型高级默认值 | 关闭 |
| 其他 OmegaConf 覆盖 | `{"optimization.dit_cache.mode":"fixed","optimization.dit_cache.recompute_steps":[1,3]}` |

其他覆盖显式确认固定第 1、3 步计算；同样的默认值也已写入配套 YAML。固定模式下余弦阈值 0.99 和 max_skips=3 不决定重算计划。若切回 adaptive，在其他覆盖中将 mode 改成 `adaptive`；关闭 DiT cache 则五步全算。不要把客户端 async 与模型 inference_mode=sync 混淆：异步队列由客户端维护，模型每次返回完整动作块。

checkpoint 目录需含 `config.yaml`、`normalization_stats.npy` 和 tokenizer。该模型的训练前缀为 `{0,6,...,16}`，握手从 checkpoint 配置读取，无需服务端额外的 trained-rtc 启动参数。此说明不启动服务或下发真机动作。

独立终端等效启动：

```bash
bash scripts/piper/serve_release_0050_rtc.sh
```

默认 checkpoint 可通过 `RTC_CHECKPOINT_DIR` 覆盖，Python 可通过 `RTC_MODEL_PYTHON` 覆盖，附加参数原样传入 deploy.py。脚本不会停止已有服务；启动前确认 8848 端口可用。
