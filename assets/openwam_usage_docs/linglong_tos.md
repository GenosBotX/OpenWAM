# Linglong TOS 数据使用

完整的数据语义、缓存行为、训练参数和 checkpoint 说明见 [Linglong EEF 数据读取与训练](linglong.md)。TOS 和本地训练共用 `LinglongDataset`，不改变 EEF 定义、归一化或采样方式。

## 当前训练数据与入口

完整训练前缀：

```text
tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/20261022_0923_1017_eef_v1
```

在配置好 1 worker × 8 GPU 和 vePFS 挂载的 Volcano 任务中执行：

```bash
cd /vePFS/tangyuting/OpenWAM-tos && \
bash scripts/launch_volcano_linglong_tos_eef_8gpu_bs6.sh
```

每卡 bs6、accum24、窗口步长 1、视频步长 4、5 epochs。288×256 输入改用 `scripts/launch_volcano_linglong_tos_eef_8gpu_bs6_h288_w256.sh`。

launcher 默认从 `/vePFS/tangyuting/.config/openwam/tos-linglong.json` 加载凭证；该文件需归任务用户所有、权限 0600。可用 `TOS_CREDENTIAL_FILE` 覆盖路径。文件包含 `TOS_ACCESS_KEY`、`TOS_SECRET_KEY`，以及可选的 `TOS_SECURITY_TOKEN`、`TOS_ENDPOINT`、`TOS_REGION`；不要将凭证写入仓库。

使用任务注入的凭证环境变量时，可直接运行通用入口：

```bash
export LINGLONG_TOS_DATASET=tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/20261022_0923_1017_eef_v1
NPROC_PER_NODE=8 bash scripts/train_linglong_tos.sh
```

## 缓存配置

对象按需下载到 `${OPENWAM_TOS_CACHE_DIR:-/tmp/openwam-tos-cache}`；同节点 rank/worker 共享 4 GiB 内容缓存，保留 1 GiB 空闲空间。缓存目录使用节点本地 Linux 磁盘，全部使用者须设置相同限额；修改限额请切换目录。

DataLoader 每 rank 8 个 worker，默认每 worker 提前准备 2 个 batch。预取与模型计算重叠，缓存不足时以 LRU 淘汰；没有全量本地副本或独立全量预取服务。磁盘限额不包含解码后图像、prefetch 队列和 parquet 的主机内存。

```bash
OPENWAM_TOS_CACHE_DIR=/tmp/openwam-tos-cache-8g \
bash scripts/launch_volcano_linglong_tos_eef_8gpu_bs6.sh \
  dataloader.cache_max_bytes=8589934592
```

## 发布数据

安装 `pip install -e '.[tos]'`，准备转换完成的 portable 桶，并选择尚未发布的新版本前缀：

```bash
cd /vePFS/tangyuting/OpenWAM-tos
/vePFS/tangyuting/envs/openwam/bin/python scripts/run_with_tos_credentials.py \
  /vePFS/tangyuting/.config/openwam/tos-linglong.json \
  /vePFS/tangyuting/envs/openwam/bin/python -m scripts.publish_linglong_tos \
  --source /path/to/converted_train_bucket \
  --destination tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/NEW_VERSION \
  --workers 8
```

发布器上传 parquet、引用的 JPEG、元数据及原训练统计，最后写入 `manifest.json`。失败且尚未写入 manifest 的前缀可重试；已有 manifest 的前缀禁止再次发布。发布后不要覆盖任何对象。test 桶使用独立前缀，训练统计始终来自 train；测试时显式配置 `dataloader.eval_dataset_dir`。

当前 checkpoint 输出到 vePFS 的 `training.output_path`，不自动同步 TOS。读取数据无需先把整个 TOS 数据集下载到本地。
