import os
import tos
from concurrent.futures import ThreadPoolExecutor, as_completed

endpoint = "tos-cn-beijing.volces.com"
region = "cn-beijing"
bucket_name = "ml-platform-auto-created-required-2109640594-cn-beijing"

local_dir = "/vePFS/data/egodex"     # 本地要上传的目录
tos_prefix = "linglong/openwam/egodex/"          # 桶内前缀（结尾加 /，相当于目标目录）
max_workers = 5                     # 并发数
multipart_threshold = 5 * 1024 * 1024 * 1024   # 超过 5GB 用分片上传


def _collect_files(root: str):
    """递归收集所有文件，返回 (本地绝对路径, 相对路径) 列表"""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            abs_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
            files.append((abs_path, rel_path))
    return files

def _upload_one(client, abs_path: str, rel_path: str):
    key = tos_prefix + rel_path
    size = os.path.getsize(abs_path)
    try:
        if size >= multipart_threshold:
            # 大文件：分片+并发+断点续传
            client.upload_file(
                bucket_name,
                key=key,
                file_path=abs_path,
                part_size=8 * 1024 * 1024,
                task_num=3,
                enable_checkpoint=True,
            )
        else:
            client.put_object_from_file(bucket_name, key, abs_path)
        return (True, key, None)
    except Exception as e:
        return (False, key, str(e))

def main():
    # 凭据仅在显式运行脚本时从环境变量读取。
    ak = os.environ.get("TOS_ACCESS_KEY")
    sk = os.environ.get("TOS_SECRET_KEY")
    if not ak or not sk:
        raise SystemExit("请设置 TOS_ACCESS_KEY 和 TOS_SECRET_KEY 环境变量")
    client = tos.TosClientV2(ak, sk, endpoint, region)

    files = _collect_files(local_dir)
    print(f"共发现 {len(files)} 个文件，开始上传到 {tos_prefix}")

    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_upload_one, client, ap, rp) for ap, rp in files]
        for fut in as_completed(futures):
            ok, key, err = fut.result()
            if ok:
                print(f"  ✓ {key}")
            else:
                print(f"  ✗ {key}  失败: {err}")
                failures.append((key, err))

    if failures:
        print(f"\n完成，{len(files) - len(failures)} 成功，{len(failures)} 失败:")
        for k, e in failures:
            print(f"  - {k}: {e}")
    else:
        print(f"\n全部上传成功，共 {len(files)} 个文件")

if __name__ == "__main__":
    main()
