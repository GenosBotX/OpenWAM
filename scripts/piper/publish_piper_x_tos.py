"""将 TOS 中的 Piper-X TAR 转为按 episode 划分的独立 parquet/JPEG 对象。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from openwam.dataloader.tos_cache import TosBackend, parse_tos_uri
from openwam.dataloader.utils.eef import quat_wxyz_to_rot6d
from scripts.piper.convert_piper_x import build_episode, _fixed_list, _contracts, _sha256, _stats
from scripts.piper.kinematics import PiperXKinematics

PROMPT = 'Use the right hand to continuously place the objects on the table into the container on the table.'
BASELINE = Path('/vePFS/data/piper-x_openwam/20261022_ax_only_full/piper_x_openwam_train_15hz')
ROT_DIMS = [3,4,5,6,7,8,13,14,15,16,17,18]


def pack_eef20(values):
    values = np.asarray(values,dtype=np.float64).copy()
    result=[]
    for start in (0,8):
        q=values[:,start+3:start+7]
        q /= np.linalg.norm(q,axis=-1,keepdims=True)
        result.extend([values[:,start:start+3],quat_wxyz_to_rot6d(q),values[:,start+7:start+8]])
    return np.concatenate(result,axis=-1).astype(np.float32)


def vector_stats(values):
    values=np.asarray(values,dtype=np.float64)
    stats={'mean':values.mean(0),'std':values.std(0,ddof=1),'min':values.min(0),
           'max':values.max(0),'q01':np.quantile(values,.01,axis=0),'q99':np.quantile(values,.99,axis=0)}
    for name,identity in {'min':-1,'max':1,'q01':-1,'q99':1,'mean':0,'std':1}.items():
        stats[name]=stats[name].astype(np.float32)
        stats[name][ROT_DIMS]=identity
    return {k:v.tolist() for k,v in stats.items()}


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()


def sanitize_gripper_endstop(segment):
    """Clamp only sub-millimetre closed-end encoder noise; reject real range errors."""
    path = segment / 'samples.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    changed = 0
    for row in rows:
        for side in ('left', 'right'):
            fields = (row['arms'][side]['gripper'], row['actions'][side])
            for field in fields:
                key = 'width' if 'width' in field else 'position'
                value = float(field[key] if key == 'width' else field[key][6])
                if value < -0.001 or value > 0.101:
                    raise ValueError(f'gripper range violation: {side}={value}')
                if value < 0:
                    if key == 'width':
                        field[key] = 0.0
                    else:
                        field[key][6] = 0.0
                    changed += 1
    if changed:
        path.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows))
    return rows, changed


def retry(function):
    for attempt in range(5):
        try: return function()
        except Exception:
            if attempt==4: raise
            time.sleep(min(8,2**attempt))


class Publisher:
    def __init__(self,destination,work,workers,fk_mismatch_policy="error"):
        self.client=TosBackend().client
        self.bucket,self.prefix=parse_tos_uri(destination)
        self.destination=destination.rstrip('/')
        self.work=Path(work).resolve()
        if not str(self.work).startswith('/tmp/'):
            raise ValueError('中间数据目录必须位于 /tmp')
        self.work.mkdir(exist_ok=True,parents=True)
        self.workers=workers
        self.fk_mismatch_policy=fk_mismatch_policy

    def put(self,relative,data):
        retry(lambda:self.client.put_object(self.bucket,self.prefix+'/'+relative,content=data))

    def get(self,relative):
        try: return self.client.get_object(self.bucket,self.prefix+'/'+relative).read()
        except Exception as e:
            if getattr(e,'status_code',None)==404: return None
            raise

    def upload(self,pair):
        relative,path=pair
        digest=hashlib.md5(path.read_bytes()).hexdigest()
        out=retry(lambda:self.client.put_object_from_file(self.bucket,self.prefix+'/'+relative,str(path)))
        if out.etag.strip('"') != digest:
            raise IOError(f'上传 ETag 不匹配: {relative}')
        return {'key':relative,'bytes':path.stat().st_size,'md5':digest}

    def download(self,key,path,source_bucket):
        # 原始 TAR 暂存 /tmp；按块写入，不在内存中持有整个 archive。
        head=self.client.head_object(source_bucket,key)
        if path.exists() and path.stat().st_size==head.content_length: return
        if shutil.disk_usage(self.work).free<head.content_length+1024**3:
            raise RuntimeError('/tmp 空间不足')
        partial=path.with_suffix('.partial')
        def once():
            out=self.client.get_object(source_bucket,key,if_match=head.etag)
            with partial.open('wb') as f:
                while data:=out.read(4*1024**2): f.write(data)
            if partial.stat().st_size != head.content_length: raise IOError('TAR 下载不完整')
        retry(once)
        partial.replace(path)

    def run(self,inventory,baseline):
        inventory=json.loads(Path(inventory).read_text())
        entries=inventory['episodes']
        if len(entries)!=52 or inventory['counts']!={'train':47,'test':5}:
            raise ValueError('预期 52 episodes，47 train / 5 test')
        if len({x['episode'] for x in entries})!=len(entries) or len({x['sha256'] for x in entries})!=len(entries):
            raise ValueError('重复 episode 或 samples')
        plan=json_bytes(inventory)
        old=self.get('split.json')
        if old is not None and old!=plan: raise ValueError('目标目录已有不同划分')
        if self.get('publication.json') is not None: raise ValueError('目标目录已完成发布')
        self.put('split.json',plan)
        fk=PiperXKinematics()
        stats16={name:(baseline/'meta'/name).read_bytes() for name in
                 ['normalization_stats.json','action_eef_stats.json','state_eef_stats.json']}
        base_episodes=[json.loads(x) for x in (baseline/'meta/episodes.jsonl').read_text().splitlines()]
        base_arrays={name:[] for name in ['action_eef','observation.state_eef']}
        for i in range(len(base_episodes)):
            table=pq.read_table(baseline/f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet',columns=list(base_arrays))
            for name in base_arrays: base_arrays[name].append(np.asarray(table[name].to_pylist(),dtype=np.float32))
        stats20={name:vector_stats(pack_eef20(np.concatenate(rows))) for name,rows in base_arrays.items()}
        payload20={'format':'piper_x_eef20_normalization_v1','source':str(baseline),
                   'num_episodes':len(base_episodes),'num_frames':sum(r['length'] for r in base_episodes),
                   'rotation_normalization':'identity','action':stats20['action_eef'],'state':stats20['observation.state_eef']}
        base_manifest=json.loads((baseline/'manifest.json').read_text())
        summaries=[]
        for archive in inventory['archives']:
            group=[x for x in entries if x['archive']==archive['key']]
            source_bucket=self.bucket
            path=self.work/Path(archive['key']).name
            if all(self.get(f"episodes/{r['split']}_{r['episode_index']:06d}.json") is not None for r in group):
                summaries.extend(json.loads(self.get(f"episodes/{r['split']}_{r['episode_index']:06d}.json")) for r in group)
                continue
            self.download(archive['key'],path,source_bucket)
            if _sha256(path)!=archive['sha256']: raise IOError('TAR SHA256 不匹配')
            with tarfile.open(path) as tar:
                members=tar.getmembers()
                for entry in sorted(group,key=lambda r:r['folder']):
                    split,index=entry['split'],entry['episode_index']
                    marker=f'episodes/{split}_{index:06d}.json'
                    previous=self.get(marker)
                    if previous:
                        summaries.append(json.loads(previous));continue
                    stage=self.work/'episode'
                    shutil.rmtree(stage,ignore_errors=True)
                    segment=stage/PurePosixPath(entry['folder']).name
                    segment.mkdir(parents=True)
                    for member in members:
                        parent=PurePosixPath(entry['folder'])
                        p=PurePosixPath(member.name)
                        if parent not in p.parents:continue
                        relative=p.relative_to(parent)
                        if relative.is_absolute() or '..' in relative.parts or not member.isfile():
                            raise ValueError('不安全 TAR member')
                        dest=segment/str(relative)
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        with tar.extractfile(member) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
                    original_sha256=_sha256(segment/'samples.jsonl')
                    if original_sha256!=entry['sha256']:raise ValueError('samples SHA256 不匹配')
                    rows,gripper_endstop_clamps=sanitize_gripper_endstop(segment)
                    record=build_episode(segment,fk,PROMPT,{'prompt':PROMPT,'split':split}, fk_mismatch_policy=self.fk_mismatch_policy)
                    origins={side:{} for side in ['left','right']}
                    for row in rows:
                        for side in origins:
                            source=row.get('action_sources',{}).get(side,{})
                            text=json.dumps({k:v.get('origin') if isinstance(v,dict) else v for k,v in source.items()},sort_keys=True)
                            origins[side][text]=origins[side].get(text,0)+1
                    n=record['frames']
                    image_paths={}
                    jobs=[]
                    for camera,sources in record['image_sources'].items():
                        paths=[]
                        for sample,src in zip(record['sample_ids'],sources):
                            with Image.open(src) as im:
                                if im.mode!='RGB':raise ValueError(f'非 RGB 图像: {src}')
                                im.verify()
                            relative=f'images/episode_{index:06d}/{camera}/{sample}.jpg'
                            paths.append(relative);jobs.append((split+'/'+relative,src))
                        if len(set(paths))!=n:raise ValueError('重复图像 sample_id')
                        image_paths[camera]=paths
                    offset=sum(r['frames'] for r in entries if r['split']==split and r['episode_index']<index)
                    values={'observation.state_eef':_fixed_list(record['state'],16),'action_eef':_fixed_list(record['action'],16),
                            'timestamp':pa.array(np.arange(n,dtype=np.float32)/15),'frame_index':pa.array(np.arange(n,dtype=np.int64)),
                            'episode_index':pa.array(np.full(n,index,dtype=np.int64)),'index':pa.array(np.arange(offset,offset+n,dtype=np.int64)),
                            'task_index':pa.array(np.zeros(n,dtype=np.int64)),
                            **{f'observation.images.{camera}':pa.array(paths) for camera,paths in image_paths.items()}}
                    parquet=stage/'episode.parquet'
                    pq.write_table(pa.table(values),parquet,compression='zstd')
                    jobs.append((f'{split}/data/chunk-{index//1000:03d}/episode_{index:06d}.parquet',parquet))
                    outside={}
                    for name,data in [('state',record['state']),('action',record['action'])]:
                        packed=pack_eef20(data)
                        lo=np.array(payload20[name]['min']);hi=np.array(payload20[name]['max'])
                        outside[name]=np.mean((packed<lo-1e-6)|(packed>hi+1e-6),axis=0).tolist()
                    with ThreadPoolExecutor(max_workers=self.workers) as pool:
                        objects=list(pool.map(self.upload,jobs))
                    summary={'episode':entry['episode'],'split':split,'episode_index':index,'sha256':entry['sha256'],
                             'frames':n,'windows':n-32,'fk_report':record['fk_report'],'action_sources':origins,
                             'timestamp_gap_over_100ms':int(np.sum(np.diff([r['timestamp_ns'] for r in rows])>100000000)),
                             'gripper_endstop_clamps':gripper_endstop_clamps,
                             'outside_baseline_minmax_fraction_per_dim':outside,'objects':len(objects),
                             'bytes':sum(x['bytes'] for x in objects),'archive':archive['key']}
                    self.put(f'objects/{split}_{index:06d}.json',json_bytes(objects))
                    self.put(marker,json_bytes(summary))
                    summaries.append(summary)
                    (self.work/'progress.json').write_bytes(json_bytes(summaries))
                    print(json.dumps({k:summary[k] for k in ['episode','split','frames','objects','bytes']}),flush=True)
                    shutil.rmtree(stage)
            path.unlink()
        if len(summaries)!=52:raise ValueError('episode 发布不完整')
        # 每个训练 episode 的每帧只计一次；绝不读取今天的 test parquet。
        train_entries=sorted([r for r in entries if r['split']=='train'],key=lambda r:r['episode_index'])
        for entry in train_entries:
            index=entry['episode_index']
            relative=f'train/data/chunk-{index//1000:03d}/episode_{index:06d}.parquet'
            table=pq.read_table(pa.BufferReader(self.get(relative)),columns=list(base_arrays))
            for name in base_arrays:
                base_arrays[name].append(np.asarray(table[name].to_pylist(),dtype=np.float32))
        num_train=len(base_episodes)+len(train_entries)
        total_train_frames=sum(len(x) for x in base_arrays['action_eef'])
        if num_train!=96 or total_train_frames!=116849:
            raise ValueError(f'合并训练统计来源不完整: {num_train} episodes, {total_train_frames} frames')
        stats16={'normalization_stats.json':json_bytes({
            stream:_stats(base_arrays[column]) for stream,column in
            [('action','action_eef'),('state','observation.state_eef')]})}
        for stream,column in [('action','action_eef'),('state','observation.state_eef')]:
            values=_stats(base_arrays[column],ddof=1)
            stats16[f'{stream}_eef_stats.json']=json_bytes({
                'source_column':column,'statistics_source':'20261022 + 20261023 train frames once; test excluded',
                'num_episodes':num_train,'num_transition':values['count'],'std_ddof':1,
                stream:{'default':{f'global_{key}':value for key,value in values.items() if key!='count'}}})
        payload20={
            'format':'piper_x_eef20_normalization_v1',
            'sources':[str(baseline),self.destination+'/train'],
            'num_episodes':num_train,'num_frames':total_train_frames,
            'rotation_normalization':'identity','test_excluded':True,
            **{stream:vector_stats(pack_eef20(np.concatenate(base_arrays[column]))) for stream,column in
               [('action','action_eef'),('state','observation.state_eef')]}}
        self.put('normalization/meta/eef20_normalization_stats.json',json_bytes(payload20))
        for name,data in stats16.items():self.put('normalization/meta/'+name,data)
        print(json.dumps({'normalization_train_episodes':num_train,'normalization_train_frames':total_train_frames}),flush=True)
        for split in ['train','test']:
            selected=sorted([x for x in summaries if x['split']==split],key=lambda r:r['episode_index'])
            if [r['episode_index'] for r in selected]!=list(range(len(selected))):raise ValueError('episode index 不连续')
            episodes=[{'episode_index':r['episode_index'],'episode_id':r['episode'],'length':r['frames'],'tasks':[PROMPT],'task_index':0} for r in selected]
            frames=sum(r['frames'] for r in selected)
            for name,data in stats16.items():self.put(f'{split}/meta/{name}',data)
            self.put(f'{split}/meta/eef20_normalization_stats.json',json_bytes(payload20))
            self.put(f'{split}/meta/episodes.jsonl',''.join(json.dumps(r)+'\n' for r in episodes).encode())
            self.put(f'{split}/meta/tasks.jsonl',(json.dumps({'task_index':0,'task':PROMPT})+'\n').encode())
            self.put(f'{split}/source_mapping.json',json_bytes(selected))
            features={name:{'dtype':'float32','shape':[16]} for name in ['observation.state_eef','action_eef']}
            features.update({f'observation.images.{c}':{'dtype':'image'} for c in ['head','left','right']})
            self.put(f'{split}/meta/info.json',json_bytes({'codebase_version':'openwam_piper_x_eef_v1','robot_type':'piper_x',
                     'total_episodes':len(selected),'total_frames':frames,'total_tasks':1,'fps':15,'splits':{split:f'0:{len(selected)}'},
                     'features':features,'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'}))
            manifest={'format':'openwam_piper_x_eef_v1','status':'complete','split':split,'source_root':'.','prompt':PROMPT,
                      'fps':15,'num_frames':33,'video_stride':4,'total_episodes':len(selected),'total_frames':frames,
                      'total_windows':sum(r['windows'] for r in selected),'image_storage':'independent JPEG objects in this bucket',
                      'normalization':'20261022 + 20261023 train frames once; test excluded','normalization_source':self.destination+'/normalization',
                      'eef20_stats':'meta/eef20_normalization_stats.json','active_raw_action_dims':list(range(16)),
                      'active_raw_state_dims':list(range(16)),'single_episode_reused_for_test':False,
                      'split_seed':0,'test_fraction':0.1,'split_method':'seeded permutation of samples.jsonl SHA256, episode holdout',
                      'fk_mismatch_policy':self.fk_mismatch_policy,'quality_filter':False,'task_stratification':False,'urdf_sha256':_sha256(fk.urdf_path),
                      'tos_immutable':True,**_contracts(fk)}
            if manifest['urdf_sha256']!=base_manifest['urdf_sha256']:raise ValueError('URDF 与基线不一致')
            self.put(f'{split}/manifest.json',json_bytes(manifest))
        report={'status':'complete','destination':self.destination,'splits':{s:{'episodes':sum(r['split']==s for r in summaries),
                'frames':sum(r['frames'] for r in summaries if r['split']==s),'windows':sum(r['windows'] for r in summaries if r['split']==s)} for s in ['train','test']},
                'fk_exceptions':[{'episode':r['episode'],'split':r['split'],'side':side,**report} for r in summaries for side,report in r['fk_report'].items() if report['position_error_mm_max']>12 or report['orientation_error_deg_max']>4],
                'gripper_endstop_clamps':sum(r['gripper_endstop_clamps'] for r in summaries),
                'normalization':{'episodes':num_train,'frames':total_train_frames,'test_excluded':True,'root':self.destination+'/normalization'},
                'objects':sum(r['objects'] for r in summaries),'bytes':sum(r['bytes'] for r in summaries)}
        self.put('publication.json',json_bytes(report))
        (self.work/'publication.json').write_bytes(json_bytes(report))
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--work',default='/tmp/piper-tos-20261023')
    p.add_argument('--baseline',type=Path,default=BASELINE)
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--fk-mismatch-policy',choices=['error','record'],default='error',help='record 保留原始位姿并记录超阈值；默认 error 中止')
    a=p.parse_args()
    Publisher(a.destination,a.work,a.workers,a.fk_mismatch_policy).run(a.inventory,a.baseline)
