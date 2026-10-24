"""将已有 Piper-X parquet/JPEG 数据集直接发布至 TOS，不创建本地数据副本。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from scripts_piper.publish_piper_x_tos import Publisher, json_bytes, retry


def publish(source, destination, normalization, workers):
    publisher=Publisher(destination,'/tmp/piper-tos-20261022',workers)
    if publisher.get('publication.json'):
        print('已有完整发布，跳过');return
    def upload_checked(pair):
        relative,path=pair
        data=path.read_bytes()
        if path.suffix.lower() in ('.jpg','.jpeg'):
            with Image.open(io.BytesIO(data)) as im:
                if im.mode!='RGB':raise ValueError(f'Non-RGB image: {relative}')
                im.verify()
        digest=hashlib.md5(data).hexdigest()
        out=retry(lambda:publisher.client.put_object(publisher.bucket,publisher.prefix+'/'+relative,content=data))
        if out.etag.strip('"')!=digest:raise IOError(f'ETag mismatch: {relative}')
        return {'key':relative,'bytes':len(data),'md5':digest}
    summaries=[]
    for split in ('train','test'):
        root=Path(source)/f'piper_x_openwam_{split}_15hz'
        manifest=json.loads((root/'manifest.json').read_text())
        assert manifest['split']==split and manifest['source_root']=='.'
        episodes=[json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
        assert len(episodes)=={'train':49,'test':6}[split]
        for i,episode in enumerate(episodes):
            marker=f'episodes/{split}_{i:06d}.json'
            old=publisher.get(marker)
            if old:
                summaries.append(json.loads(old));continue
            relative=f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet'
            path=root/relative
            table=pq.read_table(path)
            assert len(table)==episode['length']
            jobs=[(split+'/'+relative,path)]
            for camera in ('head','left','right'):
                paths=table[f'observation.images.{camera}'].to_pylist()
                assert len(set(paths))==len(table)
                for rel in paths:
                    image=root/rel
                    if Path(rel).is_absolute() or '..' in Path(rel).parts:raise ValueError('invalid image path')
                    jobs.append((split+'/'+rel,image))
            with ThreadPoolExecutor(max_workers=workers) as pool:objects=list(pool.map(upload_checked,jobs))
            result={'split':split,'episode_index':i,'episode_id':episode.get('episode_id'),
                    'frames':len(table),'objects':len(objects),'bytes':sum(o['bytes'] for o in objects)}
            publisher.put(f'objects/{split}_{i:06d}.json',json_bytes(objects))
            publisher.put(marker,json_bytes(result));summaries.append(result)
            print(json.dumps(result),flush=True)
        for name in ('episodes.jsonl','episodes_stats.jsonl','tasks.jsonl','info.json'):
            publisher.put(f'{split}/meta/{name}',(root/'meta'/name).read_bytes())
        publisher.put(f'{split}/source_mapping.json',(root/'source_mapping.json').read_bytes())
        # 归一化指向两日合并 train 的独立统计，不使用本地旧统计。
        manifest.update({'normalization':'20261022 + 20261023 train frames once; test excluded',
                         'normalization_source':normalization.rstrip('/'),'eef20_stats':normalization.rstrip('/')+'/meta/eef20_normalization_stats.json',
                         'tos_immutable':True})
        publisher.put(f'{split}/manifest.json',json_bytes(manifest))
    report={'status':'complete','destination':destination,
            'objects':sum(r['objects'] for r in summaries),'bytes':sum(r['bytes'] for r in summaries),
            'normalization_source':normalization,'source':str(source)}
    publisher.put('publication.json',json_bytes(report))
    Path('/tmp/piper-tos-20261022/publication.json').write_bytes(json_bytes(report))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--normalization',required=True)
    p.add_argument('--workers',type=int,default=64)
    a=p.parse_args();publish(a.source,a.destination,a.normalization,a.workers)
