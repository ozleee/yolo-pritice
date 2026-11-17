"""Generate editable YOLO segmentation prelabels without modifying source images."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('datasets/desktop/raw'))
    p.add_argument('--output', type=Path, default=Path('datasets/desktop/prelabels'))
    p.add_argument('--model', default='yolo26n-seg.pt')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, default=0, help='0 表示处理全部图片')
    a = p.parse_args()
    if not 0 < a.conf <= 1 or a.limit < 0:
        p.error('conf 必须在 (0,1]，limit 不能为负')
    if not a.source.is_dir():
        p.error(f'图片目录不存在：{a.source}')
    files = sorted(f for f in a.source.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'})
    if a.limit:
        files = files[:a.limit]
    if not files:
        p.error('没有找到图片')
    if len({f.stem for f in files}) != len(files):
        p.error('存在同名不同扩展名的图片，请先重命名，防止标签冲突')
    if a.output.exists():
        p.error('输出目录已存在，请使用新的 --output，避免覆盖人工修正的标签')
    import cv2
    from ultralytics import YOLO
    model = YOLO(a.model, task='segment')
    for folder in ['images/train','labels/train','previews']:
        (a.output/folder).mkdir(parents=True, exist_ok=True)
    # Preserve the original model's class IDs. Do not renumber independently in the editor.
    names = dict(sorted(model.names.items()))
    yaml = 'path: '+json.dumps(str(a.output.resolve()))+'\ntrain: images/train\nnames:\n'
    yaml += ''.join(f'  {i}: {json.dumps(name)}\n' for i,name in names.items())
    (a.output/'data.yaml').write_text(yaml,encoding='utf-8')
    report, totals = [], Counter()
    for index, file in enumerate(files,1):
        frame = cv2.imread(str(file))
        if frame is None:
            raise RuntimeError(f'无法读取图片：{file}')
        result = model.predict(frame,conf=a.conf,device=a.device,retina_masks=True,verbose=False)[0]
        lines, detected = [], []
        if result.masks is not None:
            for box, polygon in zip(result.boxes,result.masks.xyn):
                if len(polygon) < 3:
                    continue
                cls = int(box.cls.item())
                coords = polygon.clip(0,1).reshape(-1)
                lines.append(str(cls)+' '+' '.join(f'{float(v):.6f}' for v in coords))
                detected.append(dict(class_id=cls,name=names[cls],confidence=float(box.conf.item())))
                totals[names[cls]] += 1
        shutil.copy2(file,a.output/'images/train'/file.name)
        (a.output/'labels/train'/f'{file.stem}.txt').write_text('\n'.join(lines)+ ('\n' if lines else ''),encoding='utf-8')
        # Previews are for review only, never for training.
        ok = cv2.imwrite(str(a.output/'previews'/f'{file.stem}.jpg'),result.plot())
        if not ok:
            raise RuntimeError(f'预览图保存失败：{file}')
        report.append(dict(image=file.name,objects=detected,requires_manual_review=True))
        print(f'[{index}/{len(files)}] {file.name}: {len(lines)} 个预标注',flush=True)
    (a.output/'report.json').write_text(json.dumps(dict(counts=dict(totals),images=report),ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'REVIEW_REQUIRED.txt').write_text(
        '自动预标注，尚不可直接用于正式训练。必须逐张补全漏标、修正错标和轮廓。\n'
        '空标签不表示图片没有目标，可能是模型漏检。\n'
        'data.yaml 保留原模型类别编号；添加新类别时保持编号一致。\n'
        '全部图片暂放 train，仅用于导入标注工具。审核后另划分 train/val/test。\n'
        'previews 只供检查，不要上传作训练图片。原始图片副本位于 images/train。\n',encoding='utf-8')
    print(f'完成：{a.output.resolve()}\n类别数量统计：{dict(totals)}',flush=True)


if __name__ == '__main__':
    main()
