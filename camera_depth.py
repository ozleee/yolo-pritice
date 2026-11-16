"""YOLO26 segmentation + monocular depth, no per-object dimensions required."""
import argparse
import json
import math
import time


def object_depth(depth, mask, scale=1.0):
    import numpy as np
    if depth.shape != mask.shape:
        raise ValueError('深度图与分割掩码尺寸不一致')
    values = depth[mask & np.isfinite(depth) & (depth > 0)]
    return float(np.median(values)) * scale if values.size else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='0', help='摄像头编号、/dev/video0、视频或 RTSP')
    p.add_argument('--model', default='yolo26n-seg.pt')
    p.add_argument('--depth-model', default='yolo26n-depth.pt')
    p.add_argument('--device', default='cpu')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.4)
    p.add_argument('--depth-scale', type=float, default=0.25,
                   help='统一尺度修正：实测距离/原始预测距离；默认1，未校准')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--max-frames', type=int, default=0)
    a = p.parse_args()
    if not math.isfinite(a.depth_scale) or a.depth_scale <= 0:
        p.error('depth-scale 必须为有限正数')
    if min(a.width, a.height, a.imgsz) <= 0 or a.max_frames < 0 or not 0 < a.conf <= 1:
        p.error('尺寸必须为正，帧数不能为负，conf 必须在 (0,1]')
    import cv2
    import numpy as np
    from ultralytics import YOLO
    source = int(a.source) if a.source.isdecimal() else a.source
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'无法打开 {a.source}，请检查设备映射、权限及占用')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        print('加载分割和深度权重；首次运行需要联网下载。', flush=True)
        seg = YOLO(a.model, task='segment')
        depth_model = YOLO(a.depth_model, task='depth')
        count, last_print = 0, 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                if isinstance(source, int) or str(source).startswith('/dev/') or '://' in str(source):
                    raise RuntimeError('摄像头读取失败')
                break
            start = time.monotonic()
            # Both models must see the original, unannotated frame.
            result = seg.predict(frame, imgsz=a.imgsz, conf=a.conf,
                                 device=a.device, retina_masks=True, verbose=False)[0]
            depth_result = depth_model.predict(frame, imgsz=a.imgsz, device=a.device, verbose=False)[0]
            if depth_result.depth is None:
                raise RuntimeError('模型未返回深度，请检查 Ultralytics 和深度权重版本')
            depth = depth_result.depth.data.cpu().numpy()
            if depth.shape != frame.shape[:2]:
                raise RuntimeError('深度图尺寸与原图不一致')
            items = []
            if result.masks is not None:
                for box, mask in zip(result.boxes, result.masks.data):
                    mask = mask.cpu().numpy() > 0.5
                    # Erode boundary pixels to reduce background contamination.
                    mask = cv2.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
                    value = object_depth(depth, mask, a.depth_scale)
                    name = result.names[int(box.cls.item())]
                    x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().tolist())
                    items.append(dict(name=name, confidence=float(box.conf.item()),
                                      estimated_depth_m=value, depth_scale=a.depth_scale))
                    label = f'{name} ~{value*100:.1f}cm (est)' if value is not None else f'{name} N/A'
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),2)
                    cv2.putText(frame,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,0),2)
            now = time.monotonic()
            if now-last_print >= 1:
                print(json.dumps(items,ensure_ascii=False),flush=True)
                last_print = now
            cv2.putText(frame,f'MONOCULAR ESTIMATE | FPS {1/max(now-start,1e-6):.1f}',
                        (10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,200,255),2)
            count += 1
            if not a.headless:
                cv2.imshow('YOLO26 monocular depth - Q to quit',frame)
                if cv2.waitKey(1)&255 in (ord('q'),27):
                    break
                if cv2.getWindowProperty('YOLO26 monocular depth - Q to quit',cv2.WND_PROP_VISIBLE)<1:
                    break
            if a.max_frames and count >= a.max_frames:
                break
    finally:
        cap.release()
        if not a.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise SystemExit(f'错误：{e}') from e
