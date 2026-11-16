"""YOLO26 camera detection and calibrated apparent-width ranging."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('尺寸、距离和焦距必须是有限正数')
    return value


def distance_cm(focal_px, width_cm, pixel_width):
    return positive(focal_px) * positive(width_cm) / positive(pixel_width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='0', help='摄像头编号、视频路径或 RTSP URL')
    parser.add_argument('--model', default='yolo26n.pt')
    parser.add_argument('--config', default='camera_config.json')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--conf', type=float, default=0.4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default=None, help='cpu、0 或其他 Ultralytics device')
    parser.add_argument('--calibrate', metavar='CLASS', help='标定目标英文类别，例如 bottle')
    parser.add_argument('--object-width', type=float, help='标定物体实际正面宽度，cm')
    parser.add_argument('--distance', type=float, help='标定物体到镜头的光轴距离，cm')
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--headless', action='store_true', help='无窗口运行；每秒输出一次结果')
    parser.add_argument('--max-frames', type=int, default=0, help='0 表示持续运行')
    args = parser.parse_args()
    if args.samples < 1 or args.width < 1 or args.height < 1 or args.max_frames < 0:
        parser.error('samples、width、height 必须为正数，max-frames 不能为负数')
    if not 0 < args.conf <= 1:
        parser.error('conf 必须在 (0, 1] 内')
    if args.calibrate:
        if args.object_width is None or args.distance is None:
            parser.error('标定需要 --object-width 和 --distance')
        positive(args.object_width)
        positive(args.distance)
    import cv2
    from ultralytics import YOLO

    path = Path(args.config)
    config = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    widths = {k: positive(v) for k, v in config.get('object_widths_cm', {}).items()}
    focal = config.get('focal_length_px')
    if focal is not None:
        focal = positive(focal)
    model = YOLO(args.model)
    if args.calibrate and args.calibrate not in model.names.values():
        raise ValueError(f'模型没有类别 {args.calibrate}，可选类别：{model.names}')
    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'无法打开摄像头/视频：{args.source}。检查编号、权限和设备占用。')
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        samples = []
        count = 0
        last_print = 0.0
        actual_size = None
        while True:
            ok, frame = cap.read()
            if not ok:
                if isinstance(source, int) or '://' in args.source:
                    raise RuntimeError('摄像头读取失败或连接已断开')
                break
            started = time.monotonic()
            h, w = frame.shape[:2]
            if actual_size is None:
                actual_size = [w, h]
                print(f'实际画面：{w}x{h}；按 Q / Esc 退出。')
                if not args.calibrate and focal is not None:
                    if config.get('image_size') != actual_size:
                        raise ValueError('当前分辨率与标定不一致，请按当前分辨率重新标定')
            elif actual_size != [w, h]:
                raise ValueError('视频分辨率发生变化，请使用固定分辨率')
            options = dict(conf=args.conf, imgsz=args.imgsz, verbose=False)
            if args.device is not None:
                options['device'] = args.device
            result = model.predict(frame, **options)[0]
            detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                name = result.names[int(box.cls.item())]
                confidence = float(box.conf.item())
                # Clipped boxes do not represent the complete physical width.
                clipped = x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2
                z = None
                if not args.calibrate and focal and name in widths and not clipped and x2 > x1:
                    z = distance_cm(focal, widths[name], x2 - x1)
                detections.append(dict(name=name, confidence=confidence,
                                       xyxy=[x1, y1, x2, y2], distance_cm=z, clipped=clipped))
                label = f'{name} | ' + (f'Z~{z:.1f}cm' if z else 'distance N/A')
                color = (0, 220, 0) if z else (0, 190, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), max(20, int(y1)-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                confidence_y = min(h - 8, max(44, int(y1) + 22))
                cv2.putText(frame, f'Conf: {confidence:.1%}', (int(x1), confidence_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            if args.calibrate:
                targets = [d for d in detections if d['name'] == args.calibrate]
                if len(targets) == 1 and not targets[0]['clipped']:
                    x1, _, x2, _ = targets[0]['xyxy']
                    samples.append((x2-x1) * args.distance / args.object_width)
                else:
                    samples.clear()  # Require consecutive unambiguous frames.
                print(f'标定 {len(samples)}/{args.samples}：保持一个完整目标正对镜头', end='\r', flush=True)
                if len(samples) >= args.samples:
                    median = statistics.median(samples)
                    spread = (max(samples)-min(samples)) / median
                    if spread > 0.15:
                        print('\n检测框波动超过 15%，请固定相机和物体后重试。')
                        samples.clear()
                    else:
                        widths[args.calibrate] = args.object_width
                        config.update(focal_length_px=median, image_size=actual_size,
                                      object_widths_cm=widths)
                        path.write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
                        print(f'\n标定已保存：{path}，焦距 {median:.2f}px')
                        break
            now = time.monotonic()
            if not args.calibrate and now-last_print >= 1:
                print(json.dumps(detections, ensure_ascii=False))
                last_print = now
            cv2.putText(frame, f'FPS {1/max(now-started, 1e-6):.1f} | Q: quit', (12, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if not args.headless:
                cv2.imshow('YOLO26 - approximate optical-axis distance', frame)
                if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                    break
                if cv2.getWindowProperty('YOLO26 - approximate optical-axis distance', cv2.WND_PROP_VISIBLE) < 1:
                    break
            count += 1
            if args.max_frames and count >= args.max_frames:
                if args.calibrate:
                    raise RuntimeError('帧数已用完，标定未完成；配置未保存')
                break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise SystemExit(f'错误：{exc}') from exc
