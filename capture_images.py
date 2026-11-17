"""Capture original camera frames for annotation: S/Space save, Q/Esc quit."""
import argparse
from datetime import datetime
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description='摄像头图片采集：S/空格保存，Q/Esc退出')
    parser.add_argument('--source', default='0', help='摄像头编号或设备路径')
    parser.add_argument('--output', type=Path, default=Path('datasets/desktop/raw'))
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--interval', type=float, default=0, help='自动采集间隔（秒），0为手动')
    parser.add_argument('--max-images', type=int, default=0, help='保存指定数量后退出，0为不限')
    parser.add_argument('--headless', action='store_true', help='无窗口运行，需配置 interval')
    args = parser.parse_args()
    import math
    if min(args.width, args.height) <= 0 or args.max_images < 0:
        parser.error('画面尺寸必须为正数，max-images不能为负数')
    if not math.isfinite(args.interval) or args.interval < 0:
        parser.error('interval必须为有限非负数')
    if args.headless and args.interval == 0:
        parser.error('无窗口模式必须设置 --interval，例如 --interval 2')
    import cv2

    args.output.mkdir(parents=True, exist_ok=True)
    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    count = 0
    window = 'Dataset capture | S/Space: save | Q: quit'
    try:
        if not cap.isOpened():
            raise RuntimeError(f'无法打开摄像头 {args.source}；检查设备权限、编号及占用')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f'保存目录：{args.output.resolve()}\nS/空格保存，Q/Esc退出；Ctrl+C也可退出。', flush=True)
        print('请改变物体位置、角度和光照，避免采集大量重复图片。', flush=True)
        next_save = time.monotonic() + args.interval
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('摄像头读取失败，请检查连接')
            now = time.monotonic()
            key = -1
            if not args.headless:
                preview = frame.copy()  # Never write annotations into training images.
                cv2.putText(preview, f'Saved: {count} | S/Space: save | Q: quit',
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 255
                if key in (ord('q'), ord('Q'), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if key in (ord('s'), ord('S'), 32) or (args.interval > 0 and now >= next_save):
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                path = args.output / f'desk_{stamp}_{time.time_ns()}.jpg'
                encoded, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not encoded:
                    raise RuntimeError('图片编码失败')
                # Exclusive creation prevents accidental overwriting of a previous capture.
                with path.open('xb') as file:
                    file.write(data.tobytes())
                count += 1
                print(f'[{count}] {path} ({frame.shape[1]}x{frame.shape[0]})', flush=True)
                next_save = time.monotonic() + args.interval
                if args.max_images and count >= args.max_images:
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f'本次共保存 {count} 张图片。', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise SystemExit(f'错误：{exc}') from exc
