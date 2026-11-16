# YOLO26 桌面物体识别与单目测距

使用普通 USB 摄像头，显示类别、置信度、检测框、估计距离和处理 FPS。默认 YOLO26n；首次运行需要联网下载权重。支持本地 `.pt` 权重、摄像头编号、视频和 RTSP 输入。窗口使用英文标签，避免 OpenCV 中文字体乱码。

## 安装与运行

建议 Python 3.10+，在连接摄像头的电脑执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python camera_yolo.py --source 0
```

Windows 使用 `python` 创建环境，激活命令为 `.venv\Scripts\activate`。需要有图形桌面才能显示窗口；服务器加 `--headless`。按 Q / Esc 或关闭窗口退出，终端 Ctrl+C 也可退出。

首次运行尚未标定，距离显示 `distance N/A`。默认检测模型使用 COCO 类别，例如 bottle、cup、cell phone、mouse、keyboard、book、scissors；不能保证识别所有桌面物体。自定义物体需要相应训练权重，可用 `--model /path/to/best.pt` 指定。

## 标定并测距

1. 固定摄像头，关闭/锁定自动变焦，准备一个模型能稳定识别的物体，例如瓶子。尺量瓶子正面实际宽度，例如 **6.5 cm**。
2. 将瓶子放在镜头前，保持正对镜头、完整可见。沿相机光轴量镜头到瓶子可见正面的距离，例如 **50 cm**。画面中只放一个 bottle，保持静止。
3. 执行（数值必须替换为自己的实测值）：

```bash
python camera_yolo.py --source 0 --calibrate bottle --object-width 6.5 --distance 50
```

连续采集 30 帧，检测框宽度稳定后，将像素焦距、实际画面分辨率和该物体宽度保存至 `camera_config.json`。可用 `--samples 60` 增加采样数。中途退出不会保存新标定。

4. 再次运行：

```bash
python camera_yolo.py --source 0
```

瓶子框上会出现 `Z~50.0cm` 形式的近似距离。拿同一个瓶子在 30、50、80 cm 等位置用尺复核误差。

5. 对其他物体测距，在配置文件的 `object_widths_cm` 中添加英文类别和实测正面宽度，例如：

```json
"object_widths_cm": {
  "bottle": 6.5,
  "cell phone": 7.2,
  "book": 14.8
}
```

这些只是格式示例，必须使用你自己的实测尺寸。一个类别只支持一种宽度；不同大小的瓶子同时出现会产生错误距离。宽度是当前朝向下与图像水平方向对应的正面尺寸，旋转手机、书本或斜放物体会显著影响结果。

## 距离定义与限制

采用针孔近似：`f_px = 检测框宽度_px × 已知距离_cm / 实际宽度_cm`，运行时 `Z_cm = f_px × 实际宽度_cm / 检测框宽度_px`。

输出是物体正面附近的**光轴深度 Z**，不是相机到物体中心的三维直线距离，也不是物体之间的桌面距离。检测框并非精密几何测量，遮挡、反光、畸变、倾斜、朝向和检测抖动都会引入误差；多个采样只降低随机波动，不能消除系统误差。建议物体靠近画面中央、正对镜头、无遮挡。

未配置宽度、未标定或框触碰边缘时不输出数值。切换摄像头、镜头焦距、变焦、裁剪方式或分辨率必须重新标定；程序会拒绝使用与当前分辨率不一致的标定。同分辨率切换镜头无法自动检测，请自行重新标定。

如果希望对任意未知大小物体都获得可靠厘米距离，需要深度摄像头（例如 RealSense）、经过标定的双目相机，或固定机位下结合内外参的桌面平面几何方案；本程序不会用类别平均尺寸冒充真实距离。

## 其他参数和排错

```bash
# 换摄像头
python camera_yolo.py --source 1
# NVIDIA GPU（需要与本机 CUDA 匹配的 PyTorch）；CPU 则 --device cpu
python camera_yolo.py --device 0
# 更大的检测模型
python camera_yolo.py --model yolo26s.pt
# 修改画面尺寸（需要按此尺寸重新标定）
python camera_yolo.py --width 640 --height 480
# 无窗口测试 100 帧，终端每秒输出检测结果 JSON
python camera_yolo.py --headless --max-frames 100
```

打不开摄像头时检查系统相机权限、摄像头是否被其他应用占用以及编号。Linux 可用 `ls /dev/video*` 查看设备，容器/WSL/远程环境需先映射或转发摄像头。画面卡顿时减小 `--imgsz 416` 或使用 GPU。若提示模型不支持，请更新 `python -m pip install -U ultralytics`。离线环境先下载 YOLO26 权重，再通过 `--model` 指定路径。

官方 API 与权重说明：https://docs.ultralytics.com/models/yolo26/

## 本项目验证

语法、CLI 帮助以及针孔距离计算已在 Python 3.10 验证。初版仅在沙箱内检查，未发现摄像头。后续已在沙箱外使用 conda yolo 环境和 /dev/video0 完成两帧分割与单目深度推理，识别到屏幕和键盘；绝对距离精度尚未用尺子验证。

## 已有 conda yolo 环境：无需逐个物体尺寸的深度模式（推荐）

新增 `camera_depth.py`：YOLO26 实例分割 + YOLO26 单目深度，取物体掩码内部有效深度中位数。界面标注 `est`，表示预测的可见表面深度，不是精密测距或物体中心距离。无需为每类物体填写宽度。绝对尺度依赖场景，首次应使用尺子验证。

本机启动脚本固定使用 `/home/zy0002-hlw/miniconda3/envs/yolo/bin/python`，不需要重建或重新安装环境：

```bash
./run_yolo.sh --source 0
```

也可以手动激活：

```bash
source /home/zy0002-hlw/miniconda3/etc/profile.d/conda.sh
conda activate yolo
python camera_depth.py --source 0
```

当前系统摄像头名称为 `2K USB Camera`，优先使用 `0`。脚本必须在能够访问 `/dev/video0` 的本机终端运行，受限沙箱可能隐藏设备。当前环境 CUDA 不可用，默认使用 CPU，分割和深度双模型速度需要实测。

若同一场景中预测 80cm、实测 50cm，可先用 `./run_yolo.sh --depth-scale 0.625` 做全局尺度修正，并在多个位置复核。这是简化的单参数修正，不能保证整个距离范围准确；官方深度模型还提供对数尺度两参数校准方法。不同摄像头、场景或距离范围可能需要重新验证，不能直接把未校准数值视为真实厘米距离。

官方深度说明：https://docs.ultralytics.com/tasks/depth/
