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

## 多目标跟踪与像素偏移

`camera_depth.py` 默认启用 ByteTrack，无需 Supervision。使用 `./run_yolo.sh --source 0` 启动。框上显示 ID、红色置信度，以及 dX/dY 像素位移；蓝色线为最近40个观测点轨迹，紫色点/箭头表示首次观测中心到当前中心。按 R 重置位移参考，Q/Esc退出。

偏移基于原始分辨率的检测框中心，首次跟踪位置为零点：向右 x 为正，向下 y 为正。框抖动、摄像头移动都会引起读数变化，这不是物体实际厘米位移。ID 切换或超过30帧未观测后重新出现会建立新参考点；不提供跨会话身份保证。检测失败时不会输出旧位置冒充当前测量。

加 `--motion-log /tmp/motion.jsonl` 可逐帧记录 ID、center_px、dx_px、dy_px、displacement_px（到起点直线距离）和 step_dx_px/step_dy_px（相对上一次观测，不一定是上一帧）。加 `--no-track` 可退回普通分割模式；`--tracker botsort.yaml` 可更换跟踪器。

请固定摄像头，将同一物体缓慢向右/左/下/上移动，验证同一 ID 的 dX/dY 方向，按 R 重新归零。实际桌面位移需要相机标定及额外几何约束，不能用像素距离直接当作毫米。

### 点击选择单目标（当前默认交互）

启动时显示全部识别框，不显示跟踪ID和轨迹。左键点击目标框后，只显示选中ID的框、红色置信度、距离、轨迹与位移，其他目标隐藏。重叠框优先选择较小的框。右键点击画面任意位置解锁，恢复全部识别框并清空位移参考；按R仅重置参考。

为保证点击时已有可关联的ID，后台仍持续运行ByteTrack，但未选中时不绘制轨迹。目标短暂丢失时显示LOST并保持锁定，不自动切换目标；若身份关联失败，需右键解锁再重新选择。并不保证同ID绝不发生身份切换。--no-track 禁用点击跟踪，--headless 无法用鼠标选择。显示坐标与原图保持一致，窗口使用自动尺寸。

### 独立视觉跟踪与自动重新捕获（替代前述ByteTrack交互）

当前点击锁定后使用 OpenCV 稀疏光流、前后向一致性和 RANSAC 估计目标位置，不再依赖固定 YOLO ID；固定初始类别。每10帧检测一次用于新掩码/距离，丢失时每帧搜索。CPU优先测试命令：`./run_yolo.sh --source 0 --no-depth`。去掉 --no-depth 恢复测距；无新匹配掩码时显示 depth N/A，不复用旧掩码或把检测置信度当跟踪置信度。

丢失恢复使用初始外观模板、多尺度图像搜索及检测候选，连续两次匹配后恢复同一会话ID与位移参考；不要求原YOLO类别或ID一致。相似候选分数接近时拒绝自动选择。右键解锁后可重新选择；锁定期间仅显示选中物体，丢失显示LOST。轨迹最多保留40个观测点。

该轻量外观恢复不是身份识别保证：大幅旋转、强光照变化、相似物体、纹理不足及背景特征会引起失败或漂移。光流跟踪的是图像区域，不生成分割掩码。--detect-every 可调检测间隔。旧 --tracker 参数不再使用；--no-track 保留纯识别模式。单目深度运行会增加延迟，先关闭深度测试运动。

验证：`python -m unittest discover -s tests -v`，覆盖已知平移、遮挡后跨位置恢复、无YOLO检测恢复、相同外观歧义拒绝和解锁状态。

### 当前默认方案：恢复 YOLO + ByteTrack，单独处理重新捕获

当前入口已恢复每帧 `seg.track(persist=True, tracker='bytetrack.yaml')`，不再调用 `target_tracker.py` 的光流方案（该文件保留供比较）。左键锁定、右键解锁、轨迹、像素偏移、--no-depth 仍保留；--detect-every 已移除，支持 --tracker。

新 `detection_lock.py` 负责用户锁定会话与ByteTrack ID之间的关联。正常跟踪使用ByteTrack；失去观测后，用同类别、大小比例、距最后位置范围及冻结初始颜色直方图筛选候选。最优候选需无明显歧义且同ID连续出现3帧才接续，最多搜索150个处理帧。恢复后保留会话轨迹原点，tracker_id可能改变；日志同时输出两种ID。丢失期间不输出旧框和旧距离。匹配失败可右键解锁重选。

该恢复是保守启发式，不是可靠的物体身份识别；相似颜色和同类物体仍可能误关联，大幅移动/旋转/光照变化可能无法自动恢复。识别漏检也无法靠关联逻辑修复。建议先用 `./run_yolo.sh --source 0 --no-depth` 测试跟踪，再去掉 --no-depth 对比延迟。

#### 恢复参考更新

重新捕获不再始终对比初始外观。正常同ID跟踪期间，当类别一致、检测置信度>=0.5、框大小和中心变化较稳定，连续2帧满足条件后，更新最近5次有效观测的平均颜色直方图。目标丢失后冻结此参考，候选确认期间不学习新外观，避免错误候选污染参考。位置和尺寸继续使用丢失前最后一次观测。恢复确认后重新积累外观；轨迹位移原点保持首次选择位置，不随外观更新。低置信度和剧烈运动期间可能保留较早的稳定参考，并非无条件采用最后一帧。
