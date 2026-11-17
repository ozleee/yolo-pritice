from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors
from xml.sax.saxutils import escape
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
body=ParagraphStyle('body',fontName='STSong-Light',fontSize=11,leading=18,spaceAfter=9,wordWrap='CJK')
heading=ParagraphStyle('heading',parent=body,fontSize=17,leading=24,textColor=colors.HexColor('#164e63'),spaceAfter=16)
code_style=ParagraphStyle('code',fontName='Courier',fontSize=8,leading=11,backColor=colors.HexColor('#f1f5f9'),borderPadding=8,spaceBefore=5,spaceAfter=12)
story=[]
def p(s): story.append(Paragraph(escape(s),body))
def h(s): story.append(Paragraph(escape(s),heading))
def code(s): story.append(Preformatted(s,code_style))
def page(): story.append(PageBreak())
h('YOLO26 桌面物体标注与训练指南')
p('适用工程：/home/zy0002-hlw/yolo-pritice ｜ 环境：conda yolo')
p('目标：用自己的摄像头图片微调 yolo26n-seg.pt，改善物体识别与轮廓分割，再接入 camera_depth.py。深度模型继续使用原权重；识别训练不会直接校准距离。')
h('1. 确定类别')
p('先选 3 至 5 类跑通流程。示例编号：0=cup（杯子），1=bottle（瓶子），2=mouse（鼠标），3=keyboard（键盘），4=cell_phone（手机）。')
p('如果所有杯子都应识别为“杯子”，统一标为 cup；如果需要区分两个特定杯子，就建立不同类别并分别采集足够图片。标注和配置文件的编号必须一致。')
h('2. 激活已有环境')
code('source /home/zy0002-hlw/miniconda3/etc/profile.d/conda.sh\nconda activate yolo\ncd /home/zy0002-hlw/yolo-pritice')
p('先退出当前摄像头程序，避免设备被占用。下一页提供采集脚本；按 S 保存原始画面，按 Q 退出。')
page();h('3. 用摄像头采集图片')
p('在工程根目录新建 capture_images.py，粘贴以下代码。此代码需自行创建文件；本指南不修改现有摄像头程序。')
code('''from pathlib import Path
import time
import cv2

output = Path("datasets/desktop/raw")
output.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(0)
try:
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read frame")
        cv2.imshow("S: save / Q: quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            path = output / f"desk_{time.time_ns()}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Cannot save: {path}")
            print(path)
        elif key in (ord("q"), 27):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
''')
code('python capture_images.py')
p('先拍 100 至 200 张验证流程，之后可按每类出现在约 200 至 500 张图片中作为起步目标；这不是保证效果的门槛。一张图可包含多个类别。')
page();h('4. 采集变化并划分数据')
p('每次拍摄前改变物体位置、方向、距离、光照或桌面背景。既拍单个物体，也拍多个物体及适量遮挡。同类有多种外观时都应覆盖。加入少量无目标背景图。')
p('建议训练集 train 约 80%，验证集 val 约 10%，测试集 test 约 10%。分几次重新摆放后采集，同一次摆放的相似图片保持在同一组，不要将连续相邻帧随机分散，否则验证结果可能虚高。')
h('5. 标注每个物体的轮廓')
p('使用 CVAT 等支持多边形实例分割标注、并能导出 Ultralytics YOLO Segmentation 格式的工具。新建项目，创建类别，上传图片，选择 Polygon / 多边形工具。')
p('沿物体可见边缘逐点绘制并闭合。每个物体单独一个轮廓：两个杯子要画两个。图片中所有属于目标类别的可见物体都应标注。')
p('轮廓尽量贴边，不要包含大片桌面。遮挡时只标可见部分，不凭空补全。无法确定类别或边界的图片可先剔除。没有目标的背景图保留为空标签。')
p('关键：camera_depth.py 依赖分割掩码，仅画矩形框的数据不适用于本训练流程。')
h('6. 导出并检查格式')
p('导出选择 Ultralytics YOLO Segmentation，而不是 Detection。标签文件每个物体一行，格式为：')
code('class_id x1 y1 x2 y2 x3 y3 ...')
p('坐标由工具归一化到 0 至 1，多边形至少三个点。如果一行只有类别编号加四个数，通常是矩形检测框格式，需要重新导出。')
page();h('7. 整理数据目录')
code('''datasets/desktop/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/''')
p('图片与标签同名、分组一致，例如 images/train/desk_001.jpg 对应 labels/train/desk_001.txt。将 raw 中的图片及导出的标注整理到各组。')
p('创建 datasets/desktop/data.yaml：')
code('''path: /home/zy0002-hlw/yolo-pritice/datasets/desktop
train: images/train
val: images/val
test: images/test

names:
  0: cup
  1: bottle
  2: mouse
  3: keyboard
  4: cell_phone''')
p('以标注工具实际导出的类别顺序为准。如果顺序不同，必须同步修改 YAML。')
h('8. 先训练一轮检查')
code('''yolo segment train \\
  model=yolo26n-seg.pt \\
  data=datasets/desktop/data.yaml \\
  epochs=1 imgsz=640 batch=2 device=cpu workers=0 \\
  project=runs/segment name=desktop_check''')
p('打开输出目录中的 train_batch*.jpg，检查类别与轮廓是否正确，排查漏标及损坏标签。日志中有数据错误时先修复，再正式训练。')
page();h('9. 正式微调与验证')
p('当前已检查的 yolo 环境 CUDA 不可用，因此使用 CPU。训练可能耗时较长；若之后配置好 CUDA 并确认可用，可改用 device=0。')
code('''yolo segment train \\
  model=yolo26n-seg.pt \\
  data=datasets/desktop/data.yaml \\
  epochs=50 imgsz=640 batch=2 device=cpu workers=0 \\
  project=runs/segment name=desktop_v1''')
p('优先使用训练输出的 weights/best.pt；last.pt 是最后一轮权重。若输出目录已存在，工具可能生成 desktop_v12 等新目录，以日志中的实际路径为准。')
p('使用保留的 test 数据评价：')
code('''yolo segment val \\
  model=runs/segment/desktop_v1/weights/best.pt \\
  data=datasets/desktop/data.yaml split=test device=cpu''')
p('保存测试集预测图片，直观看误检、漏检及轮廓贴合情况：')
code('''yolo segment predict \\
  model=runs/segment/desktop_v1/weights/best.pt \\
  source=datasets/desktop/images/test \\
  device=cpu save=True''')
p('比较原模型与新模型时使用同一测试集和相同推理参数。不要只看置信度是否升高，也要观察各类召回、误检以及分割质量。')
page();h('10. 接入摄像头')
code('''./run_yolo.sh --source 0 \\
  --model runs/segment/desktop_v1/weights/best.pt''')
p('如果已有距离修正系数，可在命令末尾附加 --depth-scale 你的系数；不要直接照搬示例系数。camera_depth.py 的深度模型仍使用 yolo26n-depth.pt。')
p('只用五类数据训练后，输出类别将变为这五类，不会自动保留原模型所有类别。如需识别更多类别，应添加相应数据并重新训练。')
h('后续迭代')
p('将实际使用中的漏检、误检场景重新拍摄并正确标注，补入训练集，训练下一版。验证集用于调参；测试集保持独立，不因测试结果反复调整它。')
p('建议保存原始数据、标注、data.yaml、训练参数和每版 best.pt。大量图片与权重不宜直接提交普通 Git 仓库，可单独备份；源码与小型配置可提交分支。')
h('识别与测距的区别')
p('此流程训练的是物体识别与实例分割，不训练真实距离。更好的轮廓可减少背景污染，但深度尺度、距离跳动和绝对误差仍需单独校准与验证。训练深度模型需要 RGB 图片配套真实深度标签。')
h('官方参考')
p('实例分割训练：https://docs.ultralytics.com/tasks/segment/')
p('分割数据格式：https://docs.ultralytics.com/datasets/segment/')
p('小型流程测试集：https://docs.ultralytics.com/datasets/segment/coco8-seg/')
def footer(c,d):
 c.setStrokeColor(colors.HexColor('#cbd5e1'));c.line(42,43,553,43)
 c.setFont('STSong-Light',9);c.setFillColor(colors.HexColor('#64748b'))
 c.drawString(42,29,'YOLO26 桌面物体训练 | conda yolo');c.drawRightString(553,29,str(d.page))
SimpleDocTemplate('output/pdf/YOLO26_桌面物体标注与训练指南.pdf',pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=58,title='YOLO26 桌面物体标注与训练指南',author='').build(story,onFirstPage=footer,onLaterPages=footer)
