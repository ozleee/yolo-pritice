"""YOLO26 segmentation + monocular depth, no per-object dimensions required."""
import argparse
import json
import math
import time
from collections import deque


def object_depth(depth, mask, scale=1.0):
    import numpy as np
    if depth.shape != mask.shape:
        raise ValueError('深度图与分割掩码尺寸不一致')
    values = depth[mask & np.isfinite(depth) & (depth > 0)]
    return float(np.median(values)) * scale if values.size else None


class TrackHistory:
    """BBox-center displacement in original-image pixels; not physical motion."""
    def __init__(self, length=40, ttl=30):
        self.length, self.ttl, self.states = length, ttl, {}

    def update(self, track_id, center, frame_index):
        state = self.states.get(track_id)
        if state is None or frame_index - state['last'] > self.ttl:
            state = dict(origin=center, previous=center,
                         points=deque(maxlen=self.length), last=frame_index)
            self.states[track_id] = state
        x, y = center
        ox, oy = state['origin']
        px, py = state['previous']
        state['points'].append(center)
        state['previous'], state['last'] = center, frame_index
        return dict(dx_px=x-ox, dy_px=y-oy,
                    displacement_px=math.hypot(x-ox, y-oy),
                    step_dx_px=x-px, step_dy_px=y-py)

    def prune(self, frame_index):
        self.states = {k:v for k,v in self.states.items()
                       if frame_index-v['last'] <= self.ttl}


def main():
    import argparse
    from detection_lock import DetectionLock
    p=argparse.ArgumentParser(description='YOLO + ByteTrack 点击锁定与保守重新捕获')
    p.add_argument('--source',default='0')
    p.add_argument('--model',default='yolo26n-seg.pt')
    p.add_argument('--depth-model',default='yolo26n-depth.pt')
    p.add_argument('--device',default='cpu')
    p.add_argument('--width',type=int,default=640)
    p.add_argument('--height',type=int,default=480)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--conf',type=float,default=.4)
    p.add_argument('--depth-scale',type=float,default=.25)
    p.add_argument('--no-depth',action='store_true',help='关闭深度模型，优先测试跟踪速度')
    p.add_argument('--tracker',default='bytetrack.yaml',help='Ultralytics 跟踪器配置')
    p.add_argument('--no-track',action='store_true')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--motion-log')
    a=p.parse_args()
    if min(a.width,a.height,a.imgsz)<=0 or a.max_frames<0 or not 0<a.conf<=1:
        p.error('尺寸及间隔须为正，帧数不能为负，conf须在(0,1]')
    if not math.isfinite(a.depth_scale) or a.depth_scale<=0:p.error('无效深度比例')
    import cv2
    import numpy as np
    from ultralytics import YOLO
    source=int(a.source) if a.source.isdecimal() else a.source
    cap=cv2.VideoCapture(source)
    log=None
    tracker=DetectionLock();history=TrackHistory(ttl=1000000000)
    display={'frame':None,'detections':[]};pending=[];session_id=0
    window='YOLO26 - Left select/reselect | Right unlock | R reset | Q quit'
    def mouse(event,x,y,flags,param):
        if event==cv2.EVENT_RBUTTONDOWN:pending.append(('unlock',None,None))
        elif event==cv2.EVENT_LBUTTONDOWN and not a.no_track and display['frame'] is not None:
            hits=[d for d in display['detections'] if d['box'][0]<=x<=d['box'][2] and d['box'][1]<=y<=d['box'][3]]
            if hits:
                d=min(hits,key=lambda d:(d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]))
                pending.append(('select',display['frame'].copy(),d))
    try:
        if not cap.isOpened():raise RuntimeError('无法打开摄像头，请检查占用及编号')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.width);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,a.height)
        seg=YOLO(a.model,task='segment')
        depth_model=None if a.no_depth else YOLO(a.depth_model,task='depth')
        if a.motion_log:log=open(a.motion_log,'w',encoding='utf-8')
        if not a.headless:
            cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(window,mouse)
        count=0;last_print=0
        while True:
            ok,frame=cap.read()
            if not ok:
                if isinstance(source,int) or str(source).startswith('/dev/') or '://' in str(source):raise RuntimeError('摄像头读取失败')
                break
            start=time.monotonic()
            for action,reference,d in pending:
                if action=='unlock':tracker.reset();history.states.clear()
                elif tracker.start(reference,d):
                    session_id+=1;history.states.clear()
                    b=np.array(d['box']);history.update(session_id,tuple((b[:2]+b[2:])/2),count)
                    print(f'锁定 {tracker.name}，会话ID {session_id}',flush=True)
                else:print('目标暂时没有跟踪ID，请稍后重选',flush=True)
            pending.clear()
            options=dict(imgsz=a.imgsz,conf=min(a.conf,.15) if tracker.active else a.conf,
                         device=a.device,retina_masks=True,verbose=False)
            result=(seg.predict(frame,**options) if a.no_track else
                    seg.track(frame,persist=True,tracker=a.tracker,**options))[0]
            detections=[]
            for i,box in enumerate(result.boxes):
                mask=None if result.masks is None else result.masks.data[i].cpu().numpy()>.5
                detections.append(dict(box=box.xyxy[0].cpu().tolist(),name=result.names[int(box.cls.item())],
                                       confidence=float(box.conf.item()),mask=mask,
                                       id=int(box.id.item()) if box.id is not None else None))
            chosen,recovered=tracker.update(frame,detections) if tracker.active else (None,False)
            tracked=None if chosen is None else np.array(chosen['box'])
            depth=None
            if depth_model is not None and detections:
                dr=depth_model.predict(frame,imgsz=a.imgsz,device=a.device,verbose=False)[0]
                if dr.depth is None:raise RuntimeError('深度模型未返回深度')
                depth=dr.depth.data.cpu().numpy()
                if depth.shape!=frame.shape[:2]:raise RuntimeError('深度尺寸不匹配')
            shown=detections if not tracker.active else ([] if chosen is None else [chosen])
            canvas=frame.copy();items=[]
            for d in shown:
                box=d['box'];x1,y1,x2,y2=map(int,box);value=None
                if depth is not None and d['mask'] is not None:
                    mask=cv2.erode(d['mask'].astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
                    value=object_depth(depth,mask,a.depth_scale)
                center=((box[0]+box[2])/2,(box[1]+box[3])/2)
                item=dict(name=d['name'],confidence=d['confidence'],estimated_depth_m=value,
                          track_id=session_id if tracker.active else None,tracker_id=d.get('id'),center_px=center)
                label=d['name']+(f' ~{value*100:.1f}cm' if value is not None else ' depth N/A')
                if tracker.active:
                    item.update(history.update(session_id,center,count));label=f'#{session_id} '+label
                    pts=np.rint(list(history.states[session_id]['points'])).astype(np.int32)
                    cv2.polylines(canvas,[pts],False,(255,190,0),2)
                    cv2.putText(canvas,f'dX {item["dx_px"]:+.1f} dY {item["dy_px"]:+.1f} px',
                                (x1,min(canvas.shape[0]-8,y1+42)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,190,0),2)
                cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,220,0),2)
                cv2.putText(canvas,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,0),2)
                conf=f'Conf: {d["confidence"]:.1%}' if d['confidence'] is not None else 'Visual tracker (no detection score)'
                cv2.putText(canvas,conf,(x1,min(canvas.shape[0]-8,y1+22)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
                items.append(item)
            # Keep other targets hidden while locked, including during recovery.
            display['frame']=frame;display['detections']=detections if not tracker.active else shown
            status='DETECT: left select' if not tracker.active else ('LOST: searching / right click to unlock' if tracked is None else ('RECAPTURED' if recovered else 'TRACK'))
            cv2.putText(canvas,status,(10,45),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,200,255),2)
            now=time.monotonic();cv2.putText(canvas,f'FPS {1/max(now-start,1e-6):.1f} | Right: unlock',
                                             (10,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,200,255),2)
            record=dict(frame=count,state=tracker.state,selected_id=session_id if tracker.active else None,recovered=recovered,objects=items)
            if log:log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
            if now-last_print>=1:print(json.dumps(record,ensure_ascii=False),flush=True);last_print=now
            count+=1
            if not a.headless:
                cv2.imshow(window,canvas);key=cv2.waitKey(1)&255
                if key in (ord('q'),27):break
                if key in (ord('r'),ord('R')):history.states.clear()
                if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if a.max_frames and count>=a.max_frames:break
    finally:
        cap.release()
        if log:log.close()
        if not a.headless:cv2.destroyAllWindows()


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:pass
    except Exception as e:raise SystemExit(f'错误：{e}') from e
