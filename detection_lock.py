"""Keep a selected ByteTrack ID, conservatively rebind after detection loss."""
from collections import deque
import cv2
import numpy as np


class DetectionLock:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active=False
        self.state='DETECT'
        self.pending_id=None
        self.pending_hits=0
        self.lost_frames=0
        self.current=None
        self.recent_appearances=deque(maxlen=5)
        self.stable_hits=0

    @staticmethod
    def appearance(frame,box):
        h,w=frame.shape[:2]
        x1,y1,x2,y2=map(int,box)
        crop=frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
        if min(crop.shape[:2])<4:return None
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1,2],None,[12,8,8],[0,180,0,256,0,256])
        cv2.normalize(hist,hist,alpha=1,norm_type=cv2.NORM_L1)
        return hist

    def start(self,frame,d):
        if d.get('id') is None:return False
        appearance=self.appearance(frame,d['box'])
        if appearance is None:return False
        self.reset()
        self.active=True;self.state='TRACK'
        self.track_id=d['id'];self.name=d['name']
        self.box=np.array(d['box'],float);self.reference=appearance.copy()
        self.recent_appearances.append(appearance.copy())
        return True

    def update(self,frame,detections):
        if not self.active:return None,False
        same=next((d for d in detections if d['id']==self.track_id),None)
        # Normal association remains ByteTrack's job, with no appearance gate on rotation.
        if same is not None and self.lost_frames==0:
            box=np.array(same['box'],float)
            old_area=max(1,np.prod(self.box[2:]-self.box[:2]))
            ratio=np.prod(box[2:]-box[:2])/old_area
            distance=np.linalg.norm((box[:2]+box[2:]-self.box[:2]-self.box[2:])/2)
            stable=(same['name']==self.name and same.get('confidence',1.)>=.5
                    and .65<ratio<1.55
                    and distance<max(20,np.linalg.norm(self.box[2:]-self.box[:2])*.6))
            self.stable_hits=self.stable_hits+1 if stable else 0
            if self.stable_hits>=2:
                appearance=self.appearance(frame,box)
                if appearance is not None:
                    self.recent_appearances.append(appearance.copy())
                    self.reference=np.mean(self.recent_appearances,axis=0).astype(np.float32)
            self.box=box;self.current=same;self.state='TRACK'
            return dict(same,name=self.name),False
        # Freeze the last stable appearance throughout loss and candidate confirmation.
        self.stable_hits=0
        self.lost_frames+=1;self.state='LOST';self.current=None
        if self.lost_frames>150:
            self.pending_id=None;self.pending_hits=0
            return None,False # Avoid indefinite automatic identity claims.
        old=self.box;diag=max(20,np.linalg.norm(old[2:]-old[:2]));rank=[]
        for d in detections:
            if d['id'] is None or d['name']!=self.name:continue
            b=np.array(d['box'],float)
            ratio=np.prod(b[2:]-b[:2])/max(1,np.prod(old[2:]-old[:2]))
            distance=np.linalg.norm((b[:2]+b[2:]-old[:2]-old[2:])/2)
            if not .45<ratio<2.2 or distance>diag*2+40:continue
            hist=self.appearance(frame,b)
            if hist is None:continue
            similarity=1-cv2.compareHist(self.reference,hist,cv2.HISTCMP_BHATTACHARYYA)
            if similarity<.65:continue
            score=.8*similarity+.2*max(0,1-distance/(diag*2+40))
            rank.append((score,d))
        rank.sort(key=lambda x:x[0],reverse=True)
        if not rank or (len(rank)>1 and rank[0][0]-rank[1][0]<.08):
            self.pending_id=None;self.pending_hits=0;return None,False
        candidate=rank[0][1]
        if candidate['id']==self.pending_id:self.pending_hits+=1
        else:self.pending_id=candidate['id'];self.pending_hits=1
        if self.pending_hits<3:return None,False
        self.track_id=candidate['id'];self.box=np.array(candidate['box'],float)
        self.state='TRACK';self.lost_frames=0;self.pending_id=None;self.pending_hits=0
        self.current=candidate
        # Drop pre-loss samples only after recovery is confirmed.
        self.recent_appearances.clear()
        self.recent_appearances.append(self.reference.copy())
        return dict(candidate,name=self.name),True
