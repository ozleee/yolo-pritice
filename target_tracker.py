"""Single-object optical flow with appearance-verified, confirmed recovery."""
import cv2
import numpy as np


class TargetTracker:
    def __init__(self):
        self.active = False
        self.pending = None
        self.state = 'DETECT'

    @staticmethod
    def crop(frame, box):
        h,w = frame.shape[:2]
        x1,y1,x2,y2 = np.rint(box).astype(int)
        return frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]

    def reset(self):
        self.active = False
        self.pending = None
        self.state = 'DETECT'

    def seed(self, frame, box):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, np.uint8)
        x1,y1,x2,y2 = np.rint(box).astype(int)
        cv2.rectangle(mask,(max(0,x1),max(0,y1)),(min(gray.shape[1]-1,x2),min(gray.shape[0]-1,y2)),255,-1)
        self.points = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=.01,
                                              minDistance=4, mask=mask)
        self.gray = gray
        self.box = np.array(box,dtype=float)

    def start(self, frame, detection):
        patch = self.crop(frame,detection['box'])
        if min(patch.shape[:2]) < 8:
            return False
        gray = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        if gray.std() < 4:
            return False
        self.template = gray.copy() # Frozen reference, never learn background drift.
        self.name = detection['name']
        self.seed(frame,detection['box'])
        self.active = True
        self.state = 'TRACK'
        self.pending = None
        return True

    def similarity(self, frame, box):
        patch = self.crop(frame,box)
        if min(patch.shape[:2]) < 8:
            return -1.
        gray = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(self.template.shape[1],self.template.shape[0]))
        return float(cv2.matchTemplate(gray,self.template,cv2.TM_CCOEFF_NORMED)[0,0])

    def update(self, frame):
        if not self.active or self.state == 'LOST':
            return None
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if self.points is None or len(self.points)<6:
            self.state='LOST';return None
        nxt,st,_ = cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None)
        if nxt is None:
            self.state='LOST';return None
        back,bst,_ = cv2.calcOpticalFlowPyrLK(gray,self.gray,nxt,None)
        if back is None:
            self.state='LOST';return None
        valid=(st.ravel()==1)&(bst.ravel()==1)&(np.linalg.norm(back-self.points,axis=2).ravel()<1.5)
        old=self.points[valid].reshape(-1,2);new=nxt[valid].reshape(-1,2)
        if len(old)<6:
            self.state='LOST';return None
        transform,inliers=cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=2)
        if transform is None or inliers.sum()<6 or inliers.mean()<.55:
            self.state='LOST';return None
        x1,y1,x2,y2=self.box
        corners=np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]).reshape(-1,1,2)
        moved=cv2.transform(corners,transform).reshape(-1,2)
        box=np.r_[moved.min(axis=0),moved.max(axis=0)]
        h,w=gray.shape
        box[[0,2]]=np.clip(box[[0,2]],0,w);box[[1,3]]=np.clip(box[[1,3]],0,h)
        ratio=np.prod(box[2:]-box[:2])/max(1,np.prod(self.box[2:]-self.box[:2]))
        if not .5<ratio<2 or min(box[2:]-box[:2])<8 or self.similarity(frame,box)<.25:
            self.state='LOST';return None
        self.seed(frame,box)
        return self.box.copy()

    def recover(self, frame, detections):
        """Search changed YOLO IDs/classes AND image templates, require two hits."""
        if not self.active or self.state!='LOST':
            return None
        candidates=[]
        old_area=max(1,np.prod(self.box[2:]-self.box[:2]))
        for d in detections:
            b=np.array(d['box'],float)
            ratio=np.prod(b[2:]-b[:2])/old_area
            if .4<ratio<2.5:
                score=self.similarity(frame,b)
                if score>=.72: candidates.append((score,b))
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        for scale in [.8,1.,1.2]:
            t=cv2.resize(self.template,None,fx=scale,fy=scale)
            th,tw=t.shape
            if th>gray.shape[0] or tw>gray.shape[1]:continue
            scores=cv2.matchTemplate(gray,t,cv2.TM_CCOEFF_NORMED)
            # Keep two spatially separate peaks to reject ambiguous duplicates.
            for _ in range(2):
                _,score,_,(x,y)=cv2.minMaxLoc(scores)
                if score>=.82:candidates.append((score,np.array([x,y,x+tw,y+th],float)))
                scores[max(0,y-th//2):y+th//2+1,max(0,x-tw//2):x+tw//2+1]=-1
        candidates.sort(key=lambda v:v[0],reverse=True)
        if not candidates:
            self.pending=None;return None
        score,best=candidates[0]
        center=(best[:2]+best[2:])/2
        for other_score,other in candidates[1:]:
            distance=np.linalg.norm((other[:2]+other[2:])/2-center)
            if distance>max(20,np.linalg.norm(best[2:]-best[:2])*.6) and score-other_score<.08:
                self.pending=None;return None
        if self.pending is None or np.linalg.norm(center-self.pending)>max(25,np.linalg.norm(best[2:]-best[:2])):
            self.pending=center;return None
        self.seed(frame,best)
        self.pending=None
        self.state='TRACK'
        return self.box.copy()
