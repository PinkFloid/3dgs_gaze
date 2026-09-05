"""Extract the unannotated, calibrated grasp frame used by Fig. 2 v3."""
from pathlib import Path
import sys, json
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Eye_Tracker/tools'))
from pupil_localizer import load_fisheye, scale_K

def main():
    rec=Path('/home/liuchy/recordings/2026_08_27/020')
    out=ROOT/'output/fig2_recorded_v3/assets';out.mkdir(parents=True,exist_ok=True)
    ts=np.load(rec/'world_timestamps.npy'); requested=31.5
    i=int(np.searchsorted(ts,ts[0]+requested))
    cap=cv2.VideoCapture(str(rec/'world.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    ok,img=cap.read();cap.release()
    if not ok: raise RuntimeError(f'Unable to read frame {i}')
    K0,D=load_fisheye(str(ROOT/'SceneRebuild/Calibration_result/world_camera_calibration.npz'))
    h,w=img.shape[:2];K=scale_K(K0,(1920,1080),(w,h))
    m1,m2=cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),K,(w,h),cv2.CV_16SC2)
    img=cv2.remap(img,m1,m2,cv2.INTER_LINEAR)
    crop=[820,470,1290,990];x0,y0,x1,y1=crop
    assert cv2.imwrite(str(out/'grasp_recorded.png'),img[y0:y1,x0:x1])
    provenance={'recording':str(rec),'source_video':'world.mp4','frame_index':i,
        'time_requested_s':requested,'time_actual_s':float(ts[i]-ts[0]),
        'undistortion':'OpenCV fisheye, calibrated K and D, output K equals original K',
        'calibration':'SceneRebuild/Calibration_result/world_camera_calibration.npz',
        'crop_xyxy':crop,'modifications':'calibrated undistortion and crop only'}
    (out/'grasp_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
