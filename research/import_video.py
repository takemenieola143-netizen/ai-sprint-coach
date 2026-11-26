"""Convert a lawfully obtained original video into an unverified local training draft.

Install locally: pip install opencv-python mediapipe
Use official MediaPipe Pose Landmarker task model (download separately from Google).
Usage: python research/import_video.py clip.mp4 pose_landmarker.task draft.json
No web scraping, upload, auto-labeling or automatic permission/timing claims.
"""
import argparse, json, math
from pathlib import Path
def run(video, model, output):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Cannot decode video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid nominal frame rate or image dimensions")
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    rows = []
    last_ms = -1
    try:
        with vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if len(rows) >= 36000:
                    raise ValueError("Clip exceeds 36000-frame limit; trim original first")
                i = len(rows)
                t = i/fps
                ms = max(last_ms+1, round(t*1000))
                last_ms = ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
                points = []
                if result.pose_landmarks:
                    for p in result.pose_landmarks[0]:
                        vals = [p.x,p.y,p.z,p.visibility]
                        points.append(dict(x=p.x,y=p.y,z=p.z,visibility=p.visibility) if all(v is not None and math.isfinite(v) for v in vals) else None)
                rows.append(dict(seekSeconds=t,duplicate=False,points=points,
                                 reportedDecoderPositionMs=cap.get(cv2.CAP_PROP_POS_MSEC)))
    finally:
        cap.release()
    if not rows:
        raise ValueError("No decoded frames")
    draft=dict(schema="sprint-contact-training-draft-v1",id=None,athlete=None,
               split=None,license="pending",source=None,groundTruth="unverified",
               timeVerified=False,completeAnnotation=False,width=width,height=height,
               playbackTimeScale=1,nominalFPS=fps,rows=rows,events=[],
               notes="Frame-index/fps timestamps are provisional. Verify native PTS, variable frame rate, slow motion and repeats before training. Independently annotate all contact events and license/provenance. Keep private footage local.")
    Path(output).write_text(json.dumps(draft,allow_nan=False),encoding="utf-8")
    print(f"Decoded {len(rows)} frames; {sum(len(r['points'])==33 for r in rows)} full poses. UNVERIFIED draft, not training ground truth.")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("video");p.add_argument("model");p.add_argument("output")
    a=p.parse_args();run(a.video,a.model,a.output)
