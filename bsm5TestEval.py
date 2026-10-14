
import csv
import glob
import os
import pickle
import random
import shutil
from collections import deque

import cv2
import numpy as np

import csv
import glob
import os
import pickle
import random
import shutil
from collections import deque

import cv2
import numpy as np

#SETTINGS 
MODE = "run"             
VIDEO_PATH = r"C:\Users\ytabt\OneDrive\VSCode\cameraBlindSpotCode\testDataBSM9.mp4"   
VIDEO_DIR = r"C:\Users\ytabt\OneDrive\VSCode\blindSpotCode"      
TRAIN_GLOB = "labels_*.csv"
MODEL_FILE = "adjacent_model_v2.pkl"
DISPLAY_SCALE = 0.5        

VEHICLES = {2, 3, 5, 7}     
CONF = 0.30
CROP = (0, 0, 1650, 1080)  
                            
                           
SHOW_ALL_BOXES = False     
                            
IMGSZ = 640                
                           

# Run-mode filtering
PROB_SMOOTHING = 0.4       
ENTER_P = 0.6            
EXIT_P = 0.35               
MIN_TRACK_AGE = 3           
COAST_FRAMES = 10          
TRACK_TIMEOUT = 45
HOLD_FRAMES = 15           
RECENT_ROWS = 45            

# Evaluate mode
EVAL_TOL_FRAMES = 15       
EVAL_GAP_FRAMES = 15       

FEATURES = ["w_frac", "h_frac", "cx_frac", "cy_frac", "top_frac", "bottom_frac", "aspect",
            "clip_bottom", "clip_left", "clip_right", "is_big", "is_moto",
            "dcx", "dw", "age_s"]
META = ["video", "frame", "track_id", "label"]


class FeatureBuilder:
    """Turns a tracked box into features; keeps short per-track history for motion."""

    def __init__(self, fw, fh, fps):
        self.fw, self.fh, self.fps = fw, fh, fps or 30.0
        self.hist = {}
        self.born = {}

    def build(self, frame_idx, tid, cls, x1, y1, x2, y2):
        fw, fh, edge = self.fw, self.fh, 3
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        cx, wf = ((x1 + x2) / 2) / fw, w / fw
        hist = self.hist.setdefault(tid, deque(maxlen=6))
        self.born.setdefault(tid, frame_idx)
        hist.append((frame_idx, cx, wf))
        dcx = dw = 0.0
        if len(hist) >= 2:
            f0, c0, w0 = hist[0]
            dt = (frame_idx - f0) / self.fps
            if dt > 0:
                dcx, dw = (cx - c0) / dt, (wf - w0) / dt
        age = (frame_idx - self.born[tid]) / self.fps
        return [wf, h / fh, cx, ((y1 + y2) / 2) / fh, y1 / fh, y2 / fh, w / h,
                float(y2 >= fh - edge), float(x1 <= edge), float(x2 >= fw - edge),
                float(cls in (5, 7)), float(cls == 3), dcx, dw, age]


def show(frame):
    if DISPLAY_SCALE != 1.0:
        frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("blind spot", frame)


def vehicle_boxes(results):
    """Yield (track_id, class_id, x1, y1, x2, y2) for tracked vehicles."""
    for box in results.boxes:
        c = int(box.cls)
        if c in VEHICLES and box.id is not None:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            yield int(box.id), c, x1, y1, x2, y2


def crop_frame(frame):
    if CROP is None:
        return frame
    x1, y1, x2, y2 = CROP
    return frame[y1:y2, x1:x2]


def video_info(cap):
    """Width/height of the frames the model will actually see (after cropping), and fps."""
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if CROP is not None:
        x1, y1, x2, y2 = CROP
        w, h = min(x2, w) - x1, min(y2, h) - y1
    return w, h, cap.get(cv2.CAP_PROP_FPS) or 30.0


def write_rows(path, rows, append=False):
    new = not (append and os.path.exists(path))
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META + FEATURES)
        if new:
            w.writeheader()
        w.writerows(rows)


def make_row(video, frame_idx, tid, label, feats):
    row = {"video": video, "frame": frame_idx, "track_id": tid, "label": label}
    row.update(dict(zip(FEATURES, feats)))
    return row


# LABEL 
def label_mode(cap):
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    video_name = os.path.splitext(os.path.basename(str(VIDEO_PATH)))[0]
    out_csv = f"labels_{video_name}.csv"
    fw, fh, fps = video_info(cap)
    fb = FeatureBuilder(fw, fh, fps)

    track_label, rows = {}, []
    state = {"boxes": [], "rows": {}}

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        px, py = x / DISPLAY_SCALE, y / DISPLAY_SCALE
        hits = [((x2 - x1) * (y2 - y1), tid) for tid, _, x1, y1, x2, y2 in state["boxes"]
                if x1 <= px <= x2 and y1 <= py <= y2]
        if hits:
            tid = min(hits)[1]
            track_label[tid] = 1 - track_label.get(tid, 0)
            if tid in state["rows"]:
                state["rows"][tid]["label"] = track_label[tid]

    cv2.namedWindow("blind spot")
    cv2.setMouseCallback("blind spot", on_click)
    paused, frame, frame_idx = False, None, 0
    print("Click a car's box when it enters your blind spot, click again when it leaves.")
    print("SPACE = pause/resume   S or Esc = save and quit")

    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break
            frame_idx += 1
            frame = crop_frame(frame)
            results = model.track(frame, persist=True, conf=CONF, imgsz=IMGSZ,
                                  tracker="bytetrack.yaml", verbose=False)[0]
            state["boxes"] = list(vehicle_boxes(results))
            state["rows"] = {}
            for tid, cls, x1, y1, x2, y2 in state["boxes"]:
                feats = fb.build(frame_idx, tid, cls, x1, y1, x2, y2)
                row = make_row(video_name, frame_idx, tid, track_label.get(tid, 0), feats)
                rows.append(row)
                state["rows"][tid] = row

        view = frame.copy()
        for tid, _, x1, y1, x2, y2 in state["boxes"]:
            lab = track_label.get(tid, 0)
            color = (0, 0, 255) if lab else (0, 200, 0)
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(view, f"#{tid} {'BLIND SPOT' if lab else ''}",
                        (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(view, "PAUSED" if paused else "click a box to toggle label", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        show(view)

        key = cv2.waitKey(30 if paused else 1) & 0xFF
        if key == ord(" "):
            paused = not paused
        elif key in (ord("s"), 27):
            break

    if rows:
        write_rows(out_csv, rows)
        pos = sum(int(r["label"]) for r in rows)
        print(f"Saved {len(rows)} rows to {out_csv} ({pos} labeled as blind spot).")
    else:
        print("Nothing to save.")


# REFEATURE 
def refeature():
    """Carry v1 labels over to v2 features by re-running tracking on the same videos."""
    from ultralytics import YOLO

    old_files = []
    for p in sorted(glob.glob(TRAIN_GLOB)):
        with open(p) as f:
            header = next(csv.reader(f), [])
        if "bottom_frac" not in header:
            old_files.append(p)
    if not old_files:
        print("No old-format label files found.")
        return

    for p in old_files:
        with open(p) as f:
            old = list(csv.DictReader(f))
        if not old:
            continue
        name = old[0]["video"]
        labels = {(int(r["frame"]), int(r["track_id"])): int(r["label"]) for r in old}

        vpath = next((os.path.join(VIDEO_DIR, name + e) for e in (".mp4", ".MP4", ".mov", ".MOV")
                      if os.path.exists(os.path.join(VIDEO_DIR, name + e))), None)
        if vpath is None:
            print(f"{p}: could not find video '{name}' in {VIDEO_DIR}. Skipping.")
            continue

        print(f"{p}: re-running tracking on {vpath} ...")
        cap = cv2.VideoCapture(vpath)
        fw, fh, fps = video_info(cap)
        fb = FeatureBuilder(fw, fh, fps)
        model = YOLO("yolov8n.pt")
        new_rows, frame_idx = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            frame = crop_frame(frame)
            results = model.track(frame, persist=True, conf=CONF, imgsz=IMGSZ,
                                  tracker="bytetrack.yaml", verbose=False)[0]
            for tid, cls, x1, y1, x2, y2 in vehicle_boxes(results):
                feats = fb.build(frame_idx, tid, cls, x1, y1, x2, y2)
                if (frame_idx, tid) in labels:
                    new_rows.append(make_row(name, frame_idx, tid, labels[(frame_idx, tid)], feats))
        cap.release()

        rate = len(new_rows) / len(labels)
        print(f"  matched {len(new_rows)}/{len(labels)} old rows ({rate:.0%}).")
        if rate < 0.9:
            print("  WARNING: many rows did not match (track IDs can differ if settings or the "
                  "model changed). Consider relabeling this video in label mode.")
        shutil.move(p, os.path.join(os.path.dirname(p), "old_" + os.path.basename(p)))
        write_rows(p, new_rows)
        print(f"  wrote {p} (old file kept as old_{os.path.basename(p)})")


# TRAIN
def load_rows(paths):
    X, y, groups = [], [], []
    for p in paths:
        with open(p) as f:
            reader = csv.DictReader(f)
            if not set(FEATURES) <= set(reader.fieldnames or []):
                print(f"Skipping {p}: old format. Run MODE = 'refeature' first.")
                continue
            for r in reader:
                X.append([float(r[k]) for k in FEATURES])
                y.append(int(r["label"]))
                groups.append(f"{r['video']}:{r['track_id']}")
    return np.array(X), np.array(y), np.array(groups)


def train(paths=None, model_file=None):
    from sklearn.ensemble import RandomForestClassifier

    paths = paths or sorted(glob.glob(TRAIN_GLOB))
    model_file = model_file or MODEL_FILE
    X, y, groups = load_rows(paths)
    if len(y) == 0:
        print(f"No usable rows in {TRAIN_GLOB}. Run label (or refeature) first.")
        return
    print(f"{len(y)} rows, {len(set(groups))} cars; {int(y.sum())} blind-spot rows, "
          f"{int((1 - y).sum())} not.")
    if y.sum() == 0 or (1 - y).sum() == 0:
        print("Need both blind-spot and not-blind-spot examples.")
        return

    def make():
        return RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=8,
                                      n_jobs=-1, random_state=0)

    # Validate on whole cars the model has not seen
    ids = sorted(set(groups))
    random.Random(0).shuffle(ids)
    val_ids = set(ids[:max(1, len(ids) // 4)])
    is_val = np.array([g in val_ids for g in groups])
    if len(ids) >= 8 and y[~is_val].sum() > 0 and y[is_val].sum() > 0 and (1 - y[is_val]).sum() > 0:
        clf = make().fit(X[~is_val], y[~is_val])
        p = clf.predict_proba(X[is_val])[:, 1]
        yv = y[is_val]
        print(f"\nValidation on {len(val_ids)} unseen cars (frame-level):")
        print("  threshold | blind-spot frames caught | false alarms (non-blind-spot frames)")
        for t in (0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8):
            pred = p >= t
            tp, fn = int((pred & (yv == 1)).sum()), int((~pred & (yv == 1)).sum())
            fp, tn = int((pred & (yv == 0)).sum()), int((~pred & (yv == 0)).sum())
            print(f"  {t:>8.2f}  | {tp:>5}/{tp + fn:<5} ({tp / max(tp + fn, 1):>4.0%})      "
                  f"| {fp:>5}/{fp + tn:<5} ({fp / max(fp + tn, 1):>4.0%})")
        print("  Pick ENTER_P from this table. Higher = fewer false alarms but more misses.")
    else:
        print("Too little data for a validation split. Label more clips.")

    clf = make().fit(X, y)
    print("\nFeature importance (what the model relies on):")
    for name, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:<12} {imp:.3f}")
    with open(model_file, "wb") as f:
        pickle.dump({"clf": clf, "features": FEATURES}, f)
    print(f"\nSaved {model_file}")


# RUN 
class RunTrack:
    def __init__(self, frame_idx):
        self.p = None
        self.inside = False
        self.born = frame_idx
        self.last_seen = frame_idx
        self.recent = deque(maxlen=RECENT_ROWS)   # feature rows, for hard-negative saving

    def update(self, frame_idx, p):
        self.p = p if self.p is None else (1 - PROB_SMOOTHING) * self.p + PROB_SMOOTHING * p
        self.last_seen = frame_idx
        self.inside = (self.p >= EXIT_P) if self.inside else (self.p >= ENTER_P)


def load_model():
    """Return the trained classifier, or None (with a message) if it is missing/outdated."""
    try:
        with open(MODEL_FILE, "rb") as f:
            saved = pickle.load(f)
    except FileNotFoundError:
        print(f"No {MODEL_FILE}. Run label mode, then train mode.")
        return None
    if saved["features"] != FEATURES:
        print("Model was trained with different features. Retrain.")
        return None
    clf = saved["clf"]
    clf.n_jobs = 1   # thread start-up per call is slower than the prediction itself
    return clf


class Pipeline:
    """Per-frame decision logic, shared by run and evaluate so they behave identically."""

    def __init__(self, clf, fb, video_name):
        self.clf, self.fb, self.video_name = clf, fb, video_name
        self.pos_col = list(clf.classes_).index(1)
        self.tracks, self.hold = {}, 0

    def step(self, frame_idx, results):
        """Returns (drawn, alert). drawn = [(tid, x1, y1, x2, y2, track), ...]."""
        drawn = []
        dets = list(vehicle_boxes(results))
        all_feats = [self.fb.build(frame_idx, tid, cls, x1, y1, x2, y2)
                     for tid, cls, x1, y1, x2, y2 in dets]
        # One batched prediction per frame instead of one call per car
        probs = (self.clf.predict_proba(np.array(all_feats))[:, self.pos_col]
                 if all_feats else [])
        for (tid, cls, x1, y1, x2, y2), feats, p in zip(dets, all_feats, probs):
            tr = self.tracks.setdefault(tid, RunTrack(frame_idx))
            tr.update(frame_idx, float(p))
            tr.recent.append(make_row(self.video_name, frame_idx, tid, 0, feats))
            drawn.append((tid, x1, y1, x2, y2, tr))

        for tid in [t for t, tr in self.tracks.items()
                    if frame_idx - tr.last_seen > TRACK_TIMEOUT]:
            del self.tracks[tid]

        occupied = any(tr.inside and (frame_idx - tr.born) >= MIN_TRACK_AGE
                       and (frame_idx - tr.last_seen) <= COAST_FRAMES
                       for tr in self.tracks.values())
        self.hold = HOLD_FRAMES if occupied else max(0, self.hold - 1)
        return drawn, self.hold > 0


def run(cap):
    from ultralytics import YOLO

    clf = load_model()
    if clf is None:
        return

    video_name = os.path.splitext(os.path.basename(str(VIDEO_PATH)))[0]
    hardneg_csv = f"labels_hardneg_{video_name}.csv"
    model = YOLO("yolov8n.pt")
    fw, fh, fps = video_info(cap)
    pipe = Pipeline(clf, FeatureBuilder(fw, fh, fps), video_name)
    tracks = pipe.tracks
    frame_idx, paused, frame, msg_until = 0, False, None, 0
    drawn, alert = [], False
    print("SPACE = pause   N = false alarm (saves hard negatives)   Esc = quit")

    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                print("End of video (or could not read frame).")
                break
            frame_idx += 1
            frame = crop_frame(frame)
            results = model.track(frame, persist=True, conf=CONF, imgsz=IMGSZ,
                                  tracker="bytetrack.yaml", verbose=False)[0]

            drawn, alert = pipe.step(frame_idx, results)

        view = frame.copy()
        for tid, x1, y1, x2, y2, tr in drawn:
            if not tr.inside and not SHOW_ALL_BOXES:
                continue   # only box the cars the model says are in the blind spot
            color = (0, 0, 255) if tr.inside else (255, 128, 0)
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(view, f"#{tid} p={tr.p:.2f}", (int(x1), max(25, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if alert:
            cv2.putText(view, "BLIND SPOT", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
            # Later: turn on the LED here (Raspberry Pi GPIO)
        if frame_idx < msg_until:
            cv2.putText(view, "saved as false alarm", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        show(view)

        key = cv2.waitKey(30 if paused else 1) & 0xFF
        if key == 27:
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("n"):
            saved_rows = [r for tr in tracks.values() if tr.inside for r in tr.recent]
            if saved_rows:
                write_rows(hardneg_csv, saved_rows, append=True)
                msg_until = frame_idx + 45
                print(f"Saved {len(saved_rows)} hard-negative rows to {hardneg_csv}")
            else:
                print("No car currently flagged, nothing saved.")


# EVALUATE
def episodes(frames, gap):
    """Group frame numbers into [start, end] runs, merging gaps of <= gap frames."""
    eps = []
    for f in sorted(frames):
        if eps and f - eps[-1][1] <= gap:
            eps[-1][1] = f
        else:
            eps.append([f, f])
    return eps


def evaluate(cap):
    """Measure the real alert behaviour against your labels on a clip the model never trained on.

    Setup: label the test clip in label mode, then RENAME labels_<clip>.csv to eval_<clip>.csv
    so train ignores it. Set VIDEO_PATH to that clip and MODE = "evaluate".
    """
    from ultralytics import YOLO

    clf = load_model()
    if clf is None:
        return
    video_name = os.path.splitext(os.path.basename(str(VIDEO_PATH)))[0]
    gt_file = f"eval_{video_name}.csv"
    if not os.path.exists(gt_file):
        print(f"Missing {gt_file}. Label this clip in label mode, then rename "
              f"labels_{video_name}.csv to {gt_file}.")
        return
    if os.path.exists(f"labels_{video_name}.csv"):
        print(f"WARNING: labels_{video_name}.csv also exists, so this clip may have been used "
              "for training. Results would be too optimistic. Remove or rename it and retrain.")

    gt_frames = set()
    with open(gt_file) as f:
        for r in csv.DictReader(f):
            if int(r["label"]) == 1:
                gt_frames.add(int(r["frame"]))

    model = YOLO("yolov8n.pt")
    fw, fh, fps = video_info(cap)
    pipe = Pipeline(clf, FeatureBuilder(fw, fh, fps), video_name)
    alert_frames, frame_idx = set(), 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print("Evaluating (no window; this takes as long as the video takes to process) ...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = crop_frame(frame)
        results = model.track(frame, persist=True, conf=CONF, imgsz=IMGSZ,
                              tracker="bytetrack.yaml", verbose=False)[0]
        _, alert = pipe.step(frame_idx, results)
        if alert:
            alert_frames.add(frame_idx)
        if frame_idx % 150 == 0:
            print(f"  frame {frame_idx}/{total}")

    def t(frame):
        s = frame / fps
        return f"{int(s // 60)}:{s % 60:04.1f}"

    gt_eps = episodes(gt_frames, EVAL_GAP_FRAMES)
    al_eps = episodes(alert_frames, EVAL_GAP_FRAMES)
    tol = EVAL_TOL_FRAMES

    print(f"\n=== Results for {video_name} ({frame_idx} frames, {fps:.0f} fps) ===")
    print(f"Cars/passes in your blind spot (from your labels): {len(gt_eps)}")
    print(f"Alert episodes the program produced:               {len(al_eps)}\n")

    caught, latencies = 0, []
    print("Blind-spot passes:")
    for s, e in gt_eps:
        hits = [f for f in alert_frames if s - tol <= f <= e + tol]
        if hits:
            caught += 1
            lat = (min(hits) - s) / fps
            latencies.append(lat)
            print(f"  {t(s)}-{t(e)}  CAUGHT   (alert {lat:+.2f}s relative to your label start)")
        else:
            print(f"  {t(s)}-{t(e)}  MISSED")
    false_eps = [(s, e) for s, e in al_eps
                 if not any(s - tol <= f <= e + tol for f in gt_frames)]
    print("\nFalse alarm episodes (alert with no labeled car nearby):")
    if false_eps:
        for s, e in false_eps:
            print(f"  {t(s)}-{t(e)}  FALSE ALARM  ({(e - s + 1) / fps:.1f}s)")
    else:
        print("  none")

    tp = len(gt_frames & alert_frames)
    fn = len(gt_frames - alert_frames)
    fp = len(alert_frames - gt_frames)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    print("\n--- Summary ---")
    if gt_eps:
        print(f"Passes caught:        {caught}/{len(gt_eps)} ({caught / len(gt_eps):.0%})")
    print(f"False alarm episodes: {len(false_eps)}")
    if latencies:
        print(f"Median alert timing:  {sorted(latencies)[len(latencies) // 2]:+.2f}s "
              "(negative = alert came before your label, positive = after)")
    print(f"Frame-level: alert on during {recall:.0%} of labeled frames; "
          f"{precision:.0%} of alert frames were labeled blind-spot frames")
    print(f"(episodes merge gaps up to {EVAL_GAP_FRAMES} frames; matching tolerance "
          f"{EVAL_TOL_FRAMES} frames)")


# main 
def main():
    if MODE == "train":
        train()
        return
    if MODE == "refeature":
        refeature()
        return
    cap = cv2.VideoCapture(VIDEO_PATH)
    print("Opened:", cap.isOpened())
    if not cap.isOpened():
        print("Video did not open. Check the path, or convert HEVC to H.264.")
        return
    if MODE == "label":
        label_mode(cap)
    elif MODE == "run":
        run(cap)
    elif MODE == "evaluate":
        evaluate(cap)
    else:
        print("Unknown MODE:", MODE)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
