import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

#INPUT_VIDEO = r"Vehicle-Detection-and-In-Out-Counting-using-YOLO/sampleVideo.mp4"


INPUT_VIDEO = r"Vehicle-Detection-and-In-Out-Counting-using-YOLO/sampleVideo.mp4"

MODEL_NAME = "yolov8s.pt"
CONF_THRESH = 0.40
INFERENCE_SIZE = 1280

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
WINDOW_NAME = "Vehicle Detection, IN/OUT count"

ENTRY_LINE_RATIO = 0.60
EXIT_LINE_RATIO = 0.65

LINE_CROSS_TOLERANCE = 5

LINE_START = 0.0
LINE_END = 0.5

EXIT_LINE_START = 0.5
EXIT_LINE_END = 1.0

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

CLASS_COLORS = {
    2: (0,   220,   0),
    3: (220,   0, 220),
    5: (0,   160, 255),
    7: (0,     0, 220),
}

ENTRY_LINE_COLOR = (0, 255, 255)
EXIT_LINE_COLOR = (0,  80, 255)
TRACK_COLOR = (200, 200, 200)

class CentroidTracker:
    def __init__(self, max_lost=30):
        self.next_id = 0
        self.objects = {}
        self.cls_map = {}
        self.lost = {}
        self.trails = defaultdict(list)
        self.max_lost = max_lost

    def update(self, detections):
        """
        detections: list of (cx, cy, cls_id)
        Returns: dict {id: (cx, cy, cls_id)}
        """
        if not detections:
            for oid in list(self.lost):
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    del self.objects[oid]
                    del self.cls_map[oid]
                    del self.lost[oid]
                    self.trails.pop(oid, None)
            return {}

        if not self.objects:
            for (cx, cy, cls_id) in detections:
                self.objects[self.next_id] = (cx, cy)
                self.cls_map[self.next_id] = cls_id
                self.lost[self.next_id] = 0
                self.next_id += 1
        else:
            obj_ids = list(self.objects.keys())
            obj_pts = np.array(list(self.objects.values()), dtype=float)
            det_pts = np.array([(d[0], d[1]) for d in detections], dtype=float)

            # Cost matrix: euclidean distance
            cost = np.linalg.norm(obj_pts[:, None] - det_pts[None, :], axis=2)

            used_rows, used_cols = set(), set()
            for _ in range(min(len(obj_ids), len(detections))):
                idx = np.argmin(cost)
                r, c = divmod(idx, cost.shape[1])
                if cost[r, c] > 120:
                    break
                oid = obj_ids[r]
                self.objects[oid] = (det_pts[c][0], det_pts[c][1])
                self.cls_map[oid] = detections[c][2]
                self.lost[oid] = 0
                cost[r, :] = 1e9
                cost[:, c] = 1e9
                used_rows.add(r)
                used_cols.add(c)

            # Unmatched detections → new objects
            for c, (cx, cy, cls_id) in enumerate(detections):
                if c not in used_cols:
                    self.objects[self.next_id] = (cx, cy)
                    self.cls_map[self.next_id] = cls_id
                    self.lost[self.next_id] = 0
                    self.next_id += 1

            # Unmatched existing → increment lost
            for r, oid in enumerate(obj_ids):
                if r not in used_rows:
                    self.lost[oid] += 1
                    if self.lost[oid] > self.max_lost:
                        del self.objects[oid]
                        del self.cls_map[oid]
                        del self.lost[oid]
                        self.trails.pop(oid, None)

        # Update trails
        for oid, (cx, cy) in self.objects.items():
            self.trails[oid].append((int(cx), int(cy)))
            if len(self.trails[oid]) > 40:
                self.trails[oid].pop(0)

        return {oid: (cx, cy, self.cls_map[oid])
                for oid, (cx, cy) in self.objects.items()}

def scale_thickness(w):
    return max(1, int(w / 500))


def draw_counting_lines(frame, entry_y, exit_y):
    h, w = frame.shape[:2]

    # Entry line (left half)
    ex_start = int(w * LINE_START)
    ex_end = int(w * LINE_END)
    cv2.line(frame, (ex_start, entry_y), (ex_end, entry_y), ENTRY_LINE_COLOR, 2)
    cv2.putText(frame, "IN", (ex_start + 10, entry_y - 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, ENTRY_LINE_COLOR, 1)

    # Exit line (right half)
    xl_start = int(w * EXIT_LINE_START)
    xl_end = int(w * EXIT_LINE_END)
    cv2.line(frame, (xl_start, exit_y), (xl_end, exit_y), EXIT_LINE_COLOR, 2)
    cv2.putText(frame, "OUT", (xl_start + 10, exit_y - 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, EXIT_LINE_COLOR, 1)


def draw_trails(frame, tracker):
    for oid, pts in tracker.trails.items():
        for i in range(1, len(pts)):
            alpha = int(255 * i / len(pts))
            cv2.line(frame, pts[i - 1], pts[i], TRACK_COLOR, 1)


def draw_boxes(frame, results, tracker_objects):
    h, w = frame.shape[:2]
    thick = scale_thickness(w)
    fscale = 0.5

    centroid_to_id = {(int(cx), int(cy)): oid
                      for oid, (cx, cy, _) in tracker_objects.items()}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        conf = float(box.conf[0])
        if conf < CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = CLASS_COLORS[cls_id]
        label = VEHICLE_CLASSES[cls_id]
        caption = f"ID?  {label}  {conf:.2f}"

        best_id, best_dist = None, 9999
        for (tx, ty), oid in centroid_to_id.items():
            d = abs(tx - cx) + abs(ty - cy)
            if d < best_dist:
                best_dist, best_id = d, oid
        if best_id is not None and best_dist < 60:
            caption = f"#{best_id}  {label}  {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        cl = max(10, (x2 - x1) // 6)
        for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(frame, (px, py), (px + dx*cl, py), color, thick+1)
            cv2.line(frame, (px, py), (px, py + dy*cl), color, thick+1)

        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_DUPLEX, fscale, thick)
        pad = 4
        cv2.rectangle(frame, (x1, y1-th-bl-pad*2), (x1+tw+pad, y1), color, -1)
        cv2.putText(frame, caption, (x1+pad//2, y1-bl-pad//2),
                    cv2.FONT_HERSHEY_DUPLEX, fscale, (255, 255, 255), thick)

        # Centroid dot
        cv2.circle(frame, (cx, cy), 4, color, -1)

    return frame


def draw_dashboard(frame, counts_entry, counts_exit, fps_display, frame_idx, total):
    """Semi-transparent """
    h, w = frame.shape[:2]
    panel_w = 220
    panel_x = w - panel_w - 20
    panel_y = 50

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 8, panel_y - 8),
                  (w - 5, panel_y + 210), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    def put(text, y, color=(230, 230, 230), scale=0.6):
        cv2.putText(frame, text, (panel_x, panel_y + y),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, 1)

    put("        VEHICLE COUNTER", 0,  (255, 255, 100), 0.5)
    put("              IN | OUT", 24, ENTRY_LINE_COLOR, 0.5)

    total_entry = sum(counts_entry.values())
    total_exit = sum(counts_exit.values())

    row = 50
    for cls_id, name in VEHICLE_CLASSES.items():
        e = counts_entry.get(cls_id, 0)
        x = counts_exit.get(cls_id, 0)
        color = CLASS_COLORS[cls_id]
        put(f"{name:<12} {e:>3}  |  {x:>3}", row, color)
        row += 24

    put("----------------------------", row, (170, 180, 160), 0.45)
    row += 20
    put(f"TOTAL IN   :  {total_entry}", row, ENTRY_LINE_COLOR)
    row += 22
    put(f"TOTAL OUT  :  {total_exit}",  row, EXIT_LINE_COLOR)
    row += 22
    net = total_entry - total_exit
    net_col = (0, 220, 0) if net >= 0 else (0, 80, 255)
    put(f"NET FLOW   : {net:+d}", row, net_col)

    # Top info bar
    pct  = frame_idx / total * 100 if total else 0
    info = f"  Frame {frame_idx}/{total}  ({pct:.1f}%)   FPS: {fps_display:.1f}   |  Q = quit"
    bar  = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, info, (8, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (230, 230, 230), 1)

    return frame

def process_video(input_path):
    model = YOLO(MODEL_NAME)
    tracker = CentroidTracker(max_lost=30)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    entry_y = int(orig_h * ENTRY_LINE_RATIO)
    exit_y = int(orig_h * EXIT_LINE_RATIO)

    # Counters  {cls_id: count}
    counts_entry = defaultdict(int)
    counts_exit = defaultdict(int)
    crossed_entry = set()
    crossed_exit = set()

    print(f"\n▶  Input       : {input_path}")
    print(f"📐  Resolution  : {orig_w}×{orig_h}  |  {fps:.1f} fps  |  {total} frames")
    print(f"📏  Entry line  : y = {entry_y}  ({ENTRY_LINE_RATIO*100:.0f}% from top)")
    print(f"📏  Exit  line  : y = {exit_y}  ({EXIT_LINE_RATIO*100:.0f}% from top)")
    print("   Press  Q  to quit.\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    tick = cv2.getTickFrequency()
    frame_idx = 0
    fps_display = 0.0

    while True:
        t0 = cv2.getTickCount()

        ret, frame = cap.read()
        if not ret:
            break

        # ── Detection ────────────────────────────────────────────────────────
        results = model(frame, imgsz=INFERENCE_SIZE, verbose=False)

        # Collect vehicle centroids for tracker
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            if float(box.conf[0]) < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(((x1+x2)//2, (y1+y2)//2, cls_id))

        # ── Tracking ─────────────────────────────────────────────────────────
        tracked = tracker.update(detections)

        # ── Line crossing logic ───────────────────────────────────────────────
        for oid, (cx, cy, cls_id) in tracked.items():
            trail = tracker.trails[oid]
            if len(trail) < 2:
                continue
            prev_y = trail[-2][1]
            curr_y = trail[-1][1]

            # Entry line crossing (moving downward: prev_y < entry_y <= curr_y)
            if oid not in crossed_entry:
                if prev_y < entry_y <= curr_y or curr_y < entry_y <= prev_y:
                    crossed_entry.add(oid)
                    counts_entry[cls_id] += 1
                    print(f"  ✅ ENTRY  #{oid} {VEHICLE_CLASSES[cls_id]}  "
                          f"| Total IN={sum(counts_entry.values())}")

            # Exit line crossing
            if oid not in crossed_exit:
                if prev_y < exit_y <= curr_y or curr_y < exit_y <= prev_y:
                    crossed_exit.add(oid)
                    counts_exit[cls_id] += 1
                    print(f"  🚪 EXIT   #{oid} {VEHICLE_CLASSES[cls_id]}  "
                          f"| Total OUT={sum(counts_exit.values())}")

        # ── Draw everything ───────────────────────────────────────────────────
        draw_trails(frame, tracker)
        draw_counting_lines(frame, entry_y, exit_y)
        frame = draw_boxes(frame, results, tracked)
        frame = draw_dashboard(frame, counts_entry, counts_exit,
                               fps_display, frame_idx + 1, total)

        # ── Display ───────────────────────────────────────────────────────────
        display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                             interpolation=cv2.INTER_LINEAR)
        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("⏹  Stopped by user.")
            break

        fps_display = tick / (cv2.getTickCount() - t0 + 1e-9)
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n── Final Count ─────────────────────────────────────")
    for cls_id, name in VEHICLE_CLASSES.items():
        e = counts_entry.get(cls_id, 0)
        x = counts_exit.get(cls_id, 0)
        print(f"  {name:<12}  IN: {e:>3}   OUT: {x:>3}")
    print(f"  {'TOTAL':<12}  IN: {sum(counts_entry.values()):>3}   "
          f"OUT: {sum(counts_exit.values()):>3}   "
          f"NET: {sum(counts_entry.values()) - sum(counts_exit.values()):+d}")
    print("────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    process_video(INPUT_VIDEO)
