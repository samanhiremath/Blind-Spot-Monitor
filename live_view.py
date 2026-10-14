"""
Minimal live radar view with a zone alert (AWR1843).


"""
import argparse
import csv
import math
import queue
import struct
import threading
import time
from collections import defaultdict, deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

import alert_zone as A
import awr1843_logger as L



class LiveSource:
    def __init__(self, a):
        import serial
        self.cli = serial.Serial(a.cli, 115200, timeout=0.1)
        self.data = serial.Serial(a.data, 921600, timeout=0.1)
        print(f"Sending {a.cfg} ...")
        L.send_cfg(self.cli, a.cfg)
        self.data.reset_input_buffer()
        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        buf = b""
        while not self.stop_flag.is_set():
            chunk = self.data.read(self.data.in_waiting or 1)
            if chunk:
                buf += chunk
            while True:
                i = buf.find(L.MAGIC)
                if i < 0:
                    buf = buf[-7:]
                    break
                if i > 0:
                    buf = buf[i:]
                if len(buf) < L.HEADER_SIZE:
                    break
                total_len = struct.unpack_from("<I", buf, 12)[0]
                if total_len < L.HEADER_SIZE or total_len > 200000:
                    buf = buf[8:]
                    continue
                if len(buf) < total_len:
                    break
                packet, buf = buf[:total_len], buf[total_len:]
                try:
                    self.q.put(L.parse_frame(packet))
                except struct.error:
                    pass

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self.stop_flag.set()
        self.thread.join(timeout=1)
        try:
            self.cli.write(b"sensorStop\n")
            time.sleep(0.2)
        except Exception:
            pass
        self.cli.close()
        self.data.close()


class ReplaySource:
    def __init__(self, path, frame_ms=100.0, clock=time.time, loop=True):
        frames = defaultdict(list)
        with open(path) as f:
            for r in csv.DictReader(f):
                frames[int(r["frame"])].append((float(r["x_m"]), float(r["y_m"]), float(r["z_m"]),
                                                float(r["vel_mps"]), float(r["snr_db"]), float(r["noise_db"])))
        if not frames:
            raise SystemExit("No points in file.")
        self.first, self.last = min(frames), max(frames)
        self.frames, self.frame_s, self.clock, self.loop = frames, frame_ms / 1000.0, clock, loop
        self.t0, self.next = clock(), self.first

    def poll(self):
        out = []
        due = self.first + int((self.clock() - self.t0) / self.frame_s)
        while self.next <= min(due, self.last):
            out.append((self.next, self.frames.get(self.next, [])))
            self.next += 1
        if self.loop and self.next > self.last:
            self.t0, self.next = self.clock(), self.first
        return out

    def close(self):
        pass



class Viewer:
    def __init__(self, source, a, beep=True):
        self.src, self.a, self.beep_on = source, a, beep
        self.alert = A.ZoneAlert(a.on_frames, a.off_frames)
        self.trail = deque(maxlen=a.trail)
        self.last_frame, self.fps_t, self.fps_n, self.fps = None, time.time(), 0, 0.0

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        poly = A.zone_polygon(a)
        if a.view_range is None and a.mount_yaw is None:
            a.view_range = 4.0
        if a.view_range is None:   # car-frame zone: fit the map to the zone
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            self.ax.set_xlim(min(0, min(xs)) - 1.5, max(0, max(xs)) + 1.5)
            self.ax.set_ylim(min(-0.3, min(ys) - 1.5), max(ys) + 1.5)
        else:
            r = a.view_range
            self.ax.set_xlim(-r, r)
            self.ax.set_ylim(-0.3, r * 1.4 if a.view_range < 8 else r)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x, right (m)")
        self.ax.set_ylabel("y, forward (m)")
        self.ax.grid(alpha=0.3)
        self.zone = Polygon(poly, closed=True, facecolor="#2ca02c", alpha=0.25, edgecolor="#2ca02c",
                            linewidth=2, zorder=1)
        self.ax.add_patch(self.zone)
        self.ax.plot(0, 0, "k^", markersize=12, zorder=5)
        self.ignored = self.ax.scatter([], [], s=14, c="#bbbbbb", zorder=2)
        self.accepted = self.ax.scatter([], [], s=24, c="#1f77b4", zorder=3)
        self.inzone = self.ax.scatter([], [], s=40, c="#d62728", zorder=4)
        self.banner = self.ax.text(0.5, 0.97, "CLEAR", transform=self.ax.transAxes, ha="center", va="top",
                                   fontsize=26, fontweight="bold", color="#2ca02c")
        self.status = self.ax.text(0.01, 0.01, "", transform=self.ax.transAxes, fontsize=9, va="bottom")
        self.fig.canvas.manager.set_window_title("AWR1843 zone monitor") if self.fig.canvas.manager else None

    def _passes(self, p):
        x, y, z, v, snr, noise = p
        a = self.a
        return math.hypot(x, y) <= a.max_range and snr >= a.min_snr and abs(v) >= a.min_speed

    def step(self, _=None):
        new = self.src.poll()
        for frame, points in new:
            hits = A.points_in_zone(points, self.a)
            event = self.alert.update(len(hits) >= self.a.min_points)
            if event:
                A.report(event, frame, 0.0, hits) if False else print(
                    f"frame {frame}: alert {'ON' if event == 'ON' else 'cleared'}")
                if event == "ON" and self.beep_on:
                    A.beep()
            self.trail.append(points)
            self.last_frame = frame
            self.fps_n += 1
        a = self.a
        ign, acc, inz = [], [], []
        for pts in self.trail:
            for p in pts:
                if not self._passes(p):
                    ign.append((p[0], p[1]))
                elif A.in_zone(a, p[0], p[1]):
                    inz.append((p[0], p[1]))
                else:
                    acc.append((p[0], p[1]))
        for sc, pts in ((self.ignored, ign), (self.accepted, acc), (self.inzone, inz)):
            sc.set_offsets(pts if pts else [[float("nan"), float("nan")]])
        on = self.alert.active
        color = "#d62728" if on else "#2ca02c"
        self.zone.set_facecolor(color)
        self.zone.set_edgecolor(color)
        self.banner.set_text("ALERT" if on else "CLEAR")
        self.banner.set_color(color)
        self.fig.patch.set_facecolor("#ffd6d6" if on else "white")
        now = time.time()
        if now - self.fps_t >= 1.0:
            self.fps, self.fps_n, self.fps_t = self.fps_n / (now - self.fps_t), 0, now
        self.status.set_text(f"frame {self.last_frame}   {self.fps:.0f} fps")
        return self.ignored, self.accepted, self.inzone, self.zone, self.banner, self.status

    def run(self):
        self.anim = FuncAnimation(self.fig, self.step, interval=50, blit=False, cache_frame_data=False)
        try:
            plt.show()
        finally:
            self.src.close()


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    live = sub.add_parser("live")
    live.add_argument("--cli", default="COM5")
    live.add_argument("--data", default="COM6")
    live.add_argument("--cfg", default="blindspot_awr1843.cfg")
    rep = sub.add_parser("replay")
    rep.add_argument("source")
    rep.add_argument("--frame-ms", type=float, default=100.0)
    for p in (live, rep):
        A.add_zone_args(p)
        p.add_argument("--view-range", type=float, default=None,
                       help="half-width of the map in meters (default 4, or automatic with --mount-yaw)")
        p.add_argument("--trail", type=int, default=4, help="how many recent frames of points to keep on screen")
    return ap.parse_args(argv)


def main():
    a = build_args()
    A.announce_zone(a)
    src = LiveSource(a) if a.mode == "live" else ReplaySource(a.source, a.frame_ms)
    Viewer(src, a).run()


if __name__ == "__main__":
    main()
