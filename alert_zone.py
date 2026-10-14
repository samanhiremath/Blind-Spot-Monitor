"""
Zone alert for the AWR1843: raises an alert when radar points appear inside a rectangle.

Two ways to define the zone:

 1) Radar frame (default): a rectangle in meters from the radar, x = right, y = forward (along the radar's
    boresight). Default x 0..1, y 0..1.   Options: --x-min --x-max --y-min --y-max

 2) Car frame (use this to mimic a real blind-spot radar): give --mount-yaw, the angle the radar is turned
    outward from straight back (45 = rear quarter panel angled 45 degrees). The zone is then a rectangle on
    the road next to the car, and the code rotates it into radar coordinates for you.
        --mount-yaw 45        radar boresight is 45 deg outward from straight back
        --side right|left     which side of the car the radar is on (default right)
        --lat-min 0.5 --lat-max 4      sideways distance outward from the car's side (m)
        --rear-min 0 --rear-max 8      distance behind the radar (m); negative = ahead of the radar

LIVE (needs the radar; close the Visualizer first):
    python alert_zone.py live --cli COM5 --data COM6 --cfg blindspot_awr1843.cfg

REPLAY a log recorded by awr1843_logger.py (no radar needed):
    python alert_zone.py replay logs\\<folder>\\points.csv

Useful options:
    --x-min 0 --x-max 1 --y-min 0 --y-max 1     zone rectangle
    --min-snr 18        ignore weak points (dB)
    --min-speed 0.5     ignore points slower than this (m/s); use it to ignore stationary clutter
    --max-range 20      ignore far points
    --on-frames 3       zone must have points for this many frames in a row to alert (10 fps -> 0.3 s)
    --off-frames 5      zone must be clear this many frames in a row to clear the alert
    --min-points 1      points needed in a frame to count as a hit

Needs awr1843_logger.py in the same folder. Live mode needs: pip install pyserial
"""
import argparse
import csv
import math
import struct
import sys
import time
from collections import defaultdict

import awr1843_logger as L


class ZoneAlert:
    """Debounced alert: ON after `on_frames` hit frames in a row, OFF after `off_frames` clear frames in a row."""

    def __init__(self, on_frames, off_frames):
        self.on_frames, self.off_frames = on_frames, off_frames
        self.hits = self.clear = 0
        self.active = False

    def update(self, hit):
        if hit:
            self.hits += 1
            self.clear = 0
        else:
            self.clear += 1
            self.hits = 0
        if not self.active and self.hits >= self.on_frames:
            self.active = True
            return "ON"
        if self.active and self.clear >= self.off_frames:
            self.active = False
            return "OFF"
        return None


def zone_polygon(a):
    """Zone corners in radar coordinates (x right, y along boresight)."""
    if a.mount_yaw is None:
        return [(a.x_min, a.y_min), (a.x_max, a.y_min), (a.x_max, a.y_max), (a.x_min, a.y_max)]
    yaw = math.radians(a.mount_yaw)
    sy, cy = math.sin(yaw), math.cos(yaw)
    poly = []
    for lat, rear in ((a.lat_min, a.rear_min), (a.lat_max, a.rear_min),
                      (a.lat_max, a.rear_max), (a.lat_min, a.rear_max)):
        r, f = lat, -rear          # car frame for a right-side radar: r = outward, f = forward
        x = -r * cy - f * sy
        y = r * sy - f * cy
        poly.append((-x if a.side == "left" else x, y))   # left side is the mirror image
    return poly


def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def in_zone(a, x, y):
    if a.mount_yaw is None:  # plain rectangle: inclusive edges
        return a.x_min <= x <= a.x_max and a.y_min <= y <= a.y_max
    poly = getattr(a, "_poly", None)
    if poly is None:
        poly = a._poly = zone_polygon(a)
    return point_in_poly(x, y, poly)


def describe_zone(a):
    poly = zone_polygon(a)
    corners = "  ".join(f"({x:.1f},{y:.1f})" for x, y in poly)
    if a.mount_yaw is None:
        return f"Zone (radar frame) corners: {corners}"
    return (f"Zone (car frame, {a.side}-side radar, mount yaw {a.mount_yaw:g} deg): sideways {a.lat_min}..{a.lat_max} m, "
            f"behind radar {a.rear_min}..{a.rear_max} m\n  = radar-frame corners: {corners}")


def announce_zone(a):
    print(describe_zone(a))
    for x, y in zone_polygon(a):
        ang = math.degrees(math.atan2(x, y)) if y > 0 else 180.0
        if y <= 0 or abs(ang) > 60:
            print(f"  WARNING: corner ({x:.1f},{y:.1f}) is outside the radar's +/-60 deg field of view "
                  f"({'behind the radar' if y <= 0 else f'{ang:.0f} deg off boresight'}); that part of the zone cannot be seen.")
        if math.hypot(x, y) > a.max_range:
            print(f"  WARNING: corner ({x:.1f},{y:.1f}) is beyond --max-range {a.max_range} m.")


def add_zone_args(p):
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--x-max", type=float, default=1.0)
    p.add_argument("--y-min", type=float, default=0.0)
    p.add_argument("--y-max", type=float, default=1.0)
    p.add_argument("--mount-yaw", type=float, default=None,
                   help="car-frame zone: radar boresight angle outward from straight back (deg), e.g. 45")
    p.add_argument("--side", choices=["right", "left"], default="right")
    p.add_argument("--lat-min", type=float, default=0.5)
    p.add_argument("--lat-max", type=float, default=4.0)
    p.add_argument("--rear-min", type=float, default=0.0)
    p.add_argument("--rear-max", type=float, default=8.0)
    p.add_argument("--min-snr", type=float, default=0.0)
    p.add_argument("--min-speed", type=float, default=0.0)
    p.add_argument("--max-range", type=float, default=22.0)
    p.add_argument("--on-frames", type=int, default=3)
    p.add_argument("--off-frames", type=int, default=5)
    p.add_argument("--min-points", type=int, default=1)


def points_in_zone(points, a):
    """points: list of (x, y, z, vel, snr, noise). Returns the points that pass filters and lie in the zone."""
    found = []
    for (x, y, z, v, snr, noise) in points:
        if math.hypot(x, y) > a.max_range:
            continue
        if snr < a.min_snr or abs(v) < a.min_speed:
            continue
        if in_zone(a, x, y):
            found.append((x, y, v, snr))
    return found


def beep():
    try:
        import winsound
        winsound.Beep(1000, 200)
    except Exception:
        print("\a", end="")


def report(event, frame, t, hits):
    if event == "ON":
        where = ", ".join(f"({x:.2f},{y:.2f}) v={v:+.2f}" for x, y, v, _ in hits[:3])
        print(f"[{t:7.2f}s frame {frame}] *** ALERT ON ***  {where}")
        beep()
    elif event == "OFF":
        print(f"[{t:7.2f}s frame {frame}] alert cleared")


def run_replay(a):
    frames = defaultdict(list)
    with open(a.source) as f:
        for r in csv.DictReader(f):
            frames[int(r["frame"])].append((
                float(r["x_m"]), float(r["y_m"]), float(r["z_m"]), float(r["vel_mps"]),
                float(r["snr_db"]), float(r["noise_db"])))
    if not frames:
        raise SystemExit("No points in file.")
    first, last = min(frames), max(frames)
    announce_zone(a)
    alert = ZoneAlert(a.on_frames, a.off_frames)
    n_on, active_frames = 0, 0
    for fr in range(first, last + 1):  # frames with no detections are not in the csv: treat as empty
        hits = points_in_zone(frames.get(fr, []), a)
        event = alert.update(len(hits) >= a.min_points)
        t = (fr - first) * a.frame_ms / 1000.0
        if event == "ON":
            n_on += 1
        if alert.active:
            active_frames += 1
        report(event, fr, t, hits)
    total = last - first + 1
    print(f"\nReplayed {total} frames ({total * a.frame_ms / 1000:.1f} s): "
          f"{n_on} alert(s), alert active {100 * active_frames / total:.0f}% of the time.")


def run_live(a):
    import serial
    cli = serial.Serial(a.cli, 115200, timeout=0.1)
    data = serial.Serial(a.data, 921600, timeout=0.1)
    print(f"Sending {a.cfg} ...")
    L.send_cfg(cli, a.cfg)
    data.reset_input_buffer()
    announce_zone(a)
    print("Watching (Ctrl+C to stop)")
    alert = ZoneAlert(a.on_frames, a.off_frames)
    buf, start = b"", time.time()
    try:
        while True:
            chunk = data.read(data.in_waiting or 1)
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
                    frame, points = L.parse_frame(packet)
                except struct.error:
                    continue
                hits = points_in_zone(points, a)
                event = alert.update(len(hits) >= a.min_points)
                report(event, frame, time.time() - start, hits)
                state = "ALERT" if alert.active else "clear"
                print(f"frame {frame}: {len(hits)} point(s) in zone  [{state}]    ", end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cli.write(b"sensorStop\n")
            time.sleep(0.2)
        except Exception:
            pass
        cli.close()
        data.close()
        print("\nStopped.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    live = sub.add_parser("live")
    live.add_argument("--cli", default="COM5")
    live.add_argument("--data", default="COM6")
    live.add_argument("--cfg", default="blindspot_awr1843.cfg")
    rep = sub.add_parser("replay")
    rep.add_argument("source", help="points.csv from awr1843_logger.py")
    rep.add_argument("--frame-ms", type=float, default=100.0, help="frame period, for time stamps (cfg uses 100)")
    for p in (live, rep):
        add_zone_args(p)
    a = ap.parse_args()
    run_live(a) if a.mode == "live" else run_replay(a)


if __name__ == "__main__":
    main()