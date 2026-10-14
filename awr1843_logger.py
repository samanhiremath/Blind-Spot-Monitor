"""
AWR1843 logger: sends a .cfg to the out-of-box demo over the CLI port,
then reads the data port, parses frames, and logs detected points.

Requires: pip install pyserial
Close the mmWave Demo Visualizer first (only one program can hold a COM port).

Usage (Windows example):
    python awr1843_logger.py --cli COM5 --data COM6 --cfg blindspot_awr1843.cfg --note "car passing, 40 km/h"

Outputs in ./logs/<timestamp>/ :
    points.csv   one row per detected point (host_time, frame, x, y, z, vel, snr_db, noise_db)
    frames.csv   one row per frame (host_time, frame, num_points)
    raw.bin      raw bytes from the data port (for replay / re-parsing)
    note.txt     your note for this recording

Coordinates (TI convention): x = right, y = forward (boresight), z = up, in meters.
Velocity in m/s: negative = approaching the radar, positive = moving away.

Assumes SDK 3.x out-of-box demo output (TLV 1 = detected points as 4 floats,
TLV 7 = side info). If your SDK differs, parse_frame() is the one place to adjust.
"""
import argparse
import csv
import os
import struct
import sys
import time
from datetime import datetime

import serial

MAGIC = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])
HEADER_FMT = "<8sIIIIIIII"  # magic, version, totalLen, platform, frameNum, cpuCycles, numObj, numTLVs, subFrame
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 40 bytes
TLV_HDR_FMT = "<II"
TLV_HDR_SIZE = 8

TLV_DETECTED_POINTS = 1
TLV_SIDE_INFO = 7


# Lines that vary between SDK versions; if the radar rejects one, warn and continue.
OPTIONAL = ("sensorStop", "bpmCfg", "lvdsStreamCfg", "CQRxSatMonitor", "CQSigImgMonitor",
            "analogMonitor", "calibData", "measureRangeBiasAndRxChanPhase",
            "compRangeBiasAndRxChanPhase", "calibDcRangeSig", "extendedMaxVelocity",
            "multiObjBeamForming", "aoaFovCfg", "cfarFovCfg", "lowPower")


def send_cfg(cli, cfg_path):
    """Send each non-comment line of the cfg and wait for 'Done' or an error."""
    if not os.path.exists(cfg_path):  # also look next to this script, not just in the terminal's folder
        beside = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_path)
        if os.path.exists(beside):
            cfg_path = beside
    with open(cfg_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("%")]
    for line in lines:
        cli.write((line + "\n").encode())
        reply = b""
        t0 = time.time()
        while time.time() - t0 < 3.0:
            reply += cli.read(cli.in_waiting or 1)
            if b"Done" in reply or b"Error" in reply or b"not recognized" in reply:
                break
        text = reply.decode(errors="replace").strip().replace("\r", "")
        ok = "Done" in text and "Error" not in text
        print(f"{'OK ' if ok else 'ERR'} {line}")
        if not ok:
            print("   radar replied:", text)
            if line.split()[0] in OPTIONAL:
                print("   (optional line, skipping)")
            else:
                raise RuntimeError(f"Config line failed: {line}")
        time.sleep(0.02)


def parse_frame(packet):
    """Parse one complete packet (bytes starting at the magic word).
    Returns (frame_number, points) where points is a list of
    (x, y, z, vel, snr_db, noise_db)."""
    (_magic, _ver, _total, _plat, frame_num, _cpu,
     num_obj, num_tlvs, _sub) = struct.unpack_from(HEADER_FMT, packet, 0)
    offset = HEADER_SIZE
    xyzv = []
    side = []
    for _ in range(num_tlvs):
        if offset + TLV_HDR_SIZE > len(packet):
            break
        tlv_type, tlv_len = struct.unpack_from(TLV_HDR_FMT, packet, offset)
        offset += TLV_HDR_SIZE
        payload = packet[offset:offset + tlv_len]
        offset += tlv_len
        if tlv_type == TLV_DETECTED_POINTS:
            n = len(payload) // 16
            for i in range(n):
                xyzv.append(struct.unpack_from("<ffff", payload, i * 16))
        elif tlv_type == TLV_SIDE_INFO:
            n = len(payload) // 4
            for i in range(n):
                snr, noise = struct.unpack_from("<HH", payload, i * 4)
                side.append((snr * 0.1, noise * 0.1))  # units of 0.1 dB
    points = []
    for i, (x, y, z, v) in enumerate(xyzv):
        snr, noise = side[i] if i < len(side) else (float("nan"), float("nan"))
        points.append((x, y, z, v, snr, noise))
    return frame_num, points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="COM5", help="CLI port (115200 baud)")
    ap.add_argument("--data", default="COM6", help="data port (921600 baud)")
    ap.add_argument("--cfg", default="blindspot_awr1843.cfg")
    ap.add_argument("--note", default="", help="short description of this recording")
    ap.add_argument("--outdir", default="logs")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until Ctrl+C)")
    args = ap.parse_args()

    run_dir = os.path.join(args.outdir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "note.txt"), "w") as f:
        f.write(f"cfg: {args.cfg}\nnote: {args.note}\n")

    cli = serial.Serial(args.cli, 115200, timeout=0.1)
    data = serial.Serial(args.data, 921600, timeout=0.1)

    print(f"Sending {args.cfg} ...")
    send_cfg(cli, args.cfg)  # ends with sensorStart
    data.reset_input_buffer()
    print(f"Logging to {run_dir}  (Ctrl+C to stop)")

    pts_f = open(os.path.join(run_dir, "points.csv"), "w", newline="")
    frm_f = open(os.path.join(run_dir, "frames.csv"), "w", newline="")
    raw_f = open(os.path.join(run_dir, "raw.bin"), "wb")
    pts = csv.writer(pts_f)
    frm = csv.writer(frm_f)
    pts.writerow(["host_time", "frame", "x_m", "y_m", "z_m", "vel_mps", "snr_db", "noise_db"])
    frm.writerow(["host_time", "frame", "num_points"])

    buf = b""
    start = time.time()
    n_frames = 0
    try:
        while True:
            if args.seconds and time.time() - start > args.seconds:
                break
            chunk = data.read(data.in_waiting or 1)
            if chunk:
                raw_f.write(chunk)
                buf += chunk
            # extract complete packets
            while True:
                i = buf.find(MAGIC)
                if i < 0:
                    buf = buf[-7:]  # keep a possible partial magic word
                    break
                if i > 0:
                    buf = buf[i:]
                if len(buf) < HEADER_SIZE:
                    break
                total_len = struct.unpack_from("<I", buf, 12)[0]
                if total_len < HEADER_SIZE or total_len > 200000:  # corrupt header; resync
                    buf = buf[8:]
                    continue
                if len(buf) < total_len:
                    break
                packet, buf = buf[:total_len], buf[total_len:]
                t = time.time()
                try:
                    frame_num, points = parse_frame(packet)
                except struct.error:
                    continue
                frm.writerow([f"{t:.3f}", frame_num, len(points)])
                for p in points:
                    pts.writerow([f"{t:.3f}", frame_num] + [f"{v:.3f}" for v in p])
                n_frames += 1
                if n_frames % 10 == 0:
                    print(f"frame {frame_num}: {len(points)} points   (logged {n_frames} frames)", end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping sensor ...")
        try:
            cli.write(b"sensorStop\n")
            time.sleep(0.2)
        except Exception:
            pass
        for fobj in (pts_f, frm_f, raw_f):
            fobj.close()
        cli.close()
        data.close()
        print(f"Saved {n_frames} frames to {run_dir}")


if __name__ == "__main__":
    main()