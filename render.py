"""Render an ASCII scene.

    python render.py aurora --out aurora.mp4          # video (needs ffmpeg)
    python render.py aurora --png frame.png --time 3  # one still frame
    python render.py aurora                           # live window (needs pygame)
    python render.py aurora --params                  # list tunable parameters
    python render.py aurora --out a.mp4 --size 1080x1920 --set pink=0 --set horizon=0.6
"""

import argparse
import ast

from asciiscene import SCENES, make_frame_fn, run_window, save_png, save_video


def parse_set(items):
    params = {}
    for item in items:
        key, _, value = item.partition("=")
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("scene", choices=sorted(SCENES))
    ap.add_argument("--out", help="write an mp4")
    ap.add_argument("--png", help="write a single still frame")
    ap.add_argument("--params", action="store_true", help="print the scene's parameters and exit")
    ap.add_argument("--size", default="1920x1080", help="WIDTHxHEIGHT in pixels")
    ap.add_argument("--cell", default="5x8", help="pixels per character, WxH")
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--time", type=float, default=3.0, help="scene time for --png")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a scene parameter (repeatable)")
    args = ap.parse_args()

    if args.params:
        for key, value in SCENES[args.scene].PARAMS.items():
            print(f"{key} = {value!r}")
        return
    width, height = map(int, args.size.lower().split("x"))
    cell = tuple(map(int, args.cell.lower().split("x")))
    frame_fn = make_frame_fn(args.scene, width, height, cell, **parse_set(args.set))
    if args.png:
        save_png(frame_fn, args.png, args.time)
    elif args.out:
        save_video(frame_fn, args.out, args.seconds, args.fps)
    else:
        run_window(frame_fn, args.fps, args.scene)


if __name__ == "__main__":
    main()
