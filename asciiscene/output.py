"""Ways to show frames. Each takes frame_fn(t) -> uint8 RGB array (height, width, 3)."""

import subprocess

from PIL import Image


def save_png(frame_fn, path, t=0.0):
    Image.fromarray(frame_fn(t)).save(path)


def save_video(frame_fn, path, seconds=12, fps=30):
    """Pipe frames to ffmpeg. Width and height are rounded down to even numbers."""
    first = frame_fn(0.0)
    h, w = first.shape[0] // 2 * 2, first.shape[1] // 2 * 2
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(int(seconds * fps)):
        frame = first if i == 0 else frame_fn(i / fps)
        proc.stdin.write(frame[:h, :w].tobytes())
    proc.stdin.close()
    if proc.wait():
        raise SystemExit("ffmpeg failed")


def run_window(frame_fn, fps=30, title="asciiscene"):
    """Live preview. Needs pygame and a display."""
    import pygame
    pygame.init()
    first = frame_fn(0.0)
    screen = pygame.display.set_mode((first.shape[1], first.shape[0]))
    pygame.display.set_caption(title)
    clock, t = pygame.time.Clock(), 0.0
    while not any(e.type == pygame.QUIT for e in pygame.event.get()):
        pygame.surfarray.blit_array(screen, frame_fn(t).swapaxes(0, 1))
        pygame.display.flip()
        t += clock.tick(fps) / 1000
