"""Deterministic scripted playback for performance benchmarking.

Boots the renderer, ascends to a high altitude, pitches the camera straight
down (so the fog ring is visible on the ground below), then sprints forward
while looking down for a fixed duration.  Uses a fixed frame dt so every run
produces an identical camera path — terrain is seed-deterministic, and radius
adjustment is disabled so the chunk load set stays constant.

Usage:
    .venv\\Scripts\\python.exe playback.py

Controls (still active during playback):
    Escape  quit early
    F1      toggle debug HUD
"""

import os

os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan")

import time
import numpy as np
from rendercanvas.glfw import loop

from main import App
from sky import compute_sky_params

# ---------------------------------------------------------------------------
# Script configuration — tweak these to change the playback sequence.
# ---------------------------------------------------------------------------
FRAME_DT = 1.0 / 60.0               # fixed dt for deterministic motion
REALTIME_PACE = True                # sleep to match sim time to wall-clock time
TARGET_ALTITUDE = 80.0              # moderate altitude — overview without fog issues
LOOK_DOWN_PITCH = -(np.pi / 2 - 0.01)  # near-straight-down (camera clamp limit)
ASCEND_SPEED = 200.0                # m/s during ascent phase (fast climb)
PAN_SPEED = 60.0                    # m/s forward sprint while looking down
PAN_TIME_SEC = 52.0                 # how long to sprint forward while looking down
WARMUP_FRAMES = 300                 # let chunks settle after boot (5s @ 60fps)
LOOK_DOWN_FRAMES = 30               # pitch to look-down over 0.5s
HOLD_FRAMES = 90                    # hold position after script before exiting (1.5s)
# Total: 5s warmup + 0.4s ascend + 0.5s look_down + 52s pan + 1.5s hold ≈ 59.4s


class PlaybackApp(App):
    """App subclass with a scripted, deterministic camera path."""

    def __init__(self):
        # Prevent the parent from grabbing the mouse — we control the camera
        # entirely from the script, and mouse input would perturb yaw/pitch.
        self._skip_mouse_grab = True
        super().__init__()

        # Reset camera to known initial state.
        self.camera.pos = np.array([0.0, 8.0, 25.0], dtype=np.float32)
        self.camera.yaw = -np.pi / 2
        self.camera.pitch = 0.0
        self.camera.keys.clear()

        # Override chunk radius for playback — slightly wider than default
        # so terrain doesn't vanish when panning at altitude, but not so
        # wide it overwhelms memory (radius 12 crashed the window).
        self.chunk_manager.radius = 10
        self.chunk_manager.min_radius = 10
        self.chunk_manager.max_radius = 10

        self._frame_count = 0
        self._phase = "warmup"
        self._phase_frame = 0
        # Frame limiter: pace sim time to wall-clock so 60s sim = 60s real.
        self._pace_origin = time.perf_counter()
        self._pace_frame = 0

    # Disable mouse grab so the script has full camera control.
    def _grab_mouse(self):
        if getattr(self, "_skip_mouse_grab", False):
            return
        super()._grab_mouse()

    def _scripted_update(self, dt):
        """Drive the camera through deterministic phases."""
        self._phase_frame += 1

        if self._phase == "warmup":
            # Camera stays still; chunks load around initial position.
            if self._phase_frame >= WARMUP_FRAMES:
                self._phase = "ascend"
                self._phase_frame = 0
                print(f"[PLAYBACK] warmup done ({self._frame_count} frames), ascending to {TARGET_ALTITUDE}m...")

        elif self._phase == "ascend":
            # Move straight up at constant speed.
            if self.camera.pos[1] < TARGET_ALTITUDE:
                self.camera.pos[1] += ASCEND_SPEED * dt
            else:
                self.camera.pos[1] = TARGET_ALTITUDE
                self._phase = "look_down"
                self._phase_frame = 0
                print(f"[PLAYBACK] reached {self.camera.pos[1]:.0f}m, pitching to look-down...")

        elif self._phase == "look_down":
            # Smoothly pitch to straight-down over LOOK_DOWN_FRAMES.
            t = min(self._phase_frame / float(LOOK_DOWN_FRAMES), 1.0)
            # Smoothstep for ease-in-out.
            t = t * t * (3.0 - 2.0 * t)
            self.camera.pitch = LOOK_DOWN_PITCH * t
            if self._phase_frame >= LOOK_DOWN_FRAMES:
                self.camera.pitch = LOOK_DOWN_PITCH
                self._phase = "pan"
                self._phase_frame = 0
                print(f"[PLAYBACK] looking down (pitch={np.degrees(self.camera.pitch):.1f}deg), "
                      f"panning {PAN_TIME_SEC}s...")

        elif self._phase == "pan":
            # Sprint forward (horizontally, in the yaw direction) while
            # looking straight down. Terrain scrolls below the camera.
            # Forward is flattened to the horizontal plane so altitude stays
            # constant — the camera keeps looking down at the ground.
            pan_frames = int(PAN_TIME_SEC / FRAME_DT)
            fwd = self.camera.forward()
            fx, fz = float(fwd[0]), float(fwd[2])
            fn = (fx * fx + fz * fz) ** 0.5
            if fn > 1e-6:
                fx, fz = fx / fn, fz / fn
            self.camera.pos[0] += fx * PAN_SPEED * dt
            self.camera.pos[2] += fz * PAN_SPEED * dt
            self.camera.pos[1] = TARGET_ALTITUDE
            if self._phase_frame >= pan_frames:
                self._phase = "hold"
                self._phase_frame = 0
                print(f"[PLAYBACK] pan done at pos={self.camera.pos}")

        elif self._phase == "hold":
            if self._phase_frame >= HOLD_FRAMES:
                print("[PLAYBACK] script complete, exiting.")
                self._phase = "done"
                self.canvas.close()

    def _scripted_draw(self):
        """Identical to App._draw but with fixed dt and scripted camera."""
        if self._phase == "done":
            return
        self.profiler.frame_start()

        dt = FRAME_DT

        # Advance day/night cycle at the fixed rate.
        self.time_of_day = (self.time_of_day + dt * (24.0 / self.day_length_sec)) % 24.0

        # Scripted camera control (replaces self.camera.update(dt)).
        self._scripted_update(dt)
        if self._phase == "done":
            return

        compute_time = self._update_chunks()

        width, height = self.renderer.get_physical_size()
        aspect = width / height if height else 1.0
        view = self.camera.view_matrix()

        sky = compute_sky_params(self.time_of_day)
        if sky["sun_intensity"] > 0.01:
            light = sky["sun_dir"]
            sun_color = sky["sun_color"] * sky["sun_intensity"]
        else:
            light = sky["moon_dir"]
            sun_color = sky["moon_color"] * sky["moon_intensity"]

        render_distance = self.chunk_manager.radius * self.chunk_manager.chunk_size
        self.camera.far = max(2500.0, render_distance * 1.5)
        proj = self.camera.projection_matrix(aspect)

        # Altitude-aware fog: push fog start/end outward when the camera is
        # high up so terrain below isn't swallowed by fog. At ground level,
        # fog uses the standard render_distance-based values. At 150m+, fog
        # starts near the far edge of loaded terrain so the ground stays
        # visible while distant terrain still fades naturally.
        # fog_start is kept close to the loaded-terrain edge so the chunks
        # that ARE loaded stay visible — previously fog_start sat at 0.6x
        # render_distance, which fogged out the outer ~25% of loaded chunks
        # and made it look like chunks weren't loading at all.
        altitude = float(self.camera.pos[1])
        alt_factor = min(max((altitude - 20.0) / 130.0, 0.0), 1.0)  # 0 at 20m, 1 at 150m
        fog_start = render_distance * (0.8 + 0.15 * alt_factor)  # 0.8→0.95
        fog_end = render_distance * (1.4 + 0.4 * alt_factor)     # 1.4→1.8
        fog_density = 4.605 / max(fog_end - fog_start, 1.0)
        fog_color = sky["sky_horizon"]

        t0 = time.perf_counter()
        self.renderer.draw(view, proj, light, self.camera.pos, fog_density, fog_start,
                           fog_color, sun_color=sun_color, sky_params=sky,
                           inv_view_proj=self.camera.inv_projection_matrix() @ self.camera.inv_view_matrix())
        render_time = time.perf_counter() - t0

        summary = self.profiler.frame_end(render_time=render_time, compute_time=compute_time)
        if summary:
            # Do NOT adjust radius — keep chunk load set identical across runs.
            self.profiler.log(summary, self.chunk_manager)

        self._frame_count += 1

        # Frame limiter: sleep so wall-clock keeps pace with sim time.
        # This makes 60s of scripted simulation take ~60s real time.
        if REALTIME_PACE:
            self._pace_frame += 1
            target_wall = self._pace_origin + self._pace_frame * FRAME_DT
            slack = target_wall - time.perf_counter()
            if slack > 0:
                time.sleep(slack)

    def run(self):
        # Use the scripted draw callback instead of the parent's _draw.
        self.canvas.request_draw(self._scripted_draw)
        print("Entering scripted playback loop...")
        loop.run()
        print("Playback loop exited.")


if __name__ == "__main__":
    app = PlaybackApp()
    app.run()
