"""Entry point: dream 3D renderer with infinite chunked terrain and fly camera."""

import os

# Force wgpu to use the Vulkan backend; it may otherwise fall back to OpenGL.
os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan")

import time
import numpy as np
import glfw
from rendercanvas.glfw import RenderCanvas, loop

import core.config as config
from core.config import Config
from terrain import ChunkManager
from core.camera import Camera
from render.renderer import TerrainRenderer
from core.profiler import PerformanceProfiler
from render.sky import compute_sky_params
from render.debug_hud import DebugHUD


def _ray_aabb(origin: np.ndarray, direction: np.ndarray, bbox: tuple) -> tuple[float | None, float | None]:
    """Ray-AABB intersection. Returns (tmin, tmax) or (None, None) if no hit."""
    x0, y0, z0, x1, y1, z1 = bbox
    inv_d = 1.0 / (direction + 1e-12)
    t1 = (x0 - origin[0]) * inv_d[0]
    t2 = (x1 - origin[0]) * inv_d[0]
    t3 = (y0 - origin[1]) * inv_d[1]
    t4 = (y1 - origin[1]) * inv_d[1]
    t5 = (z0 - origin[2]) * inv_d[2]
    t6 = (z1 - origin[2]) * inv_d[2]
    tmin: float = max(min(t1, t2), min(t3, t4), min(t5, t6))
    tmax: float = min(max(t1, t2), max(t3, t4), max(t5, t6))
    if tmax < 0 or tmin > tmax:
        return None, None
    return tmin, tmax


def set_fullscreen(canvas):
    """Make the glfw window full-screen on the primary monitor."""
    window = canvas._window
    if window is None:
        return
    monitor = glfw.get_primary_monitor()
    if not monitor:
        return
    mode = glfw.get_video_mode(monitor)
    x, y = glfw.get_monitor_pos(monitor)
    glfw.set_window_monitor(
        window, monitor, x, y, mode.size.width, mode.size.height, mode.refresh_rate
    )


def center_window(canvas):
    """Center the glfw window on the primary monitor (windowed mode)."""
    window = canvas._window
    if window is None:
        return
    monitor = glfw.get_primary_monitor()
    if not monitor:
        return
    mode = glfw.get_video_mode(monitor)
    mx, my = glfw.get_monitor_pos(monitor)
    w, h = glfw.get_window_size(window)
    glfw.set_window_pos(window, mx + (mode.size.width - w) // 2, my + (mode.size.height - h) // 2)


def _prewarm_numba(cfg):
    """Build one tiny chunk synchronously to compile + cache all numba-jit'd
    functions in the main process. Worker processes then load the disk cache
    instead of recompiling (~200ms saved per worker on first run).
    """
    import time as _t
    t0 = _t.perf_counter()
    from terrain.chunk import Chunk
    # Same tuning as real chunks (so the same jitted specialisations are
    # compiled), just on a tiny grid.
    Chunk(**cfg.prewarm_chunk_kwargs()).build()
    print(f"Numba pre-warm: {_t.perf_counter() - t0:.2f}s (cached to disk for workers)")


class App:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or config.DEFAULT
        cfg = self.cfg

        disp = cfg.display
        self.canvas = RenderCanvas(size=(disp.width, disp.height), title=disp.title,
                                   update_mode="continuous")
        if disp.fullscreen:
            set_fullscreen(self.canvas)
        else:
            center_window(self.canvas)
        self.renderer = TerrainRenderer(self.canvas)

        cam = cfg.camera
        self.camera = Camera(pos=cam.start_pos, yaw=cam.start_yaw, pitch=cam.start_pitch,
                             fov=cam.fov, near=cam.near)
        self.camera.speed = cam.speed
        self.camera.shift_mult = cam.shift_mult
        self.camera.sensitivity = cam.sensitivity

        self._last_time = time.time()
        self._mouse_locked = False

        # Debug HUD: F1 toggle, F2 freeze chunks, F3 mark wrongfully culled chunk.
        self.debug_hud = DebugHUD(self.renderer.device, self.renderer.format)
        self._debug_enabled = False
        self._chunks_frozen = False
        self._marked_chunks = []  # log of (chunk_key, cam_pos, yaw, pitch)

        self.time_of_day = cfg.time.start_hour
        self.day_length_sec = cfg.time.day_length_sec

        self.profiler = PerformanceProfiler(log_interval=cfg.profiler_log_interval)

        self._bind_events()
        self._grab_mouse()

        # Numba prewarm runs on a background thread so the window appears
        # immediately with a loading screen instead of blocking ~6.5s.
        # Chunk manager creation is deferred until prewarm completes, since
        # its worker processes would otherwise recompile every jitted function.
        self._prewarm_thread = None
        self._prewarm_done = False
        self._prewarm_started = False
        self.chunk_manager = None
        self._start_prewarm()

        # Loading-screen HUD text.
        self.debug_hud.update_text([
            "COMPILING TERRAIN SHADERS...",
            "PLEASE WAIT",
        ])

    def _start_prewarm(self):
        import threading
        def _worker():
            _prewarm_numba(self.cfg)
            self._prewarm_done = True
        self._prewarm_started = True
        self._prewarm_thread = threading.Thread(
            target=_worker, name="numba-prewarm", daemon=True)
        self._prewarm_thread.start()

    def _finish_startup(self):
        """Create the chunk manager and do the initial chunk load.

        Called from _draw once the prewarm thread signals completion.
        """
        if self._prewarm_thread is not None:
            self._prewarm_thread.join()
            self._prewarm_thread = None
        self.chunk_manager = ChunkManager(**self.cfg.chunk_manager_kwargs())
        self.debug_hud.update_text([])
        # Reset the frame clock so the first real frame doesn't get a 6s dt
        # (which would jerk the camera and jump the day/night cycle).
        self._last_time = time.time()
        # Initial chunk load
        self._update_chunks()

    def _bind_events(self):
        self.canvas.add_event_handler(self._on_key_down, "key_down")
        self.canvas.add_event_handler(self._on_key_up, "key_up")
        self.canvas.add_event_handler(self._on_resize, "resize")

    def _on_key_down(self, event):
        key = event.get("key", "")
        self.camera.handle_key(key, True)
        if key == "Escape":
            self.canvas.close()
        elif not self._prewarm_done:
            # Ignore debug toggles during the loading screen.
            return
        elif key == "F1":
            self._debug_enabled = not self._debug_enabled
            print(f"[DEBUG] HUD {'ON' if self._debug_enabled else 'OFF'}")
            if not self._debug_enabled:
                self.debug_hud.set_selected_chunk()
                self.debug_hud.set_hovered_chunk()
                self.debug_hud.update_text([])
        elif key == "F2":
            self._chunks_frozen = not self._chunks_frozen
            print(f"[DEBUG] Chunks {'FROZEN' if self._chunks_frozen else 'UNFROZEN'}")
        elif key == "F3" and self._debug_enabled:
            self._mark_current_chunk()

    def _on_key_up(self, event):
        key = event.get("key", "")
        self.camera.handle_key(key, False)

    def _on_resize(self, event):
        pass

    def _grab_mouse(self):
        """Lock and hide the cursor, focus the window on boot."""
        window = self.canvas._window
        if window is None:
            return

        glfw.focus_window(window)
        if glfw.raw_mouse_motion_supported():
            glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        glfw.set_cursor_pos_callback(window, self._raw_mouse_callback)

        w, h = glfw.get_window_size(window)
        glfw.set_cursor_pos(window, w / 2, h / 2)
        self._mouse_locked = True

    def _raw_mouse_callback(self, window, x, y):
        """Recenter the cursor and turn absolute position into a delta."""
        w, h = glfw.get_window_size(window)
        cx, cy = w / 2.0, h / 2.0
        dx = x - cx
        dy = y - cy
        self.camera.handle_mouse(dx, dy)
        glfw.set_cursor_pos(window, cx, cy)

    def _update_chunks(self):
        if self._chunks_frozen:
            return 0.0
        compute_time = 0.0
        t0 = time.perf_counter()
        changed, new_chunks, removed = self.chunk_manager.update(self.camera.pos)
        compute_time += time.perf_counter() - t0
        if changed:
            t1 = time.perf_counter()
            self.renderer.add_chunks(new_chunks)
            self.renderer.remove_chunks(removed)
            compute_time += time.perf_counter() - t1
        return compute_time

    def _raycast_chunk(self):
        """Find the chunk bbox the camera is looking at via ray-AABB intersection."""
        origin = self.camera.pos.astype(np.float64)
        direction = self.camera.forward().astype(np.float64)
        d_norm = np.linalg.norm(direction)
        if d_norm < 1e-6:
            return None, None
        direction = direction / d_norm
        best_t = float("inf")
        best_key = None
        best_bbox = None
        for key, cd in self.renderer.chunk_meshes.items():
            bbox = cd.bbox
            if bbox is None:
                continue
            tmin, tmax = _ray_aabb(origin, direction, bbox)
            if tmin is not None and tmax is not None and tmin < tmax and tmin < best_t and tmax > 0:
                best_t = tmin
                best_key = key
                best_bbox = bbox
        return best_key, best_bbox

    def _mark_current_chunk(self):
        """Mark the chunk at screen center as wrongfully culled, log pos/angle."""
        key, bbox = self._raycast_chunk()
        if key is None:
            return
        entry = {
            "chunk": key,
            "cam_pos": self.camera.pos.copy(),
            "yaw": self.camera.yaw,
            "pitch": self.camera.pitch,
            "bbox": bbox,
        }
        self._marked_chunks.append(entry)
        self.debug_hud.set_selected_chunk(key=key, bbox=bbox)
        print(f"[MARK] Chunk {key} marked at pos={self.camera.pos}, "
              f"yaw={self.camera.yaw:.4f}, pitch={self.camera.pitch:.4f}")

    def _draw(self):
        # Loading screen while numba prewarm runs on a background thread.
        if not self._prewarm_done:
            # Always show the HUD during loading (it carries the "Compiling..."
            # message); _debug_enabled is ignored until startup finishes.
            self.renderer.draw_loading(self.debug_hud)
            # Check again next frame; once done, finish startup in-place.
            if self._prewarm_done:
                self._finish_startup()
            return

        self.profiler.frame_start()

        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        # Advance day/night cycle.
        self.time_of_day = (self.time_of_day + dt * (24.0 / self.day_length_sec)) % 24.0

        self.camera.update(dt)
        compute_time = self._update_chunks()

        width, height = self.renderer.get_physical_size()
        aspect = width / height if height else 1.0
        view = self.camera.view_matrix()

        # Compute sky parameters from time of day.
        sky = compute_sky_params(self.time_of_day)
        # Use the brighter of sun/moon as the primary light source.
        if sky["sun_intensity"] > self.cfg.time.night_threshold:
            light = sky["sun_dir"]
            sun_color = sky["sun_color"] * sky["sun_intensity"]
        else:
            light = sky["moon_dir"]
            sun_color = sky["moon_color"] * sky["moon_intensity"]

        # Loaded radius in meters; keep the far plane beyond it but within a
        # depth-friendly range. Fog fades the chunk edge to the sky color.
        # Extra margin for the cloud plane (4000m wide centered on camera).
        render_distance = self.chunk_manager.radius * self.chunk_manager.chunk_size
        self.camera.far = max(self.cfg.camera.min_far, render_distance * 1.5)
        proj = self.camera.projection_matrix(aspect)

        fog_density, fog_start = self.cfg.fog.params(
            render_distance, float(self.camera.pos[1]),
        )
        fog_color = sky["sky_horizon"]  # fog matches horizon sky color

        t0 = time.perf_counter()
        debug_hud = self.debug_hud if self._debug_enabled else None
        if self._debug_enabled:
            hud_lines = [
                f"POS  X={self.camera.pos[0]:8.2f} Y={self.camera.pos[1]:8.2f} Z={self.camera.pos[2]:8.2f}",
                f"YAW  {self.camera.yaw:8.4f} RAD  {np.degrees(self.camera.yaw):8.2f} DEG",
                f"PITC {self.camera.pitch:8.4f} RAD  {np.degrees(self.camera.pitch):8.2f} DEG",
                f"FROZEN {'YES' if self._chunks_frozen else 'NO'}  MARKED {len(self._marked_chunks)}",
                f"FPS RAD={self.chunk_manager.radius} CHUNKS={len(self.renderer.chunk_meshes)}",
                f"F1 HUD  F2 FREEZE  F3 MARK",
            ]
            self.debug_hud.update_text(hud_lines)
            hkey, hbbox = self._raycast_chunk()
            if hbbox is not None and not self._marked_chunks:
                self.debug_hud.set_hovered_chunk(key=hkey, bbox=hbbox)
            elif not self._marked_chunks:
                self.debug_hud.set_hovered_chunk()
        self.renderer.draw(view, proj, light, self.camera.pos, fog_density, fog_start,
                           fog_color, sun_color=sun_color, sky_params=sky,
                           debug_hud=debug_hud,
                           inv_view_proj=self.camera.inv_projection_matrix() @ self.camera.inv_view_matrix())
        render_time = time.perf_counter() - t0

        summary = self.profiler.frame_end(render_time=render_time, compute_time=compute_time)
        if summary:
            self.chunk_manager.adjust_radius(summary["compute"]["peak"] * 1000.0)
            self.profiler.log(summary, self.chunk_manager)

    def run(self):
        self.canvas.request_draw(self._draw)
        print("Entering loop...")
        loop.run()
        print("Loop exited.")


if __name__ == "__main__":
    app = App()
    app.run()
