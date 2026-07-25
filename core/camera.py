"""FPS-style 3D camera with no collision."""

import numpy as np
from math import cos, sin, radians

# Module-level scratch arrays — avoid per-frame allocation in hot path.
_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_MOVE = np.zeros(3, dtype=np.float32)
_FWD = np.zeros(3, dtype=np.float32)
_RIGHT = np.zeros(3, dtype=np.float32)
_VIEW_M = np.eye(4, dtype=np.float32)
_PROJ_M = np.zeros((4, 4), dtype=np.float32)
_INV_VIEW_M = np.eye(4, dtype=np.float32)


class Camera:
    """
    A 3D fly camera.

    WASD    move forward/left/back/right in the horizontal plane
    Shift   sprint
    Space   move up
    Q       move down
    Mouse   pitch/yaw
    """

    def __init__(
        self,
        pos=(0.0, 5.0, 10.0),
        yaw=0.0,
        pitch=0.0,
        fov=75.0,
        near=0.1,
        far=1000.0,
    ):
        self.pos = np.array(pos, dtype=np.float32)
        # Cached forward vector — invalidated when yaw/pitch change.
        # Initialized before yaw/pitch so the property setters can mark it dirty.
        self._fwd_cache = np.zeros(3, dtype=np.float32)
        self._fwd_dirty = True
        self._yaw = float(yaw)
        self._pitch = float(pitch)
        self.fov = float(fov)
        self.near = float(near)
        self.far = float(far)

        self.speed = 10.0
        self.shift_mult = 2.0
        self.sensitivity = 0.002

        # Keyboard state
        self.keys = set()
        self._mouse_locked = False
        # Cached projection matrix — invalidated when fov/aspect/near/far change.
        self._proj_cache = np.zeros((4, 4), dtype=np.float32)
        self._proj_aspect = -1.0
        self._proj_fov = -1.0
        self._proj_near = -1.0
        self._proj_far = -1.0
        self._inv_proj = None
        self._inv_view = np.eye(4, dtype=np.float32)

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        self._yaw = float(value)
        self._fwd_dirty = True

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        self._pitch = float(value)
        self._fwd_dirty = True

    def forward(self):
        """World-space forward direction (not normalized)."""
        if self._fwd_dirty:
            cp = cos(self.pitch)
            self._fwd_cache[0] = cos(self.yaw) * cp
            self._fwd_cache[1] = sin(self.pitch)
            self._fwd_cache[2] = sin(self.yaw) * cp
            self._fwd_dirty = False
        return self._fwd_cache

    def right(self):
        """World-space right direction (cross of forward and up)."""
        _RIGHT[:] = np.cross(self.forward(), _UP)
        n = np.linalg.norm(_RIGHT)
        if n > 0:
            _RIGHT[:] = _RIGHT / n
        return _RIGHT

    def up(self):
        return _UP

    def rotate(self, dx: float, dy: float):
        # Bypass the property setters and mark dirty once at the end —
        # avoids redundant _fwd_dirty writes per mouse event.
        self._yaw += dx * self.sensitivity
        self._pitch -= dy * self.sensitivity
        limit = np.pi / 2 - 0.01
        self._pitch = float(np.clip(self._pitch, -limit, limit))
        self._fwd_dirty = True

    def update(self, dt: float):
        _MOVE[:] = 0.0
        f = self.forward()
        # Forward flattened to horizontal plane.
        _FWD[0] = f[0]
        _FWD[1] = 0.0
        _FWD[2] = f[2]
        f_norm = np.linalg.norm(_FWD)
        if f_norm > 0:
            _FWD[:] = _FWD / f_norm
        r = self.right()

        speed = self.speed * (self.shift_mult if "Shift" in self.keys else 1.0)

        if "w" in self.keys:
            _MOVE[:] = _MOVE + _FWD
        if "s" in self.keys:
            _MOVE[:] = _MOVE - _FWD
        if "a" in self.keys:
            _MOVE[:] = _MOVE - r
        if "d" in self.keys:
            _MOVE[:] = _MOVE + r
        if "Space" in self.keys:
            _MOVE[:] = _MOVE + _UP
        if "q" in self.keys:
            _MOVE[:] = _MOVE - _UP

        move_len = np.linalg.norm(_MOVE)
        if move_len > 0:
            self.pos += (_MOVE / move_len) * speed * dt

    def view_matrix(self):
        """Return the view matrix. Also fills inv_view_matrix() cache."""
        f = self.forward()
        # Build view + inverse-view directly without np.cross/norm overhead.
        # f is already normalized (unit vector from yaw/pitch).
        # r = normalize(cross(f, up))
        _RIGHT[:] = np.cross(f, _UP)
        rn = np.linalg.norm(_RIGHT)
        if rn > 0:
            _RIGHT[:] = _RIGHT / rn
        # u = cross(r, f)
        _FWD[:] = np.cross(_RIGHT, f)
        u = _FWD
        eye = self.pos
        # View matrix: [r | -dot(r,eye); u | -dot(u,eye); -f | dot(f,eye); 0 0 0 1]
        m = _VIEW_M
        m[0, 0] = _RIGHT[0]; m[0, 1] = _RIGHT[1]; m[0, 2] = _RIGHT[2]
        m[1, 0] = u[0];      m[1, 1] = u[1];      m[1, 2] = u[2]
        m[2, 0] = -f[0];     m[2, 1] = -f[1];     m[2, 2] = -f[2]
        m[0, 3] = -(_RIGHT[0] * eye[0] + _RIGHT[1] * eye[1] + _RIGHT[2] * eye[2])
        m[1, 3] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2])
        m[2, 3] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2]
        m[3, 0] = 0.0; m[3, 1] = 0.0; m[3, 2] = 0.0; m[3, 3] = 1.0
        # Inverse view: [r^T | eye; 0 0 0 1] (rotation is orthonormal).
        iv = _INV_VIEW_M
        iv[0, 0] = _RIGHT[0]; iv[0, 1] = u[0];      iv[0, 2] = -f[0];     iv[0, 3] = eye[0]
        iv[1, 0] = _RIGHT[1]; iv[1, 1] = u[1];      iv[1, 2] = -f[1];     iv[1, 3] = eye[1]
        iv[2, 0] = _RIGHT[2]; iv[2, 1] = u[2];      iv[2, 2] = -f[2];     iv[2, 3] = eye[2]
        iv[3, 0] = 0.0;       iv[3, 1] = 0.0;       iv[3, 2] = 0.0;       iv[3, 3] = 1.0
        self._inv_view = iv
        return m

    def inv_view_matrix(self):
        """Return the cached inverse view matrix (valid after view_matrix())."""
        return self._inv_view

    def projection_matrix(self, aspect: float):
        """Cached projection matrix — only recomputes when params change."""
        if (aspect != self._proj_aspect or self.fov != self._proj_fov
                or self.near != self._proj_near or self.far != self._proj_far):
            m = _PROJ_M
            m[:] = 0.0
            f = 1.0 / np.tan(radians(self.fov) / 2.0)
            m[0, 0] = f / aspect
            m[1, 1] = f
            m[2, 2] = self.far / (self.near - self.far)
            m[2, 3] = self.far * self.near / (self.near - self.far)
            m[3, 2] = -1.0
            self._proj_cache[:] = m
            self._proj_aspect = aspect
            self._proj_fov = self.fov
            self._proj_near = self.near
            self._proj_far = self.far
            # Invalidate cached inverse projection.
            self._inv_proj = None
        return self._proj_cache

    def inv_projection_matrix(self):
        """Cached inverse of the projection matrix."""
        if self._inv_proj is None:
            self._inv_proj = np.linalg.inv(self._proj_cache)
        return self._inv_proj

    def handle_key(self, key: str, down: bool):
        if key == " ":
            key = "Space"
        elif len(key) == 1 and key.isalpha():
            key = key.lower()
        if down:
            self.keys.add(key)
        else:
            self.keys.discard(key)

    def handle_mouse(self, dx: float, dy: float):
        self.rotate(dx, dy)


def look_at(eye, target, up):
    """Build a right-handed view matrix."""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -np.array([np.dot(r, eye), np.dot(u, eye), np.dot(-f, eye)])
    return m


def perspective(fov_y, aspect, near, far):
    """Build a right-handed perspective projection matrix."""
    f = 1.0 / np.tan(fov_y / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    # WebGPU NDC z is [0, 1] (0 near, 1 far), unlike OpenGL's [-1, 1].
    m[2, 2] = far / (near - far)
    m[2, 3] = far * near / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(left, right, bottom, top, near, far):
    """Build a right-handed orthographic projection matrix with NDC z in [0, 1]."""
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -1.0 / (far - near)
    m[2, 3] = -near / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[3, 3] = 1.0
    return m
