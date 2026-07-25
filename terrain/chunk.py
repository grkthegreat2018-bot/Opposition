"""Single chunk terrain mesh generation."""
import time
import numpy as np
from terrain._noise_core import _PERM2, _GRAD2
from terrain.biomes import _compute_biome
from terrain.fbm import _compute_height
from terrain.erosion import _thermal_erode, smooth_terrain
from terrain.features import _apply_rivers, _apply_plateaus, _apply_canyons, _apply_craters, _apply_glacial_valleys
from terrain.continental import (
    _continental_mask, _continental_elevation,
    _coastal_ridge_weight, _ocean_flatness,
)


def _build_chunk(spec: dict):
    """Top-level worker used by the process pool to build one Chunk.

    Must be picklable, so it takes a plain dict and returns plain data.
    """
    c = Chunk(**spec)
    mesh = c.build()
    return c.key(), mesh, getattr(c, "build_time", 0.0)


class Chunk:
    """A single cubic chunk of terrain (surface only)."""

    def __init__(
        self,
        cx: int,
        cy: int,
        cz: int,
        size: float = 32.0,
        grid_res: int = 24,
        seed: int = 0,
        scale: float = 25.0,
        freq: float = 0.008,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        warp_scale: float = 0.3,
        warp_amp: float = 3.0,
        ridge_weight: float = 0.35,
        detail_weight: float = 0.12,
        biome_freq: float = 0.0015,
        # Continental-scale landmass parameters (Phase 3). All seam-safe.
        continental_freq: float = 0.0002,
        sea_level: float = 0.0,
        ocean_depth: float = 18.0,
        land_boost: float = 12.0,
        coastal_peak: float = 0.65,
        coastal_width: float = 0.18,
        coastal_mountain_strength: float = 0.7,
        ocean_transition: float = 0.06,
        erosion_iters: int = 0,
        erosion_talus: float = 1.5,
        erosion_factor: float = 0.25,
        hydraulic_droplets: int = 0,
        wind_erode_iters: int = 0,
        # Large-scale terrain features (rivers/plateaus/canyons/craters).
        # All seam-safe (pure functions of world position). Set to 0 to disable.
        river_depth: float = 0.0,
        plateau_strength: float = 0.0,
        canyon_depth: float = 0.0,
        crater_strength: float = 0.0,
        smooth_strength: float = 0.0,
        glacial_strength: float = 0.0,
    ):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.size = size
        self.grid_res = grid_res
        self.seed = seed
        self.scale = scale
        self.freq = freq
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.warp_scale = warp_scale
        self.warp_amp = warp_amp
        self.ridge_weight = ridge_weight
        self.detail_weight = detail_weight
        self.biome_freq = biome_freq
        self.continental_freq = continental_freq
        self.sea_level = sea_level
        self.ocean_depth = ocean_depth
        self.land_boost = land_boost
        self.coastal_peak = coastal_peak
        self.coastal_width = coastal_width
        self.coastal_mountain_strength = coastal_mountain_strength
        self.ocean_transition = ocean_transition
        self.erosion_iters = erosion_iters
        self.erosion_talus = erosion_talus
        self.erosion_factor = erosion_factor
        self.hydraulic_droplets = hydraulic_droplets
        self.wind_erode_iters = wind_erode_iters
        self.river_depth = river_depth
        self.plateau_strength = plateau_strength
        self.canyon_depth = canyon_depth
        self.crater_strength = crater_strength
        self.smooth_strength = smooth_strength
        self.glacial_strength = glacial_strength
        self.mesh = None

    def key(self):
        return (self.cx, self.cy, self.cz)

    def bounds(self):
        x = self.cx * self.size
        y = self.cy * self.size
        z = self.cz * self.size
        return x, y, z

    def build(self):
        """Generate a 2D heightfield mesh for this xz column. Returns a dict or None."""
        t0 = time.perf_counter()
        x0, _, z0 = self.bounds()
        x1 = x0 + self.size
        z1 = z0 + self.size

        n = self.grid_res
        x = np.linspace(x0, x1, n, dtype=np.float32)
        z = np.linspace(z0, z1, n, dtype=np.float32)
        dx = x[1] - x[0]
        dz = z[1] - z[0]

        # Extended height map for continuous normals at chunk edges and for
        # seam-safe thermal erosion. Padding >= erosion_iters ensures the
        # thermal erosion of mesh-border cells uses real neighbour values
        # (not edge-clamped) on both sides, keeping results identical across
        # chunk boundaries since _compute_height is a pure function of world
        # position.
        pad = max(self.erosion_iters, 4) + 1
        x_ext = np.linspace(x0 - pad * dx, x1 + pad * dx, n + 2 * pad, dtype=np.float32)
        z_ext = np.linspace(z0 - pad * dz, z1 + pad * dz, n + 2 * pad, dtype=np.float32)
        xx_ext, zz_ext = np.meshgrid(x_ext, z_ext, indexing="ij")

        # Biome weights on the extended grid so the height field uses the
        # biome blend consistently all the way to the chunk border (and
        # beyond, for edge-normal continuity and seam-safe erosion).
        biome_ext = _compute_biome(xx_ext, zz_ext, seed=self.seed, biome_freq=self.biome_freq)
        h_ext = self._compute_height(xx_ext, zz_ext, biome=biome_ext)

        # --- Large-scale terrain features (rivers/plateaus/canyons/craters) ---
        # All are pure functions of world position → seam-safe across chunks.
        # Applied before thermal erosion so erosion can round off any too-sharp
        # transitions. Each is gated by a strength parameter (0 = disabled).
        if self.river_depth > 0.0:
            h_ext = _apply_rivers(xx_ext, zz_ext, h_ext, self.scale,
                                  self.seed, _PERM2, _GRAD2,
                                  river_depth=self.river_depth)
        if self.plateau_strength > 0.0:
            h_ext = _apply_plateaus(xx_ext, zz_ext, h_ext, self.scale,
                                    self.seed, _PERM2, _GRAD2,
                                    strength=self.plateau_strength)
        if self.canyon_depth > 0.0:
            h_ext = _apply_canyons(xx_ext, zz_ext, h_ext, self.scale,
                                   self.seed, _PERM2, _GRAD2,
                                   canyon_depth=self.canyon_depth)
        if self.crater_strength > 0.0:
            h_ext = _apply_craters(xx_ext, zz_ext, h_ext, self.scale,
                                   self.seed,
                                   crater_prob=0.18 * self.crater_strength)
        if self.glacial_strength > 0.0:
            h_ext = _apply_glacial_valleys(xx_ext, zz_ext, h_ext, self.scale,
                                           self.seed, _PERM2, _GRAD2,
                                           strength=self.glacial_strength)

        # Thermal (talus) erosion on the extended grid. Seam-safe because the
        # heightfield is a pure function of world position and the padding is
        # large enough that mesh-border erosion doesn't reach the clamped
        # edges. Thermal erosion rounds off sharp peaks and produces natural
        # talus slopes — the key cure for the bed-of-nails artifact. Hydraulic
        # and wind erosion remain disabled (hydraulic uses random per-chunk
        # droplets that can't match across seams; wind's reach exceeds
        # practical padding). Their visual effects are handled analytically
        # in the fragment shader (gully detail, dune elongation).
        if self.erosion_iters > 0:
            h_ext = _thermal_erode(
                h_ext, iters=self.erosion_iters,
                talus=self.erosion_talus, factor=self.erosion_factor,
            )

        # Optional Gaussian smoothing pass to flatten tiny spikes and sharp
        # edges. Runs on the extended grid so seam safety is preserved.
        if self.smooth_strength > 0.0:
            h_ext = smooth_terrain(h_ext, strength=self.smooth_strength, iters=1)

        sediment_ext = np.zeros_like(h_ext)
        lap_ext = np.zeros_like(h_ext)
        lap_ext[1:-1, 1:-1] = (
            h_ext[:-2, 1:-1] + h_ext[2:, 1:-1]
            + h_ext[1:-1, :-2] + h_ext[1:-1, 2:]
            - 4.0 * h_ext[1:-1, 1:-1]
        )

        # Interior height and normals.
        h = h_ext[pad:pad + n, pad:pad + n]
        # np.gradient returns ndarrays; Pyright stubs mis-infer with scalar spacings.
        gh_x: np.ndarray
        gh_z: np.ndarray
        gh_x, gh_z = np.gradient(h_ext, dx, dz)  # type: ignore[assignment]
        nx = -gh_x[pad:pad + n, pad:pad + n]
        nz = -gh_z[pad:pad + n, pad:pad + n]
        ny = np.ones_like(nx)
        normals_grid = np.stack([nx, ny, nz], axis=-1)
        nrm = np.linalg.norm(normals_grid, axis=-1, keepdims=True)
        nrm[nrm == 0.0] = 1.0
        normals_grid = (normals_grid / nrm).astype(np.float32)

        # Vertex positions.
        xx, zz = np.meshgrid(x, z, indexing="ij")
        vertices_grid = np.stack([xx, h, zz], axis=-1).astype(np.float32)
        vertices = vertices_grid.reshape(-1, 3)
        normals = normals_grid.reshape(-1, 3)
        biome = biome_ext[pad:pad + n, pad:pad + n, :].reshape(-1, 4).astype(np.float32)
        # Sediment deposition (interior), normalized to [0, 1] for shading.
        sediment = sediment_ext[pad:pad + n, pad:pad + n].reshape(-1).astype(np.float32)
        sed_max = float(sediment.max()) if sediment.size else 0.0
        if sed_max > 1e-6:
            sediment = np.clip(sediment / sed_max, 0.0, 1.0)
        else:
            sediment = np.zeros_like(sediment)
        # Curvature (interior), normalized to [-1, 1]. Positive = concave
        # (valleys), negative = convex (ridges).
        # IMPORTANT: use a FIXED scale derived from terrain parameters, not
        # per-chunk max. Per-chunk normalization amplifies flat chunks'
        # tiny curvature to ±1.0 while compressing mountain chunks', causing
        # a discontinuity at chunk boundaries — the same world position gets
        # different curvature values depending on which chunk computes it,
        # producing visible color lines (rock/snow masks differ).
        curvature = lap_ext[pad:pad + n, pad:pad + n].reshape(-1).astype(np.float32)
        curvature_scale = max(4.0 * self.scale, 1.0)
        curvature = np.clip(curvature / curvature_scale, -1.0, 1.0)

        # Indices for a regular grid with CCW, upward-facing triangles.
        i = np.arange(n - 1)
        j = np.arange(n - 1)
        ii, jj = np.meshgrid(i, j, indexing="ij")
        i0 = ii * n + jj
        i1 = (ii + 1) * n + jj
        i2 = ii * n + (jj + 1)
        i3 = (ii + 1) * n + (jj + 1)
        faces = np.empty((n - 1, n - 1, 2, 3), dtype=np.uint32)
        faces[:, :, 0, :] = np.stack([i0, i3, i1], axis=-1)
        faces[:, :, 1, :] = np.stack([i0, i2, i3], axis=-1)
        faces = faces.reshape(-1, 3)

        # --- Skirts: vertical strips around the chunk perimeter to hide
        # any residual sub-pixel seams between chunks. Each perimeter
        # vertex gets a duplicate lowered by skirt_depth in Y, connected
        # to its neighbour by two triangles forming a vertical wall.
        # This is the standard game-industry defence against terrain gaps.
        # Skirt depth scales with the chunk's height range so cliffs between
        # high and low neighbouring chunks are always fully covered.
        perim_heights = h.reshape(n, n)
        h_range = float(perim_heights.max() - perim_heights.min())
        skirt_depth = np.float32(max(8.0, h_range * 0.5 + 4.0))
        # Perimeter vertex indices in order (clockwise when viewed from +Y).
        perim = []
        perim += list(range(0, n))                      # top row (z=z0)
        perim += list(range(2 * n - 1, n * n, n))       # right col (x=x1)
        perim += list(range(n * n - 1, n * (n - 1) - 1, -1))  # bottom row (z=z1)
        perim += list(range(n * (n - 1), -1, -n))       # left col (x=x0)
        perim = np.array(perim, dtype=np.uint32)
        # De-duplicate the 4 corners (each appears twice in the path above).
        # We keep the duplicate so the skirt has a clean closed loop; the
        # tiny overdraw at corners is invisible.
        n_perim = perim.shape[0]
        # Skirt vertices: copies of perimeter vertices with Y lowered.
        skirt_verts = vertices[perim].copy()
        skirt_verts[:, 1] = skirt_verts[:, 1] - skirt_depth
        # Outward-facing normals for the skirt wall. The skirt is a vertical
        # cliff face, so its normals must point horizontally outward from the
        # chunk center — NOT along the terrain surface (which points up and
        # makes exposed skirts appear as brightly-lit strips). A slight
        # downward tilt (-0.3 y) ensures the cliff catches less top-down
        # light, appearing as a shadowed rock face that blends naturally
        # with neighbouring chunks of different heights.
        cx_center = x0 + self.size * 0.5
        cz_center = z0 + self.size * 0.5
        out_dx = skirt_verts[:, 0] - cx_center
        out_dz = skirt_verts[:, 2] - cz_center
        out_len = np.sqrt(out_dx * out_dx + out_dz * out_dz)
        out_len[out_len == 0.0] = 1.0
        out_dx = (out_dx / out_len).astype(np.float32)
        out_dz = (out_dz / out_len).astype(np.float32)
        skirt_normals = np.stack([out_dx, np.full(out_dx.shape, -0.3, dtype=np.float32), out_dz], axis=-1)
        nrm_skirt = np.linalg.norm(skirt_normals, axis=-1, keepdims=True)
        skirt_normals = (skirt_normals / nrm_skirt).astype(np.float32)
        skirt_biome = biome[perim].copy()
        skirt_sediment = sediment[perim].copy()
        skirt_curvature = curvature[perim].copy()
        # Append skirt vertices.
        base_count = vertices.shape[0]
        vertices = np.concatenate([vertices, skirt_verts], axis=0)
        normals = np.concatenate([normals, skirt_normals], axis=0)
        biome = np.concatenate([biome, skirt_biome], axis=0)
        sediment = np.concatenate([sediment, skirt_sediment], axis=0)
        curvature = np.concatenate([curvature, skirt_curvature], axis=0)
        # Skirt indices: for each consecutive pair of perimeter vertices,
        # connect (top_a, top_b, bot_b) and (top_a, bot_b, bot_a).
        # bot_i = base_count + i (skirt vertex index). Vectorized — the closed
        # loop wraps the last pair back to index 0 via np.roll.
        k_next = np.roll(np.arange(n_perim, dtype=np.uint32), -1)
        ta = perim
        tb = perim[k_next]
        ba = base_count + np.arange(n_perim, dtype=np.uint32)
        bb = base_count + k_next
        skirt_faces = np.empty((n_perim, 2, 3), dtype=np.uint32)
        skirt_faces[:, 0, 0] = ta
        skirt_faces[:, 0, 1] = tb
        skirt_faces[:, 0, 2] = bb
        skirt_faces[:, 1, 0] = ta
        skirt_faces[:, 1, 1] = bb
        skirt_faces[:, 1, 2] = ba
        skirt_faces = skirt_faces.reshape(-1, 3)
        faces = np.concatenate([faces, skirt_faces], axis=0)

        # Tight bounding box for culling (include skirt depth).
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        self.bbox = (
            float(vmin[0]),
            float(vmin[1]),
            float(vmin[2]),
            float(vmax[0]),
            float(vmax[1]),
            float(vmax[2]),
        )

        self.mesh = {
            "vertices": vertices,
            "normals": normals,
            "biome": biome,
            "sediment": sediment,
            "curvature": curvature,
            "indices": faces,
        }
        self.tri_count = faces.shape[0]
        self.vert_count = vertices.shape[0]
        self.build_time = time.perf_counter() - t0
        return self.mesh

    def _compute_height(self, xx, zz, biome=None):
        """Return the height field for this chunk's x/z grid."""
        return _compute_height(
            xx,
            zz,
            seed=self.seed,
            freq=self.freq,
            scale=self.scale,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
            warp_scale=self.warp_scale,
            warp_amp=self.warp_amp,
            ridge_weight=self.ridge_weight,
            detail_weight=self.detail_weight,
            ridge_detail_weight=getattr(self, 'ridge_detail_weight', 0.08),
            biome=biome,
            continental_freq=self.continental_freq,
            sea_level=self.sea_level,
            ocean_depth=self.ocean_depth,
            land_boost=self.land_boost,
            coastal_peak=self.coastal_peak,
            coastal_width=self.coastal_width,
            coastal_mountain_strength=self.coastal_mountain_strength,
            ocean_transition=self.ocean_transition,
        )
