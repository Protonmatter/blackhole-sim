"""Schwarzschild black-hole image renderer.

The renderer traces null geodesics from a camera, detects intersections with a
thin accretion disk, and shades the disk with a compact model that includes:

- radial emissivity,
- gravitational redshift proxy,
- special-relativistic Doppler beaming proxy,
- multi-crossing contribution for higher-order lensed images.

This is an educational renderer. Production scientific imaging normally couples
GRMHD plasma simulations to general-relativistic radiative transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geodesics import normalize, trace_null_geodesic


@dataclass(frozen=True)
class Camera:
    radius: float = 35.0
    inclination_degrees: float = 70.0
    fov_degrees: float = 42.0

    def position(self) -> np.ndarray:
        inc = np.deg2rad(self.inclination_degrees)
        # Disk normal is +z. inc=0 is pole-on, inc=90 is edge-on.
        return np.array([0.0, -self.radius * np.sin(inc), self.radius * np.cos(inc)])

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos = self.position()
        forward = normalize(-pos)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(forward, world_up))) > 0.96:
            world_up = np.array([1.0, 0.0, 0.0])
        right = normalize(np.cross(forward, world_up))
        up = normalize(np.cross(right, forward))
        return pos, forward, right, up


@dataclass(frozen=True)
class DiskModel:
    inner_radius: float = 6.0
    outer_radius: float = 24.0
    ring_radius: float = 8.0
    ring_width: float = 6.0
    emissivity_power: float = 2.2
    optical_depth_per_crossing: float = 0.72
    prograde: bool = True


@dataclass(frozen=True)
class RenderConfig:
    width: int = 480
    height: int = 270
    mass: float = 1.0
    camera: Camera = Camera()
    disk: DiskModel = DiskModel()
    dphi: float = 0.003
    max_steps: int = 10000
    escape_radius: float = 85.0
    exposure: float = 1.2
    gamma: float = 2.2
    background_stars: bool = True


def _hash01(x: float, y: float) -> float:
    # Deterministic shader-style hash, no random state.
    return float(np.mod(np.sin(x * 127.1 + y * 311.7) * 43758.5453123, 1.0))


def _background_color(direction: np.ndarray, stars: bool = True) -> np.ndarray:
    d = normalize(direction)
    # Very dark blue/gray gradient with a sparse procedural star field.
    base = np.array([0.006, 0.008, 0.013]) * (0.85 + 0.15 * max(d[2], -0.5))
    if not stars:
        return base
    theta = np.arctan2(d[1], d[0])
    phi = np.arccos(np.clip(d[2], -1.0, 1.0))
    cell_x = np.floor((theta + np.pi) * 120.0)
    cell_y = np.floor(phi * 75.0)
    h = _hash01(cell_x, cell_y)
    if h > 0.9965:
        strength = (h - 0.9965) / 0.0035
        return base + strength * np.array([0.85, 0.82, 0.75])
    return base


def _disk_color_temperature(intensity: float) -> np.ndarray:
    # A compact blackbody-ish ramp: red/orange -> gold -> white.
    x = float(np.clip(intensity, 0.0, 1.0))
    red = np.array([0.50, 0.10, 0.035])
    gold = np.array([1.00, 0.58, 0.16])
    white = np.array([1.00, 0.93, 0.75])
    if x < 0.55:
        t = x / 0.55
        return (1.0 - t) * red + t * gold
    t = (x - 0.55) / 0.45
    return (1.0 - t) * gold + t * white


def _find_disk_crossings(positions: np.ndarray, disk: DiskModel) -> list[tuple[np.ndarray, int]]:
    hits: list[tuple[np.ndarray, int]] = []
    if len(positions) < 2:
        return hits

    z = positions[:, 2]
    for i in range(len(positions) - 1):
        z0 = z[i]
        z1 = z[i + 1]
        if z0 == 0.0:
            t = 0.0
        elif z0 * z1 > 0.0:
            continue
        else:
            denom = z1 - z0
            if abs(denom) < 1e-15:
                continue
            t = -z0 / denom
            if not (0.0 <= t <= 1.0):
                continue
        p = positions[i] + t * (positions[i + 1] - positions[i])
        rho = float(np.hypot(p[0], p[1]))
        if disk.inner_radius <= rho <= disk.outer_radius:
            hits.append((p, i))
    return hits


def _shade_disk_hit(hit: np.ndarray, camera_pos: np.ndarray, disk: DiskModel, mass: float, order: int) -> np.ndarray:
    r = float(np.hypot(hit[0], hit[1]))
    if r <= 2.0 * mass:
        return np.zeros(3)

    # Radial emissivity with a preferred luminous ring.
    radial = (disk.inner_radius / r) ** disk.emissivity_power
    ring = np.exp(-0.5 * ((r - disk.ring_radius) / max(disk.ring_width, 1e-6)) ** 2)
    emissivity = radial * (0.35 + 1.25 * ring)

    # Approximate Schwarzschild circular-orbit speed measured by a local static observer.
    beta_mag = np.sqrt(max(mass / max(r - 2.0 * mass, 1e-9), 0.0))
    beta_mag = float(np.clip(beta_mag, 0.0, 0.92))
    if not disk.prograde:
        beta_mag *= -1.0
    phi = np.arctan2(hit[1], hit[0])
    beta_vec = beta_mag * np.array([-np.sin(phi), np.cos(phi), 0.0])

    n_to_observer = normalize(camera_pos - hit)
    gamma = 1.0 / np.sqrt(max(1.0 - beta_mag * beta_mag, 1e-12))
    doppler = 1.0 / max(gamma * (1.0 - float(np.dot(beta_vec, n_to_observer))), 1e-6)

    grav_redshift = np.sqrt(max(1.0 - 2.0 * mass / r, 0.0))
    g = doppler * grav_redshift

    # Liouville-like I_nu/nu^3 scaling proxy plus dimmer higher-order crossings.
    intensity = emissivity * (g ** 3) * (disk.optical_depth_per_crossing ** order)
    color = _disk_color_temperature(1.0 - np.exp(-2.0 * intensity))
    return color * intensity


def render_ray(px: int, py: int, cfg: RenderConfig) -> np.ndarray:
    width, height = cfg.width, cfg.height
    cam_pos, forward, right, up = cfg.camera.basis()
    aspect = width / height
    scale = np.tan(0.5 * np.deg2rad(cfg.camera.fov_degrees))

    sx = (2.0 * ((px + 0.5) / width) - 1.0) * aspect * scale
    sy = (1.0 - 2.0 * ((py + 0.5) / height)) * scale
    ray_dir = normalize(forward + sx * right + sy * up)

    result = trace_null_geodesic(
        cam_pos,
        ray_dir,
        mass=cfg.mass,
        dphi=cfg.dphi,
        max_steps=cfg.max_steps,
        escape_radius=cfg.escape_radius,
    )

    color = np.zeros(3)
    hits = _find_disk_crossings(result.positions, cfg.disk)
    for order, (hit, _idx) in enumerate(hits[:4]):
        color += _shade_disk_hit(hit, cam_pos, cfg.disk, cfg.mass, order)

    if not hits and result.status == "escaped" and len(result.positions) >= 2:
        out_dir = normalize(result.positions[-1] - result.positions[-2])
        color += _background_color(out_dir, cfg.background_stars)

    # Captured rays remain black unless they crossed the disk first.
    color = 1.0 - np.exp(-cfg.exposure * color)
    color = np.clip(color, 0.0, 1.0) ** (1.0 / cfg.gamma)
    return color


def render_image(cfg: RenderConfig, *, progress: bool = False) -> np.ndarray:
    img = np.zeros((cfg.height, cfg.width, 3), dtype=np.float32)
    for y in range(cfg.height):
        if progress and (y % max(1, cfg.height // 20) == 0):
            print(f"render row {y + 1}/{cfg.height}")
        for x in range(cfg.width):
            img[y, x] = render_ray(x, y, cfg)
    return img


def save_png(image: np.ndarray, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image, 0.0, 1.0))
