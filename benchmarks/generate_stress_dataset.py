#!/usr/bin/env python3
"""Realistic astrophysical stress-test dataset & action library generator.

Generates the synthetic benchmark dataset under ``benchmarks/data/stress_dataset/``
(configurable via ``--out-dir``):
1. ``sky_patches_catalog.json``: Unique 0.05-deg quantized sky coordinates sized to
   125% of ``--sqlite-max-mb`` (or ``--target-size-mb`` / ``--num-patches``),
   stratified across Showcase, World Capital Zeniths, Dense Galactic Plane,
   Ecliptic Solar System/MOST, and High-Latitude Extragalactic/High-PM regimes.
2. ``cutout_templates.npz``: 32 multi-epoch 6-detector ``(2, 6, 64, 64)`` cutout templates
   with Moffat PSF stars, proper-motion dipoles, zodiacal gradients, and 32-bit FLAGS.
3. ``action_library.json``: Parameterized actions across backend endpoints and
   12 weighted user-flow archetypes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from lib.paths import DATA_DIR, ensure_runtime_dirs, resolve_backend_dir

DEFAULT_OUT_DIR = DATA_DIR / "stress_dataset"
DEFAULT_MB_PER_PATCH = 1.23

SEARCH_TERMS = [
    "mars", "saturn", "jupiter", "uranus", "neptune", "ceres", "vesta", "pallas",
    "juno", "3i/atlas", "atlas", "barnard", "luhman", "wise", "proxima", "vega",
    "sirius", "betelgeuse", "rigel", "altair", "deneb", "polaris", "trappist",
    "andromeda", "orion", "trapezium", "crab", "m31", "m42", "m87", "omega",
    "centauri", "sagittarius", "cygnus", "pleiades", "whirlpool", "sombrero",
    "ring", "eagle", "helix", "rosette", "lagoon", "tarantula", "carina",
    "hydrae", "pegasi", "aquarii", "draconis", "scorpii", "tauri",
]

MOST_BODIES = ["Ceres", "Pallas", "Vesta", "Juno", "Mars", "Jupiter", "Saturn", "C/2025 N1"]


def _galactic_to_icrs_approx(l_deg: float, b_deg: float) -> tuple[float, float]:
    """Convert Galactic (l, b) to approximate ICRS (ra, dec) in degrees."""
    l_r = math.radians(l_deg)
    b_r = math.radians(b_deg)
    ra_ngp = math.radians(192.85948)
    dec_ngp = math.radians(27.12825)
    l_ncp = math.radians(122.93192)
    sin_dec = math.sin(b_r) * math.sin(dec_ngp) + math.cos(b_r) * math.cos(dec_ngp) * math.cos(
        l_ncp - l_r
    )
    dec = math.degrees(math.asin(max(-1.0, min(1.0, sin_dec))))
    y = math.cos(b_r) * math.sin(l_ncp - l_r)
    x = math.sin(b_r) * math.cos(dec_ngp) - math.cos(b_r) * math.sin(dec_ngp) * math.cos(
        l_ncp - l_r
    )
    ra = (math.degrees(math.atan2(y, x)) + math.degrees(ra_ngp)) % 360.0
    return ra, dec


def _ecliptic_to_icrs_approx(lam_deg: float, beta_deg: float) -> tuple[float, float]:
    """Convert Ecliptic (lambda, beta) to ICRS (ra, dec) using J2000 obliquity."""
    eps = math.radians(23.4392911)
    lam = math.radians(lam_deg)
    beta = math.radians(beta_deg)
    sin_dec = math.sin(beta) * math.cos(eps) + math.cos(beta) * math.sin(eps) * math.sin(lam)
    dec = math.degrees(math.asin(max(-1.0, min(1.0, sin_dec))))
    y = math.sin(lam) * math.cos(eps) - math.tan(beta) * math.sin(eps)
    x = math.cos(lam)
    ra = math.degrees(math.atan2(y, x)) % 360.0
    return ra, dec


def _load_all_capitals(cities_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with cities_csv.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "country": row["Country"].strip(),
                    "city": row["Capital City"].strip(),
                    "lat": lat,
                    "lon": lon,
                }
            )
    return rows


def generate_cutout_templates(out_path: Path, num_templates: int = 32, seed: int = 2026) -> None:
    """Create 32 realistic (2, 6, 64, 64) SPHEREx cutout templates (Moffat PSFs, dipoles, flags)."""
    rng = np.random.default_rng(seed)
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    clean_images = np.zeros((num_templates, 2, 6, h, w), dtype=np.float32)
    variances = np.zeros((num_templates, 2, 6, h, w), dtype=np.float32)
    flags_masks = np.zeros((num_templates, 2, 6, h, w), dtype=np.uint32)
    det_offsets = np.arange(6, dtype=np.float32) - np.float32(2.5)

    for tid in range(num_templates):
        regime = tid % 4  # 0=galactic dense, 1=ecliptic zodi, 2=extragalactic dipole, 3=showcase
        n_stars = 28 if regime == 0 else (14 if regime == 1 else (10 if regime == 2 else 18))

        sx_arr = np.empty(n_stars, dtype=np.float32)
        sy_arr = np.empty(n_stars, dtype=np.float32)
        peak_arr = np.empty(n_stars, dtype=np.float32)
        alpha_arr = np.empty(n_stars, dtype=np.float32)
        cslope_arr = np.empty(n_stars, dtype=np.float32)
        for s_idx in range(n_stars):
            sx_arr[s_idx] = float(rng.uniform(4.0, w - 5.0))
            sy_arr[s_idx] = float(rng.uniform(4.0, h - 5.0))
            peak_arr[s_idx] = float(rng.lognormal(mean=1.4, sigma=0.75))
            alpha_arr[s_idx] = float(rng.uniform(1.6, 2.5))
            cslope_arr[s_idx] = float(rng.uniform(-0.18, 0.28))

        # Vectorize static Moffat PSF profiles once per template across all 2 epochs x 6 detectors
        r2_stars = (xx[None, :, :] - sx_arr[:, None, None]) ** 2 + (
            yy[None, :, :] - sy_arr[:, None, None]
        ) ** 2
        moffat_profiles = (1.0 + r2_stars / (alpha_arr[:, None, None] ** 2)) ** np.float32(-2.5)
        fluxes_by_det = peak_arr[:, None] * np.maximum(
            0.25, 1.0 + cslope_arr[:, None] * det_offsets[None, :]
        )
        # Shape: (6, h, w)
        star_cubes = np.tensordot(
            fluxes_by_det.astype(np.float32), moffat_profiles, axes=([0], [0])
        )

        mover_x0 = float(rng.uniform(16.0, 46.0))
        mover_y0 = float(rng.uniform(16.0, 46.0))
        dx = float(rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 7.5))
        dy = float(rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 6.0))
        mover_flux = float(rng.uniform(6.0, 18.0))

        for ep in range(2):
            mx = mover_x0 + (dx if ep == 1 else 0.0)
            my = mover_y0 + (dy if ep == 1 else 0.0)
            mr2 = (xx - mx) ** 2 + (yy - my) ** 2
            mover_psf = (1.0 + mr2 / np.float32(3.2)) ** np.float32(-2.5)
            for det in range(6):
                bg_level = 0.04 + 0.015 * det + (0.06 if regime == 1 else 0.0)
                img = rng.normal(bg_level, 0.015 + 0.003 * det, size=(h, w)).astype(np.float32)
                img += star_cubes[det]

                sed_factor = 0.6 + 0.22 * det if regime == 2 else (1.3 - 0.1 * det)
                img += (np.float32(mover_flux * sed_factor) * mover_psf).astype(np.float32)

                var = (0.002 + 0.0005 * det + 0.004 * np.clip(img, 0.0, 50.0)).astype(np.float32)
                flg = np.zeros((h, w), dtype=np.uint32)

                n_cr = int(rng.integers(2, 6))
                for _ in range(n_cr):
                    cx, cy = int(rng.integers(2, w - 2)), int(rng.integers(2, h - 2))
                    flg[cy, cx] |= 1 << 0
                if regime == 0:
                    flg[img > 25.0] |= 1 << 2

                clean_images[tid, ep, det] = img
                variances[tid, ep, det] = var
                flags_masks[tid, ep, det] = flg

    np.savez_compressed(
        out_path,
        clean_images=clean_images,
        variances=variances,
        flags_masks=flags_masks,
    )


def generate_sky_catalog(
    backend_dir: Path,
    num_patches: int = 2000,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Build ``num_patches`` unique 0.05-deg quantized sky patches across 5 regimes."""
    sys.path.insert(0, str(backend_dir))
    from app.models.contracts import SHOWCASE_TARGETS
    from app.services.spatial_cache import compute_grid_key
    from app.services.zenith_bridge import compute_zenith_icrs

    cities_csv = backend_dir / "app" / "models" / "world_capital_cities.csv"
    rng = random.Random(seed)
    seen_keys: set[str] = set()
    patches: list[dict[str, Any]] = []

    def _add_patch(
        ra: float,
        dec: float,
        regime: str,
        *,
        label: str = "",
        city_meta: dict[str, Any] | None = None,
    ) -> bool:
        ra_n = float(ra) % 360.0
        dec_n = max(-89.5, min(89.5, float(dec)))
        gk, ra_q, dec_q = compute_grid_key(ra_n, dec_n, 5.0)
        if gk in seen_keys:
            return False
        seen_keys.add(gk)

        tid = len(patches) % 32
        n_gaia = rng.randint(55, 130) if regime == "galactic_plane" else rng.randint(14, 48)
        n_simbad = rng.randint(8, 24) if regime == "galactic_plane" else rng.randint(3, 12)
        gaia_rows = [
            {
                "source_id": f"GaiaDR3_{10_000_000 + len(patches) * 200 + j}",
                "ra": round((ra_q + rng.uniform(-0.038, 0.038)) % 360.0, 5),
                "dec": round(max(-89.9, min(89.9, dec_q + rng.uniform(-0.038, 0.038))), 5),
                "phot_g_mean_mag": round(rng.uniform(11.5, 20.4), 2),
                "pmra": round(rng.uniform(-35.0, 45.0), 2),
                "pmdec": round(rng.uniform(-35.0, 45.0), 2),
            }
            for j in range(n_gaia)
        ]
        simbad_rows = [
            {
                "main_id": f"2MASS J{len(patches):04d}{k:02d}",
                "otype": rng.choice(["Star", "PM*", "BD*", "Galaxy", "YSO", "Mira"]),
                "ra": round((ra_q + rng.uniform(-0.035, 0.035)) % 360.0, 5),
                "dec": round(max(-89.9, min(89.9, dec_q + rng.uniform(-0.035, 0.035))), 5),
            }
            for k in range(n_simbad)
        ]

        entry: dict[str, Any] = {
            "idx": len(patches),
            "grid_key": gk,
            "ra": ra_q,
            "dec": dec_q,
            "regime": regime,
            "label": label or f"{regime}_{len(patches)}",
            "template_id": tid,
            "flux_scale": round(rng.uniform(0.75, 1.45), 3),
            "gaia_count": len(gaia_rows),
            "simbad_count": len(simbad_rows),
            "gaia_rows": gaia_rows,
            "simbad_rows": simbad_rows,
        }
        if city_meta is not None:
            entry["city"] = city_meta
        patches.append(entry)
        return True

    for slug, meta in SHOWCASE_TARGETS.items():
        _add_patch(float(meta["ra"]), float(meta["dec"]), "showcase", label=slug)
    extra_hot = [
        ("andromeda_core", 10.6847, 41.2687),
        ("orion_trapezium", 83.8221, -5.3911),
        ("m87_core", 187.7059, 12.3911),
        ("omega_centauri", 201.6970, -47.4795),
        ("galactic_center", 266.4168, -29.0078),
        ("vega_field", 279.2347, 38.7837),
        ("cygnus_x", 310.3580, 45.2800),
        ("pegasus_deep", 344.4127, 20.7688),
    ]
    for slug, ra, dec in extra_hot:
        _add_patch(ra, dec, "showcase", label=slug)

    for cap in _load_all_capitals(cities_csv):
        if len(patches) >= num_patches:
            break
        z = compute_zenith_icrs(lat=cap["lat"], lon=cap["lon"], iso_time="2026-11-21T12:00:00Z")
        _add_patch(
            float(z["zenith_icrs"]["ra"]),
            float(z["zenith_icrs"]["dec"]),
            "capital_zenith",
            label=cap["city"],
            city_meta=cap,
        )

    regimes = ["galactic_plane", "ecliptic_solar", "extragalactic_high_pm"]
    attempts = 0
    while len(patches) < num_patches and attempts < num_patches * 50:
        attempts += 1
        regime = regimes[len(patches) % 3]
        if regime == "galactic_plane":
            ra, dec = _galactic_to_icrs_approx(rng.uniform(0.0, 360.0), rng.uniform(-14.0, 14.0))
        elif regime == "ecliptic_solar":
            ra, dec = _ecliptic_to_icrs_approx(rng.uniform(0.0, 360.0), rng.uniform(-11.0, 11.0))
        else:
            ra = rng.uniform(0.0, 360.0)
            dec = rng.choice([-1.0, 1.0]) * rng.uniform(25.0, 82.0)
        _add_patch(ra, dec, regime)

    return patches[:num_patches]


def generate_action_library(patches: list[dict[str, Any]]) -> dict[str, Any]:
    """Curate parameterized actions across 12 weighted user-flow archetypes."""
    actions: list[dict[str, Any]] = []

    def _add(aid: str, kind: str, endpoint: str, params: dict[str, Any]) -> None:
        actions.append({"id": aid, "kind": kind, "endpoint": endpoint, "params": params})

    for idx, term in enumerate(SEARCH_TERMS):
        for lim in (5, 12, 30):
            _add(
                f"search_{idx}_{lim}",
                "http",
                "/api/v1/catalogs/search",
                {"q": term, "limit": lim},
            )

    city_patches = [p for p in patches if "city" in p]
    for idx, cp in enumerate(city_patches[:140]):
        c = cp["city"]
        _add(
            f"e2s_city_{idx}",
            "http",
            "/api/v1/earth-to-sky",
            {"lat": c["lat"], "lon": c["lon"], "iso_time": "2026-11-21T12:00:00Z"},
        )
    for idx in range(20):
        _add(
            f"e2s_fallback_{idx}",
            "http",
            "/api/v1/earth-to-sky",
            {"iso_time": f"2026-11-21T{(idx % 24):02d}:00:00Z"},
        )

    for idx in range(90):
        _add(
            f"infov_{idx}",
            "http",
            "/api/v1/solar-system/in-fov",
            {
                "ra": round((idx * 17.3) % 360.0, 3),
                "dec": round(-65.0 + (idx % 14) * 10.0, 3),
                "fov": (5.0, 15.0, 35.0, 60.0)[idx % 4],
            },
        )

    showcase = [p for p in patches if p["regime"] == "showcase"]
    for idx, sp in enumerate(showcase[:20]):
        _add(
            f"obj_epochs_{idx}",
            "http",
            "/api/v1/catalogs/object-epochs",
            {
                "object_id": f"showcase:{sp['label']}",
                "label": sp["label"],
                "target_slug": sp["label"],
                "ra": sp["ra"],
                "dec": sp["dec"],
            },
        )
    for idx in range(40):
        _add(
            f"most_{idx}",
            "http",
            "/api/v1/solar-system/most",
            {"obj_name": MOST_BODIES[idx % len(MOST_BODIES)]},
        )

    for idx in range(60):
        _add(
            f"frame_variant_{idx}",
            "frame_mode",
            "/api/v1/frames",
            {
                "pass_index": (idx % 2) + 1,
                "precision": "float32" if idx % 5 == 0 else "float16",
                "accept_gzip": idx % 7 != 0,
                "fetch_flags": idx % 2 == 0,
            },
        )

    modes = ("blink", "subtract", "coadd")
    stretches = ("asinh", "linear", "zscale")
    for idx in range(150):
        min_um = round(0.75 + (idx % 14) * 0.25, 2)
        max_um = round(min(5.0, min_um + 0.5 + (idx % 4) * 0.6), 2)
        _add(
            f"blinker_ui_{idx}",
            "client_blinker",
            "local://pixelEngine",
            {
                "mode": modes[idx % 3],
                "stretch": stretches[(idx // 3) % 3],
                "wavelength_range": [min_um, max_um],
                "zoom": round(1.0 + (idx % 8) * 0.9, 1),
                "click_track": idx % 4 == 0,
            },
        )

    archetype_specs = [
        ("zenith_to_blinker", 22, 55, 85, "common"),
        ("search_blinker_spectra", 18, 60, 95, "common"),
        ("showcase_comparator", 15, 60, 90, "common"),
        ("capital_city_hopper", 12, 55, 80, "common"),
        ("solar_system_most_tracker", 8, 60, 90, "common"),
        ("deep_cold_sky_sweep", 8, 70, 105, "medium"),
        ("spectroscopy_lab_intensive", 5, 65, 100, "medium"),
        ("legacy_float32_identity", 4, 60, 90, "stress"),
        ("thundering_herd_coalescer", 3, 60, 95, "stress"),
        ("rapid_globe_frustum_thrasher", 2, 80, 115, "stress"),
        ("blinker_pixel_torture", 2, 75, 110, "stress"),
        ("polar_boundary_edge", 1, 55, 80, "edge"),
    ]
    archetypes = [
        {
            "name": name,
            "weight": weight,
            "min_actions": min_act,
            "max_actions": max_act,
            "tier": tier,
        }
        for name, weight, min_act, max_act, tier in archetype_specs
    ]

    return {
        "total_unique_actions": len(actions),
        "archetypes": archetypes,
        "actions": actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=None,
        help="Backend directory under test (defaults to STAR_HUNTERS_BACKEND_DIR or repo backend/)",
    )
    parser.add_argument(
        "--sqlite-max-mb",
        type=int,
        default=4096,
        help=(
            "Target SQLite DB cap (MB); auto-sizes to 125%% of this unless "
            "--target-size-mb or --num-patches is set"
        ),
    )
    parser.add_argument(
        "--target-size-mb",
        type=float,
        default=0.0,
        help="Explicit target logical SQLite dataset size in MB (0 = 125%% of --sqlite-max-mb)",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=0,
        help="Explicit unique sky patch count (overrides --target-size-mb and --sqlite-max-mb)",
    )
    parser.add_argument(
        "--mb-per-patch",
        type=float,
        default=DEFAULT_MB_PER_PATCH,
        help="Measured SQLite MB per patch for wire-ready gzip_float16 storage (default: 1.23)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ensure_runtime_dirs()
    backend_dir = resolve_backend_dir(args.backend_dir)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mb_per_patch = max(0.1, float(args.mb_per_patch))
    if args.num_patches > 0:
        num_patches = int(args.num_patches)
        sizing_desc = f"explicit --num-patches {num_patches}"
    elif args.target_size_mb > 0.0:
        num_patches = max(10, int(round(float(args.target_size_mb) / mb_per_patch)))
        sizing_desc = f"target-size {args.target_size_mb:.1f} MB (~{mb_per_patch:.3f} MB/patch)"
    else:
        target_mb = float(args.sqlite_max_mb) * 1.25
        num_patches = max(100, int(round(target_mb / mb_per_patch)))
        sizing_desc = f"125% of {args.sqlite_max_mb} MB SQLite cap (~{target_mb:.1f} MB)"

    print(
        f"Generating stress dataset in {out_dir} (patches={num_patches}, "
        f"~{round(num_patches * mb_per_patch, 1)} MB logical SQLite via {sizing_desc})...",
        flush=True,
    )

    cutout_path = out_dir / "cutout_templates.npz"
    generate_cutout_templates(cutout_path, num_templates=32, seed=args.seed)

    patches = generate_sky_catalog(backend_dir, num_patches=num_patches, seed=args.seed)
    catalog_path = out_dir / "sky_patches_catalog.json"
    catalog_path.write_text(json.dumps(patches, separators=(",", ":")), encoding="utf-8")

    action_lib = generate_action_library(patches)
    action_path = out_dir / "action_library.json"
    action_path.write_text(json.dumps(action_lib, indent=2), encoding="utf-8")

    regime_counts: dict[str, int] = {}
    for p in patches:
        regime_counts[p["regime"]] = regime_counts.get(p["regime"], 0) + 1

    cutout_mb = round(cutout_path.stat().st_size / (1024 * 1024), 2)
    print(
        f"Dataset ready: {len(patches)} unique sky patches ({regime_counts}), "
        f"32 cutout templates ({cutout_mb} MB compressed), "
        f"{action_lib['total_unique_actions']} unique actions across "
        f"{len(action_lib['archetypes'])} archetypes.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
