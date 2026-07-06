import os
import math
from typing import Tuple

import numpy as np
import pandas as pd

# =========================
# HARD-CODED CONFIG
# =========================
INPUT_CSV = r"G:/viva_fsn_car_lidar_pcd_global_map/GPS_FSN_CAR_LiDAR_in_Global.csv"
OUTPUT_DIR = r"G:/viva_fsn_car_lidar_pcd_global_map"  # where outputs will be written

NAME_COL = "Name"
LON_COL = "Longitude"
LAT_COL = "Latitude"
ALT_COL = "Altitude"

NAME_REGEX = r"^point_node_.*$"

# Rotation sweep (degrees)
ANGLE_START = -30
ANGLE_END = 0
ANGLE_STEP = 5

# Translation sweeps (meters in local ENU)
TRANS_START = -30
TRANS_END = 0
TRANS_STEP = 5

# If your plot behaves like an image (Y down), set True to match visual rotation direction
IMAGE_Y_DOWN = False

# "mean" recommended; "first" uses first point_node_* as pivot
ORIGIN_MODE = "mean"

# Optional: print a small preview for each file
PRINT_PREVIEW = True
PREVIEW_ROWS = 5

# =========================
# WGS84 constants
# =========================
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (N + h_m) * cos_lat * cos_lon
    y = (N + h_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + h_m) * sin_lat
    return x, y, z


def ecef_to_geodetic(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = WGS84_A * (1.0 - WGS84_F)
    ep2 = (WGS84_A**2 - b**2) / (b**2)

    lon = np.arctan2(y, x)
    p = np.sqrt(x * x + y * y)

    theta = np.arctan2(z * WGS84_A, p * b)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    lat = np.arctan2(
        z + ep2 * b * (sin_t**3),
        p - WGS84_E2 * WGS84_A * (cos_t**3)
    )

    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    cos_lat = np.cos(lat)

    h = np.where(
        np.abs(cos_lat) > 1e-12,
        p / cos_lat - N,
        np.abs(z) - (N * (1.0 - WGS84_E2)),
    )

    return np.rad2deg(lat), np.rad2deg(lon), h


def ecef_to_enu(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                lat0_deg: float, lon0_deg: float,
                x0: float, y0: float, z0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    sin_lat0 = math.sin(lat0)
    cos_lat0 = math.cos(lat0)
    sin_lon0 = math.sin(lon0)
    cos_lon0 = math.cos(lon0)

    e = -sin_lon0 * dx + cos_lon0 * dy
    n = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    u =  cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz
    return e, n, u


def enu_to_ecef(e: np.ndarray, n: np.ndarray, u: np.ndarray,
                lat0_deg: float, lon0_deg: float,
                x0: float, y0: float, z0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    sin_lat0 = math.sin(lat0)
    cos_lat0 = math.cos(lat0)
    sin_lon0 = math.sin(lon0)
    cos_lon0 = math.cos(lon0)

    dx = -sin_lon0 * e - sin_lat0 * cos_lon0 * n + cos_lat0 * cos_lon0 * u
    dy =  cos_lon0 * e - sin_lat0 * sin_lon0 * n + cos_lat0 * sin_lon0 * u
    dz =  cos_lat0 * n + sin_lat0 * u

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz
    return x, y, z


def rotate_enu(e: np.ndarray, n: np.ndarray, angle_deg: float, image_y_down: bool) -> Tuple[np.ndarray, np.ndarray]:
    ang = math.radians(angle_deg)
    if image_y_down:
        ang = -ang
    c = math.cos(ang)
    s = math.sin(ang)
    e2 = c * e - s * n
    n2 = s * e + c * n
    return e2, n2


def apply_transform_and_write(base_df: pd.DataFrame,
                              node_mask_series: pd.Series,
                              mask: np.ndarray,
                              lat_origin: float, lon_origin: float, alt_origin: float,
                              xO: float, yO: float, zO: float,
                              e: np.ndarray, n: np.ndarray, u: np.ndarray,
                              angle_deg: float = 0.0,
                              tx_m: float = 0.0,
                              ty_m: float = 0.0,
                              out_path: str = "") -> None:
    """
    Apply rotation (deg) in EN plane + translation (meters) in EN plane:
      - tx_m is East translation
      - ty_m is North translation
    Writes CSV with Lon/Lat/Alt overwritten for point_node_* rows.
    """
    # Rotate
    e2, n2 = rotate_enu(e, n, float(angle_deg), IMAGE_Y_DOWN)

    # Translate in local ENU
    e2 = e2 + float(tx_m)
    n2 = n2 + float(ty_m)

    # Back to geodetic
    xr, yr, zr = enu_to_ecef(e2, n2, u, lat_origin, lon_origin, xO, yO, zO)
    lat_r, lon_r, alt_r = ecef_to_geodetic(xr, yr, zr)

    out_df = base_df.copy()
    out_df.loc[mask, LON_COL] = lon_r
    out_df.loc[mask, LAT_COL] = lat_r
    out_df.loc[mask, ALT_COL] = alt_r

    out_df.to_csv(out_path, index=False)

    if PRINT_PREVIEW:
        preview = out_df.loc[node_mask_series, [NAME_COL, LON_COL, LAT_COL, ALT_COL]].head(PREVIEW_ROWS)
        print(f"\nWrote: {out_path}")
        print(preview.to_string(index=False))


def main():
    base_df = pd.read_csv(INPUT_CSV)

    for col in (NAME_COL, LON_COL, LAT_COL, ALT_COL):
        if col not in base_df.columns:
            raise ValueError(f"Missing required column '{col}'. Found: {list(base_df.columns)}")

    name_series = base_df[NAME_COL].astype(str)
    node_mask_series = name_series.str.match(NAME_REGEX, na=False)
    node_mask = node_mask_series.to_numpy()

    lon0_all = pd.to_numeric(base_df[LON_COL], errors="coerce").to_numpy(dtype=float)
    lat0_all = pd.to_numeric(base_df[LAT_COL], errors="coerce").to_numpy(dtype=float)
    alt0_all = pd.to_numeric(base_df[ALT_COL], errors="coerce").to_numpy(dtype=float)

    finite_mask = np.isfinite(lon0_all) & np.isfinite(lat0_all) & np.isfinite(alt0_all)
    mask = node_mask & finite_mask

    if not np.any(mask):
        print("No valid point_node_* rows found to transform (check Name regex and numeric columns).")
        return

    # Choose ENU origin/pivot from ORIGINAL coords
    if ORIGIN_MODE == "mean":
        lat_origin = float(np.mean(lat0_all[mask]))
        lon_origin = float(np.mean(lon0_all[mask]))
        alt_origin = float(np.mean(alt0_all[mask]))
    elif ORIGIN_MODE == "first":
        i0 = int(np.where(mask)[0][0])
        lat_origin = float(lat0_all[i0])
        lon_origin = float(lon0_all[i0])
        alt_origin = float(alt0_all[i0])
    else:
        raise ValueError("ORIGIN_MODE must be 'mean' or 'first'")

    xO, yO, zO = geodetic_to_ecef(np.array([lat_origin]), np.array([lon_origin]), np.array([alt_origin]))
    xO, yO, zO = float(xO[0]), float(yO[0]), float(zO[0])

    print(f"Origin (lat, lon, alt): {lat_origin:.8f}, {lon_origin:.8f}, {alt_origin:.3f}")
    print(f"Matched rows: {int(node_mask.sum())} | Transformable rows: {int(mask.sum())}")

    # Precompute ENU for original points once
    x, y, z = geodetic_to_ecef(lat0_all[mask], lon0_all[mask], alt0_all[mask])
    e, n, u = ecef_to_enu(x, y, z, lat_origin, lon_origin, xO, yO, zO)

    # =========================
    # 1) Rotation sweep: rotation_<angle>.csv
    # =========================
    for angle in range(ANGLE_START, ANGLE_END + 1, ANGLE_STEP):
        out_path = os.path.join(OUTPUT_DIR, f"rotation_{angle}.csv")
        apply_transform_and_write(
            base_df, node_mask_series, mask,
            lat_origin, lon_origin, alt_origin, xO, yO, zO,
            e, n, u,
            angle_deg=float(angle), tx_m=0.0, ty_m=0.0,
            out_path=out_path
        )

    # =========================
    # 2) Translation X sweep (East): translationX_<tx>.csv
    # =========================
    for tx in range(TRANS_START, TRANS_END + 1, TRANS_STEP):
        out_path = os.path.join(OUTPUT_DIR, f"translationX_{tx}.csv")
        apply_transform_and_write(
            base_df, node_mask_series, mask,
            lat_origin, lon_origin, alt_origin, xO, yO, zO,
            e, n, u,
            angle_deg=0.0, tx_m=float(tx), ty_m=0.0,
            out_path=out_path
        )

    # =========================
    # 3) Translation Y sweep (North): translationY_<ty>.csv
    # =========================
    for ty in range(TRANS_START, TRANS_END + 1, TRANS_STEP):
        out_path = os.path.join(OUTPUT_DIR, f"translationY_{ty}.csv")
        apply_transform_and_write(
            base_df, node_mask_series, mask,
            lat_origin, lon_origin, alt_origin, xO, yO, zO,
            e, n, u,
            angle_deg=0.0, tx_m=0.0, ty_m=float(ty),
            out_path=out_path
        )


if __name__ == "__main__":
    main()
