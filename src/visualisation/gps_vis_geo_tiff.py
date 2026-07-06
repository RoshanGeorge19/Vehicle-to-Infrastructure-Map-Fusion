import os

import rasterio
import matplotlib.pyplot as plt
from pyproj import Transformer
import numpy as np
import pandas as pd
import matplotlib.colors as colors

# =========================
# HARD-CODED CONFIG
# =========================
geotiff_path = r"C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/Geotiff Versions/Geotiff/image_modified.tif"

# Folder containing CSV files like rotation_<x>.csv, translationX_<x>.csv, translationY_<x>.csv
csv_dir = r"G:/viva_fsn_car_lidar_pcd_global_map"

# Output folder for plots
output_plot_dir = os.path.join(csv_dir, "transform_plots")
os.makedirs(output_plot_dir, exist_ok=True)

# -------- Sweeps --------
ROT_START = -25
ROT_END = 0
ROT_STEP = 5

TRANS_START = -30
TRANS_END = 0
TRANS_STEP = 5

# Which transform sets to plot
PLOT_ROTATION = False
PLOT_TRANSLATION_X = True
PLOT_TRANSLATION_Y = False

# Plot controls
SHOW_PLOTS = True          # If True, pop up a window for each plot
SAVE_PLOTS = False           # Save PNG per plot
FIGSIZE = (10, 10)
MARKERSIZE = 0.5

# Optional zoom: set to None to disable
# If you want to zoom, set these to map CRS coords (same units as GeoTIFF CRS)
# Example:
# ZOOM_XLIM = (xmin, xmax)
# ZOOM_YLIM = (ymin, ymax)
ZOOM_XLIM = None
ZOOM_YLIM = None


def plot_csv_on_geotiff(ax, rgb_image, extent, transformer, df_filtered):
    """Plot filtered points on the given axes."""
    min_colour_id, max_colour_id = df_filtered["Colour_ID"].min(), df_filtered["Colour_ID"].max()
    colormap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=min_colour_id, vmax=max_colour_id)

    ax.imshow(rgb_image, extent=extent)

    if ZOOM_XLIM is not None:
        ax.set_xlim(*ZOOM_XLIM)
    if ZOOM_YLIM is not None:
        ax.set_ylim(*ZOOM_YLIM)

    labelled_colours = set()

    # Vectorize lon/lat transform for speed
    lons = df_filtered["Longitude"].to_numpy()
    lats = df_filtered["Latitude"].to_numpy()
    map_xs, map_ys = transformer.transform(lons, lats)

    colour_ids = df_filtered["Colour_ID"].to_numpy()

    for map_x, map_y, colour_id in zip(map_xs, map_ys, colour_ids):
        point_color = colormap(norm(colour_id))
        label_name = "car"

        if colour_id == 2:
            point_color = "red"
            label_name = "Ego-Vehicle Point Cloud"
        elif colour_id == 1:
            point_color = "blue"
            label_name = "Fixed Sensor Node Point Cloud"
        elif colour_id == 20:
            point_color = "blue"
        elif colour_id == 30:
            point_color = "green"
        elif colour_id == 40:
            point_color = "yellow"
        elif colour_id == 50:
            point_color = "purple"

        ax.plot(map_x, map_y, marker=".", markersize=MARKERSIZE, color=point_color)

        if colour_id not in labelled_colours:
            ax.plot(map_x, map_y, marker=".", markersize=MARKERSIZE, color=point_color, label=label_name)
            labelled_colours.add(colour_id)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Colour_ID")

    ax.set_xlabel("Map X Coordinate (CRS)")
    ax.set_ylabel("Map Y Coordinate (CRS)")
    plt.grid(True)


def load_and_filter(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Show" not in df.columns:
        raise ValueError(f"'Show' column not found in {csv_path}")
    if "Colour_ID" not in df.columns:
        raise ValueError(f"'Colour_ID' column not found in {csv_path}")

    df_filtered = df[df["Show"] == 1]
    return df_filtered


def handle_one(csv_path: str, title: str, out_png_name: str, rgb_image, extent, transformer):
    if not os.path.exists(csv_path):
        print(f"[SKIP] Missing: {csv_path}")
        return

    df_filtered = load_and_filter(csv_path)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_csv_on_geotiff(ax, rgb_image, extent, transformer, df_filtered)
    ax.set_title(title)

    if SAVE_PLOTS:
        out_png = os.path.join(output_plot_dir, out_png_name)
        plt.savefig(out_png, dpi=600, bbox_inches="tight")
        print(f"[WROTE] {out_png}")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


# =========================
# MAIN
# =========================
with rasterio.open(geotiff_path) as dataset:
    if dataset.count < 3:
        raise ValueError("GeoTIFF doesn't have enough bands for RGB display.")

    red = dataset.read(1)
    green = dataset.read(2)
    blue = dataset.read(3)
    rgb_image = np.dstack((red, green, blue))

    bounds = dataset.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    crs = dataset.crs
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # ---- Rotation sweep ----
    if PLOT_ROTATION:
        for r in range(ROT_START, ROT_END + 1, ROT_STEP):
            csv_path = os.path.join(csv_dir, f"rotation_{r}.csv")
            handle_one(
                csv_path=csv_path,
                title=f"GeoTIFF with points (rotation={r} deg)",
                out_png_name=f"rotation_{r}.png",
                rgb_image=rgb_image,
                extent=extent,
                transformer=transformer
            )

    # ---- Translation X sweep ----
    if PLOT_TRANSLATION_X:
        for tx in range(TRANS_START, TRANS_END + 1, TRANS_STEP):
            csv_path = os.path.join(csv_dir, f"translationX_{tx}.csv")
            handle_one(
                csv_path=csv_path,
                title=f"GeoTIFF with points (translationX={tx} m East)",
                out_png_name=f"translationX_{tx}.png",
                rgb_image=rgb_image,
                extent=extent,
                transformer=transformer
            )

    # ---- Translation Y sweep ----
    if PLOT_TRANSLATION_Y:
        for ty in range(TRANS_START, TRANS_END + 1, TRANS_STEP):
            csv_path = os.path.join(csv_dir, f"translationY_{ty}.csv")
            handle_one(
                csv_path=csv_path,
                title=f"GeoTIFF with points (translationY={ty} m North)",
                out_png_name=f"translationY_{ty}.png",
                rgb_image=rgb_image,
                extent=extent,
                transformer=transformer
            )
