import dash
import pandas as pd
import rasterio
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from rasterio.enums import Resampling
from rasterio.warp import reproject

# Your GeoTIFF path here
geotiff_path = "C:/Users/Roshan George/OneDrive - National University of Ireland, Galway/NUIG/PhD/Publications/P2_V2I Map Fusion/Images/Geotiff Versions/Geotiff/image_modified.tif"

# Load GeoTIFF and process RGB bands
def load_geotiff_rgb(path):
    """Load GeoTIFF and process RGB bands."""
    with rasterio.open(path) as dataset:
        # Read the RGB bands. Assuming GeoTIFF has at least 3 bands.
        red = dataset.read(1)
        green = dataset.read(2)
        blue = dataset.read(3)

        # Stack the RGB bands into an image
        rgb_image = np.dstack((red, green, blue))

        # Calculate the bounds (extent) for visualization
        bounds = dataset.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        return rgb_image, extent


def resample_geotiff(src_image, scale_factor):
    """Resample an RGB image by reducing its resolution for better performance."""
    rows, cols, channels = src_image.shape
    new_rows = int(rows * scale_factor)
    new_cols = int(cols * scale_factor)

    # Create an array to hold the resampled image
    resized_image = np.empty((new_rows, new_cols, channels), dtype=np.uint8)

    # Resample each channel (Red, Green, Blue)
    for i in range(3):
        data = src_image[:, :, i]
        dest = np.empty((new_rows, new_cols), dtype=data.dtype)

        # Perform resampling (bilinear resampling used here)
        reproject(
            source=data,
            destination=dest,
            src_transform=rasterio.transform.Affine.identity(),
            src_crs="EPSG:4326",
            dst_transform=rasterio.transform.Affine.scale((cols / new_cols), (rows / new_rows)),
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear  # Use bilinear resampling
        )

        # Save resampled channel back into the resized image
        resized_image[:, :, i] = dest.astype(np.uint8)

    return resized_image


# Main GeoTIFF image and extent loading
rgb_image, extent = load_geotiff_rgb(geotiff_path)

# Reduce image size for performance reasons
scale_factor = 0.25  # Reduce to 25% of original
rgb_image_resampled = resample_geotiff(rgb_image, scale_factor)

# Normalize the resampled image RGB values to 0-255 range
rgb_image_resampled = (rgb_image_resampled - rgb_image_resampled.min()) / \
                      (rgb_image_resampled.max() - rgb_image_resampled.min()) * 255
rgb_image_resampled = rgb_image_resampled.astype(np.uint8)

# CSV File Path (GPS Points)
csv_file_path = "G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Scenario-2/visualisation/GPS_lidar_las_in_global.csv"

# Initialize points as an empty list globally
points = []  # Storage for GPS points to be plotted
colors = []  # Storage for the colours from the Colour_ID

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "GeoTIFF Visualisation"

# Define HTML layout
app.layout = html.Div([
    dcc.Input(id="longitude", type="number", placeholder="Longitude"),
    dcc.Input(id="latitude", type="number", placeholder="Latitude"),
    dcc.Input(id="altitude", type="number", placeholder="Altitude"),
    dcc.Graph(id="geomap")
], style={'backgroundColor': '#EDEDED'})


# Define callback function to update the graph
@app.callback(
    Output('geomap', 'figure'),
    [Input('longitude', 'value'), Input('latitude', 'value'), Input('altitude', 'value')]
)
def update_output_div(longitude, latitude, altitude):
    global points, colors  # Global points list & colors

    # If the user entered longitude/latitude manually, append to points
    if longitude is not None and latitude is not None:
        points.append({'longitude': longitude, 'latitude': latitude})
        colors.append(100)  # Assign a default color when user enters manually (since there's no Colour_ID)

    # If no manually entered point, load points from CSV
    if len(points) == 0:
        try:
            gps = pd.read_csv(csv_file_path)

            # Ensure valid points (ignore zero Long/Lat values)
            valid_gps = gps[(gps["Longitude"] != 0) & (gps["Latitude"] != 0)]

            # Map Colour_ID from CSV
            if "Colour_ID" in valid_gps.columns:
                colors.extend(valid_gps["Colour_ID"])
            else:
                colors.extend([100] * len(valid_gps))  # Default color code if Colour_ID is missing

            # Add valid GPS points to the list
            for _, row in valid_gps.iterrows():
                points.append({'longitude': row["Longitude"], 'latitude': row["Latitude"]})

        except Exception as e:
            print(f"Error reading CSV file: {e}")

    # Plot the resampled image for faster performance
    img_trace = go.Image(
        z=rgb_image_resampled,
        x0=extent[0],  # Set the minimum longitude
        y0=extent[3],  # Top latitude
        dx=(extent[1] - extent[0]) / rgb_image_resampled.shape[1],  # Longitude pixel distance
        dy=(extent[2] - extent[3]) / rgb_image_resampled.shape[0]  # Latitude pixel distance (negative for top-down)
    )

    # Create scatter plot for the GPS points with Colour_ID as the color scale
    scatter_trace = go.Scatter(
        x=[point['longitude'] for point in points],
        y=[point['latitude'] for point in points],
        mode='markers',
        marker=dict(
            color=colors,  # Use colors list derived from Colour_ID
            colorscale='inferno',  # Apply inferno color scale
            showscale=True,  # Show the color bar
            size=5,  # Marker size
            colorbar=dict(title='Colour_ID'),  # Title for the color bar
        ),
        name="Plotted Points"
    )

    # Create the figure with image and scatter plot
    geomap = go.Figure(data=[img_trace, scatter_trace])

    # Configure axes scaling and layout
    geomap.update_xaxes(range=[extent[0], extent[1]])
    geomap.update_yaxes(range=[extent[2], extent[3]])

    # Configure layout (zoom, pan, etc.)
    geomap.update_layout(
        title="GeoTIFF RGB Image with Dynamic Points",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=700,  # Define figure height
        margin=dict(r=0, t=0, l=0, b=0),
        dragmode="zoom",  # Enable Zoom via mouse drag
        hovermode="closest",  # Ensure closest point hover interaction
        autosize=True,
        uirevision=True,  # Retain zoom state when updated dynamically
        yaxis=dict(autorange=False),
        xaxis=dict(autorange=False)
    )

    return geomap


# Run the Dash app server
if __name__ == '__main__':
    app.run_server(debug=True)
