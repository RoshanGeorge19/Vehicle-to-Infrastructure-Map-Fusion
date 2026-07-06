import numpy as np
import pickle
import matplotlib.pyplot as plt
from wp2.geo_utils import Plotter
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd
import os
import pylas
import plotly.graph_objects as go
from collections import defaultdict

def plot_error_vs_distance(diff_df):
    formatter = FuncFormatter(Plotter.format_tick_value)
    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                       (14, 15), (15, 16), (16, 17), (17, 18), (18, 19)]
    range_labels = ['8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m', '14-15m',
                    '15-16m', '16-17m', '17-18m', '18-19m']
    x_plot_range = None

    heading_columns = [col for col in diff_df.columns if col.startswith('Diff_')]
    headings = []
    for col in heading_columns:
        try:
            # Extract the heading error percentage from the column name
            heading = float(col.split('_')[1].split('-')[1])
            headings.append(heading)
        except ValueError:
            pass

    compass_list = [35.77, 35.89, 36.05, 36.14, 36.27, 36.82, 36.88]
    compass_heading = compass_list[0] - 15
    heading_node = [compass_heading + i * 0.1 for i in headings]
    heading_node = [round(value, 2) for value in heading_node]

    # Apply x_plot_range to filter headings and adjust heading_node and heading_columns accordingly
    if x_plot_range is not None:
        filtered_indices = [i for i, h in enumerate(heading_node) if h >= x_plot_range[0] and h <= x_plot_range[1]]
        heading_node = [heading_node[i] for i in filtered_indices]
        heading_columns = [heading_columns[i] for i in filtered_indices]

    range_data = {}
    for i, (lower, upper) in enumerate(distance_ranges):
        # Filter rows that fall within the current distance range
        rows_in_range = diff_df[(diff_df['node_base_to_node_obj'] >= lower) &
                                (diff_df['node_base_to_node_obj'] < upper)]
        if not rows_in_range.empty:
            range_data[range_labels[i]] = rows_in_range

    # Plot graphs for each distance range
    for range_label, data_in_range in range_data.items():
        # Initialize a new figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot individual rows within the current range
        for idx, row in data_in_range.iterrows():
            # Only include filtered heading columns
            diff_values = [row[col] for col in heading_columns]
            time_car = row['Time_Car'] if 'Time_Car' in row else 'N/A'
            node_base_to_node_obj = row['node_base_to_node_obj']

            ax.plot(heading_node, diff_values, marker='o',
                    label=f'Row: {idx}, Time: {time_car}, Distance: {node_base_to_node_obj:.2f}m')

        ax.legend(loc='best', fontsize='small')  # Show a legend for trace identification

        ax.set_title('Relative Error vs. Heading (20 degree)')
        ax.set_ylabel('Error (m)')
        ax.set_xlabel('Heading (degree)')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))  # Major Tick every 500ms
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))  # Minor Tick every 100ms
        ax.yaxis.set_major_locator(MultipleLocator(0.1))  # Major Tick every 1m
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))  # Minor Tick every 0.1m
        ax.xaxis.set_major_formatter(formatter)  # Apply formatting to x-axis ticks
        ax.yaxis.set_major_formatter(formatter)  # Apply formatting to y-axis ticks
        ax.grid(which='major', linestyle='-', linewidth=0.75, color='gray')  # Major gridlines
        ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')  # Minor gridlines

        if x_plot_range is not None:
            ax.set_xlim(left=x_plot_range[0], right=x_plot_range[1])
        else:
            ax.autoscale(True, axis='x')  # Automatically scale x-axis if no x_plot_range is set

        fig.tight_layout()  # Adjust plot spacing
        plt.show()

def plot_error_vs_distance_interactive(diff_df, x_plot_range=None):
    distance_ranges = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                       (14, 15), (15, 16), (16, 17), (17, 18), (18, 19)]
    range_labels = [
        '8-9m', '9-10m', '10-11m', '11-12m', '12-13m', '13-14m', '14-15m',
        '15-16m', '16-17m', '17-18m', '18-19m']

    x_plot_range = None

    # Extract difference columns (based on headings)
    heading_columns = [col for col in diff_df.columns if col.startswith('Diff_')]
    headings = []
    for col in heading_columns:
        try:
            heading = float(col.split('_')[1].split('-')[1])  # Extract heading value
            headings.append(heading)
        except ValueError:
            pass

    compass_list = [35.77, 35.89, 36.05, 36.14, 36.27, 36.82, 36.88]
    compass_heading = compass_list[0] - 15
    heading_node = [compass_heading + i * 0.1 for i in headings]
    heading_node = [round(value, 2) for value in heading_node]

    # Apply x_plot_range to filter headings and corresponding columns
    if x_plot_range is not None:
        filtered_indices = [i for i, h in enumerate(heading_node) if h >= x_plot_range[0] and h <= x_plot_range[1]]
        heading_node = [heading_node[i] for i in filtered_indices]
        heading_columns = [heading_columns[i] for i in filtered_indices]

    # Create a Plotly figure
    fig = go.Figure()

    # Add individual row traces
    for idx, row in diff_df.iterrows():
        # Only include filtered heading columns
        diff_values = [row[col] for col in heading_columns]
        time_car = row['Time_Car'] if 'Time_Car' in row else 'N/A'
        node_base_to_node_obj = row['node_base_to_node_obj'] if 'node_base_to_node_obj' in row else 'N/A'

        fig.add_trace(go.Scatter(
            x=heading_node,
            y=diff_values,
            mode='lines+markers',
            name=f'Row: {idx}, Time: {time_car}, Distance: {node_base_to_node_obj:.2f}m',
            visible='legendonly'  # Initially hidden for clarity
        ))

    # Compute and add range averages
    range_averages = defaultdict(list)
    for _, row in diff_df.iterrows():
        node_base_to_node_obj = row['node_base_to_node_obj']
        diff_values = np.array([row[col] for col in heading_columns])

        # Determine the range label for this row
        for i, (lower, upper) in enumerate(distance_ranges):
            if lower <= node_base_to_node_obj < upper:
                range_averages[range_labels[i]].append(diff_values)

    for range_label, values in range_averages.items():
        avg_values = np.nanmean(values, axis=0)  # Compute the mean across rows
        fig.add_trace(go.Scatter(
            x=heading_node,
            y=avg_values,
            mode='lines+markers',
            name=f'Average ({range_label})',
            visible=True
        ))

    # Update layout
    fig.update_layout(
        title='Relative Error vs. Heading Error',
        xaxis_title='Heading Error (%)',
        yaxis_title='Error (m)',
        legend_title='Rows and Averages (Click to toggle)',
        hovermode='x',
        template='plotly',
        xaxis=dict(
            tickmode='array',
            tickvals=heading_node,
            showgrid=True,
            range=x_plot_range if x_plot_range is not None else None),
        yaxis=dict(showgrid=True),)

    fig.show()

def main():
    plotter = Plotter()
    with open('pickles/scenario_3_exp_2_3.pkl', 'rb') as f:
        data = pickle.load(f)

    eucl_dist_dict = data['eucl_dist_dict']
    relative_error_dict = data['relative_error_dict']
    relative_error_diff_df = data['relative_error_diff_df']

    plot_error_vs_distance_interactive(relative_error_diff_df)
    plot_error_vs_distance(relative_error_diff_df)

if __name__ == '__main__':
    main()