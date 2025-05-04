# to test and implement GPT-generated code
import cProfile
import numpy as np
import time

import pandas as pd

from src import config
from src.timetable import TI2C


def test1():
    ...


def test2():
    ...


def test3():
    ...


def test4():
    ...


def test5():
    ...


def plot_timetable_plotly(TT: pd.DataFrame, line_nid: int = 1, updown: list = [1, -1],
                          include_train_load: bool = False,
                          xylabel_fontsize: int = 10):
    """
    Plot train timetable with train load shown as line colors and a color legend (using Plotly).

    :param TT: DataFrame containing the train timetable data.
    :param line_nid: The line number for which to filter the data.
    :param updown: The direction of the train, 1 for up, -1 for down.
    :param include_train_load: Whether to include train load data in the plot.
    :param xylabel_fontsize: Font size for x and y labels.
    :return: None
    """
    import plotly.graph_objects as go
    from matplotlib import pyplot as plt
    from matplotlib import cm
    # Filter timetable for specific line and directions
    filtered_tt = TT[(TT['LINE_NID'] == line_nid) &
                     (TT['UPDOWN'].isin(updown))]

    # Ensure that we have data
    if filtered_tt.empty:
        print("No data found for the given line and direction.")
        return

    # Get the time range from the filtered timetable
    start_time = filtered_tt['ARRIVE_TS'].min()
    end_time = filtered_tt['DEPARTURE_TS'].max()
    print(
        f"Plotting for time range: {start_time} to {end_time} (seconds since midnight)")

    # Prepare data for plotting
    segments = []
    load_values = []

    # Store all unique station IDs (flattened)
    station_ids = []

    for train_id in filtered_tt.index.unique():
        train_data = filtered_tt[filtered_tt.index == train_id]

        for i in range(len(train_data) - 1):
            # Connect the current station's departure with the next station's arrival
            row1 = train_data.iloc[i]
            row2 = train_data.iloc[i + 1]

            # Convert timestamps to numeric values (seconds since midnight)
            x = [row1['DEPARTURE_TS'], row2['ARRIVE_TS']]
            y = [row1['STATION_NID'], row2['STATION_NID']]

            # Add station IDs to the list (flattened)
            station_ids.extend([row1['STATION_NID'], row2['STATION_NID']])

            # Color based on train load (if required)
            if include_train_load:
                # Random load for now (replace with actual data)
                load = np.random.randint(1, 100)
                load_values.append(load)
            else:
                # Default load for now (0 if not showing load)
                load_values.append(0)

            # Store the line segments for plotting
            segments.append([x, y])

    # Remove duplicates and get unique stations
    unique_station_ids = sorted(set(station_ids))

    # Create figure
    fig = go.Figure()

    # Define color scale using matplotlib colormap
    cmap = cm.viridis
    norm = plt.Normalize(vmin=min(load_values), vmax=max(
        load_values))  # Normalize train load values

    # Add lines for each segment
    for i, segment in enumerate(segments):
        x, y = segment
        load = load_values[i]  # Use load for color mapping
        rgba_color = cmap(norm(load))  # Get RGBA color from colormap
        # Convert to RGB string
        rgb_color = f"rgb({int(rgba_color[0] * 255)}, {int(rgba_color[1] * 255)}, {int(rgba_color[2] * 255)})"
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color=rgb_color, width=2),
            showlegend=False
        ))

    # Set axis labels and title
    fig.update_layout(
        title='Train Timetable',
        xaxis_title='Time (HH:MM)',
        yaxis_title='Station NID',
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(start_time, end_time, 3600),  # Every hour
            ticktext=[f"{int(t//3600):02}:{int((t%3600)//60):02}" for t in np.arange(
                start_time, end_time, 3600)]  # Format as HH:MM
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=unique_station_ids,  # Set station labels
            # Use unique station IDs for tick labels
            ticktext=[
                f"Station {int(station)}" for station in unique_station_ids]
        ),
    )

    # Show color bar if train load is included
    if include_train_load:
        fig.update_layout(coloraxis_colorbar=dict(
            title="Train Load",
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["Low", "20", "40", "60", "80", "High"]
        ))

    # Show plot
    fig.show()


if __name__ == "__main__":
    config.load_config()

    print("=" * 100)
    print("Test 1".center(100, " "))
    print("=" * 100)
    # test1()

    print("=" * 100)
    print("Test 2".center(100, " "))
    print("=" * 100)
    test2()

    print("=" * 100)
    print("Test 3".center(100, " "))
    print("=" * 100)
    test3()

    print("=" * 100)
    print("Test 4".center(100, " "))
    print("=" * 100)
    test4()

    print("=" * 100)
    print("Test 5".center(100, " "))
    print("=" * 100)
    test5()

    pass
