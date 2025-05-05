# to test and implement GPT-generated code
import cProfile
import numpy as np
import time

import pandas as pd

from src import config


def test1():
    from src.utils import read_, read_all
    from src.itinerary import filter_dis_file, cal_in_vehicle_penal_all, compute_itinerary_probabilities
    from src.timetable import find_overload_train_section
    from src.congest_penal import build_penal_mapper_df

    left = read_(config.CONFIG["results"]["left"], show_timer=False)
    assigned = read_all(config.CONFIG["results"]["assigned"], show_timer=False)

    # get walk link distribution attached iti from file and filter with left.
    dis_df_from_file = read_(config.CONFIG["results"]["dis"], show_timer=False)
    dis_attached_iti = filter_dis_file(
        dis_df_from_file=dis_df_from_file, left=left)

    # find overload trains and calculate in_vehicle links penalty and attach to iti -> this takes 10s
    overload_train_section = find_overload_train_section(assigned=assigned)
    penalized_iti = cal_in_vehicle_penal_all(
        penal_mapper_df=build_penal_mapper_df(
            overload_train_section=overload_train_section,
            penal_func_type="x",
            penal_agg_method="min"
        ),
        left=left
    )

    # calculate probability based on walk distribution and in_vehicle penalized iti dataframes -> this takes 10s
    prob = compute_itinerary_probabilities(
        dis_attached_iti=dis_attached_iti, penalized_iti=penalized_iti)

    print(prob)

    # get most-probable itinerary for each rid -> use numpy lexsort to enhance performance
    data = prob.reset_index()[["rid", "iti_id", "prob"]].to_numpy()
    sorted_idx = np.lexsort((-data[:, 2], data[:, 0]))  # -prob, rid
    data_sorted = data[sorted_idx]

    unique_rids, first_idx = np.unique(data_sorted[:, 0], return_index=True)
    data_result = data_sorted[first_idx]
    most_probable_iti_df = pd.DataFrame(
        data_result, columns=["rid", 'iti_id', 'prob'])
    most_probable_iti_df['rid'] = most_probable_iti_df['rid'].astype(int)
    most_probable_iti_df['iti_id'] = most_probable_iti_df['iti_id'].astype(int)

    print(most_probable_iti_df)

    ### ✅ max-prob uniqueness check ###
    is_unique_max = np.ones(len(first_idx), dtype=bool)

    prob_first = data_sorted[first_idx, 2]
    # second max prob, potentially equal to max_prob
    prob_next = data_sorted[first_idx + 1, 2]

    is_unique_max = prob_first != prob_next  # check uniqueness

    non_unique_rids = unique_rids[~is_unique_max]
    print(f"Number of rid with non-unique max prob: {len(non_unique_rids)}")
    print(f"Example rids: {non_unique_rids[:10]}")

    flag_df = pd.DataFrame({
        'rid': unique_rids,
        'is_unique_max': is_unique_max,
        'prob': prob_first
    })
    flag_df = flag_df[~is_unique_max]
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
    test1()

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
