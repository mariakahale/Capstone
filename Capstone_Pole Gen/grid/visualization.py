import matplotlib.pyplot as plt


def plot_pole_network(poles_df, spans_df, feeder_name):
    """
    Plot the synthetic overhead pole/span network.
    """

    # Fast lookup:
    # pole_id -> x, y
    pole_lookup = poles_df.set_index("pole_id")[["x", "y"]]

    fig, ax = plt.subplots(figsize=(10, 10))

    # --------------------------------
    # Draw every span
    # --------------------------------

    for _, span in spans_df.iterrows():

        start_pole = span["start_pole"]
        end_pole = span["end_pole"]

        x1 = pole_lookup.loc[start_pole, "x"]
        y1 = pole_lookup.loc[start_pole, "y"]

        x2 = pole_lookup.loc[end_pole, "x"]
        y2 = pole_lookup.loc[end_pole, "y"]

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=0.8,
            alpha=0.7
        )

    # --------------------------------
    # Separate the two kinds of poles
    # --------------------------------

    bus_poles = poles_df[
        poles_df["pole_type"] == "bus"
    ]

    intermediate_poles = poles_df[
        poles_df["pole_type"] == "intermediate"
    ]

    # Existing OpenDSS bus locations
    ax.scatter(
        bus_poles["x"],
        bus_poles["y"],
        s=14,
        label="Bus poles",
        zorder=3
    )

    # Synthetic intermediate poles
    ax.scatter(
        intermediate_poles["x"],
        intermediate_poles["y"],
        s=7,
        label="Intermediate poles",
        zorder=2
    )

    ax.set_title(
        f"Feeder {feeder_name} - Synthetic Overhead Pole Network"
    )

    ax.set_xlabel("Longitude / x")
    ax.set_ylabel("Latitude / y")

    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()