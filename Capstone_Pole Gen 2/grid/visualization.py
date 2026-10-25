import matplotlib.pyplot as plt


def plot_pole_span_network(
    poles,
    spans,
    title="Generated Overhead Pole Network"
):
    """
    Plot the generated physical pole/span network.

    Anchor poles and intermediate poles are shown separately.
    """

    # ---------------------------------------------------------
    # Coordinate lookup
    # ---------------------------------------------------------

    pole_lookup = (
        poles
        .set_index("pole_id")[["x", "y"]]
        .to_dict("index")
    )

    # ---------------------------------------------------------
    # Draw spans
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 10))

    for span in spans.itertuples(index=False):

        start = pole_lookup[span.start_pole]
        end = pole_lookup[span.end_pole]

        plt.plot(
            [start["x"], end["x"]],
            [start["y"], end["y"]],
            linewidth=0.7
        )

    # ---------------------------------------------------------
    # Separate pole types
    # ---------------------------------------------------------

    anchor_poles = poles[
        poles["pole_type"] == "anchor"
    ]

    intermediate_poles = poles[
        poles["pole_type"] == "intermediate"
    ]

    plt.scatter(
        anchor_poles["x"],
        anchor_poles["y"],
        s=14,
        label="Anchor poles"
    )

    plt.scatter(
        intermediate_poles["x"],
        intermediate_poles["y"],
        s=7,
        label="Intermediate poles"
    )

    # ---------------------------------------------------------
    # Plot formatting
    # ---------------------------------------------------------

    plt.xlabel("Longitude / X")
    plt.ylabel("Latitude / Y")

    plt.title(title)

    plt.legend()

    plt.axis("equal")

    plt.tight_layout()

    plt.show()