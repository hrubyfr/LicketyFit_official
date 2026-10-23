"""Steering macro for LicketyFit output on data (T5 hit position as reference)."""

import lf_analysis as lf

# Config: edit in place.
RUN_NUMBER = "1608"
OUTPUTS_DIR = "outputs"
PLOT_DIR = "scripts/plots/data"
RESIDUAL_RANGE = (-250, 250)  # mm


def main():
    path = lf.find_output_file(RUN_NUMBER, OUTPUTS_DIR)
    df = lf.to_dataframe(lf.load_output_dict(path))
    print(f"Number of events: {len(df)}")

    t5_x, t5_y = lf.extract_t5_truth(df.attrs["root_file"], df["event_number"].to_numpy())

    fig, _ = lf.plot_residual_grid(
        lf.data_residuals(df, t5_x, t5_y),
        xlabel="T5 - reconstructed [mm]",
        bins=50,
        hist_range=RESIDUAL_RANGE,
        log=True,
    )
    lf.save_fig(fig, f"{PLOT_DIR}/vertex_residuals_run{RUN_NUMBER}.png")

    fig, _ = lf.plot_distributions({k: df[k] for k in ("cx", "cy", "cz")})
    lf.save_fig(fig, f"{PLOT_DIR}/directions_run{RUN_NUMBER}.png")

    fig, _ = lf.plot_truth_vs_reco_2d(
        t5_x, df["x"], "x [mm]", truth_name="T5", bins=[60, 60],
        zoom_range=[[-200, 200], [-200, 200]],
    )
    lf.save_fig(fig, f"{PLOT_DIR}/x_true_vs_reco_run{RUN_NUMBER}.png")

    fig, _ = lf.plot_truth_vs_reco_2d(
        t5_y, df["y"], "y [mm]", truth_name="T5", bins=[8, 32],
        zoom_range=[[-68.072, 68.075], [-200, 200]],
    )
    lf.save_fig(fig, f"{PLOT_DIR}/y_true_vs_reco_run{RUN_NUMBER}.png")


if __name__ == "__main__":
    main()
