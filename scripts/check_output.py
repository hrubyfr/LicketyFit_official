import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Config: edit these in place of passing command-line arguments.
DICT_FILE = "outputs/estimates_wcte_muon_680MeV_general_charge_time.dict"
RESIDUALS_OUTPUT_PATH = "scripts/plots/sim_residuals.png"
NORMALIZED_RESIDUALS_OUTPUT_PATH = "scripts/plots/sim_normalized_residuals.png"
X_TRUE_VS_RECO_OUTPUT_PATH = "scripts/plots/x_true_vs_reco.png"

PAIRS = {
    "x [mm]": ("x", "truth_active_water_entry_x_mm"),
    "y [mm]": ("y", "truth_active_water_entry_y_mm"),
    "z [mm]": ("z", "truth_active_water_entry_z_mm"),
    "cx": ("cx", "truth_active_water_entry_dir_x"),
    "cy": ("cy", "truth_active_water_entry_dir_y"),
    "cz": ("cz", "truth_active_water_entry_dir_z"),
}

# Maps a fitted column name to the key used inside the per-event "errors" dicts.
ERROR_KEYS = {"x": "x0", "y": "y0", "z": "z0", "cx": "cx", "cy": "cy", "cz": "cz"}


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_gaussian_near_peak(residual, window=150.0, bins=40):
    counts, edges = np.histogram(residual, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_position = centers[np.argmax(counts)]

    # Restrict to bins within the window instead of rebinning, to avoid sparse/empty bins.
    in_window = np.abs(centers - peak_position) <= window
    fit_centers = centers[in_window]
    fit_counts = counts[in_window]
    if fit_centers.size < 3 or fit_counts.max() <= 0:
        raise RuntimeError("Not enough data near peak to fit a Gaussian")

    p0 = [fit_counts.max(), peak_position, window / 2.0]
    bounds = ([0, peak_position - window, 1e-6], [np.inf, peak_position + window, window * 5])
    popt, _ = curve_fit(gaussian, fit_centers, fit_counts, p0=p0, bounds=bounds, maxfev=10000)
    return popt


def load_dictionary(dict_file):
    print(f"Loading dictionary from: {dict_file}")
    with open(dict_file, "rb") as f:
        return pickle.load(f)


def print_dictionary_contents(data):
    print("\n" + "=" * 50)
    print("Dictionary Contents:")
    print("=" * 50 + "\n")
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {value}")
    else:
        print(data)


def _fit_axis(axis, residual, label, xlabel, window=150.0):
    axis.hist(residual, bins=40, color="steelblue", alpha=0.8)
    axis.axvline(0.0, color="black", linewidth=1)

    title = (
        f"{label}\n"
        f"bias={np.mean(residual):.3g}, "
        f"resolution={np.std(residual):.3g}, "
        f"N={residual.size}"
    )
    try:
        amplitude, mean, sigma = fit_gaussian_near_peak(residual, window=window)
        x_fit = np.linspace(mean - window, mean + window, 200)
        axis.plot(x_fit, gaussian(x_fit, amplitude, mean, sigma), color="crimson", linewidth=2)
        title += f"\nfit: mean={mean:.3g}, sigma={sigma:.3g}"
    except RuntimeError:
        pass

    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Events")


def plot_residuals(data, pairs=PAIRS, output_path="sim_residuals.png"):
    """Plot (fitted - truth) residuals for each quantity in `pairs`."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (label, (fit_key, truth_key)) in zip(axes.flat, pairs.items()):
        fitted = np.asarray(data[fit_key], dtype=float)
        truth = np.asarray(data[truth_key], dtype=float)

        valid = np.isfinite(fitted) & np.isfinite(truth)
        residual = fitted[valid] - truth[valid]

        _fit_axis(axis, residual, label, "Fitted - truth")

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved residual plot to: {output_path}")


def plot_truth_vs_reco_2d(
    data, fit_key="x", truth_key="truth_active_water_entry_x_mm",
    label="x [mm]", bins=60, output_path="x_true_vs_reco.png",
):
    """Plot a 2D histogram of truth (x axis) vs reconstructed value (y axis)."""
    fitted = np.asarray(data[fit_key], dtype=float)
    truth = np.asarray(data[truth_key], dtype=float)

    valid = np.isfinite(fitted) & np.isfinite(truth)
    fitted = fitted[valid]
    truth = truth[valid]

    fig, axis = plt.subplots(figsize=(7, 6))
    hist = axis.hist2d(truth, fitted, bins=bins, cmap="viridis")
    fig.colorbar(hist[3], ax=axis, label="Events")

    lims = [min(truth.min(), fitted.min()), max(truth.max(), fitted.max())]
    axis.plot(lims, lims, color="crimson", linewidth=1, linestyle="--", label="y = x")
    axis.legend()

    correlation = np.corrcoef(truth, fitted)[0, 1]
    axis.text(
        0.03, 0.97, f"r = {correlation:.4f}",
        transform=axis.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    axis.set_xlabel(f"MC truth {label}")
    axis.set_ylabel(f"Reconstructed {label}")
    axis.set_title(f"{label}: truth vs reconstructed (N={fitted.size})")

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved truth-vs-reco 2D histogram to: {output_path}")


def plot_normalized_residuals(
    data, pairs=PAIRS, error_keys=ERROR_KEYS, output_path="sim_normalized_residuals.png"
):
    """Plot (fitted - truth) / fitted_uncertainty for each quantity in `pairs`.

    The per-event uncertainty is read from `data["errors"]`, a list of dicts
    only present when SAVE_DETAILED_EVENT_RESULTS was enabled during the fit.
    """
    if "errors" not in data:
        print("No 'errors' entry found in dictionary; skipping normalized residuals.")
        return

    errors_list = data["errors"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (label, (fit_key, truth_key)) in zip(axes.flat, pairs.items()):
        error_key = error_keys.get(fit_key, fit_key)
        fitted = np.asarray(data[fit_key], dtype=float)
        truth = np.asarray(data[truth_key], dtype=float)
        uncertainty = np.asarray(
            [errors.get(error_key, np.nan) for errors in errors_list], dtype=float
        )

        valid = np.isfinite(fitted) & np.isfinite(truth) & np.isfinite(uncertainty) & (uncertainty > 0)
        residual = (fitted[valid] - truth[valid]) / uncertainty[valid]

        _fit_axis(axis, residual, label, "(Fitted - truth) / uncertainty", window=5.0)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved normalized residual plot to: {output_path}")


def main():
    dict_file = Path(DICT_FILE)
    if not dict_file.exists():
        print(f"File not found: {dict_file}")
        return

    data = load_dictionary(dict_file)
    print_dictionary_contents(data)

    print("\n" + "=" * 50)
    print("Plotting out residuals:")
    plot_residuals(data, output_path=RESIDUALS_OUTPUT_PATH)
    plot_normalized_residuals(data, output_path=NORMALIZED_RESIDUALS_OUTPUT_PATH)
    plot_truth_vs_reco_2d(data, output_path=X_TRUE_VS_RECO_OUTPUT_PATH)


if __name__ == "__main__":
    main()
