"""Shared helpers for analysing LicketyFit output (simulation and data)."""

import glob
import os
import pickle
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants
T5_TREE_NAME = "WCTEReadoutWindows"
T5_X_BRANCH = "T5_hit_pos_x"
T5_Y_BRANCH = "T5_hit_pos_y"
WCSIM_TREE_NAME = "wcsimT"
T5_Z_MM = -1348.76  # fixed z of the T5 reference plane

NON_COLUMN_KEYS = {"metadata", "event_failures", "mcs_failures"}

# label -> (fitted key, truth key) for simulation output
PAIRS = {
    "x [mm]": ("x", "truth_active_water_entry_x_mm"),
    "y [mm]": ("y", "truth_active_water_entry_y_mm"),
    "z [mm]": ("z", "truth_active_water_entry_z_mm"),
    "cx": ("cx", "truth_active_water_entry_dir_x"),
    "cy": ("cy", "truth_active_water_entry_dir_y"),
    "cz": ("cz", "truth_active_water_entry_dir_z"),
}

# fitted column -> key inside the per-event "errors" dicts
ERROR_KEYS = {"x": "x0", "y": "y0", "z": "z0", "cx": "cx", "cy": "cy", "cz": "cz"}


# ---------------------------------------------------------------------- I/O
def find_output_file(run_number, outputs_dir="outputs"):
    pattern = os.path.join(outputs_dir, f"estimates_run{run_number}_*.dict")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}'")
    if len(matches) > 1:
        print(f"Warning: multiple files matched, using {matches[0]}: {matches}")
    return matches[0]


def load_output_dict(path):
    print(f"Loading dictionary from: {path}")
    with open(path, "rb") as stream:
        return pickle.load(stream)


def to_dataframe(data):
    """Per-event columns as a DataFrame; the source ROOT path goes in df.attrs."""
    df = pd.DataFrame({k: v for k, v in data.items() if k not in NON_COLUMN_KEYS})
    df.attrs["root_file"] = data.get("metadata", {}).get("config_root_file")
    return df


def dump_dict(data):
    pprint.pprint(data)


def as_float(data, key):
    return np.asarray(data[key], dtype=float)


# ------------------------------------------------------------------ fitting
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_gaussian_near_peak(values, window=None, bins=40, hist_range=None):
    """Fit a Gaussian to histogram bins within `window` of the peak bin.

    With window=None it is 2 robust sigmas (MAD) so it scales with the quantity.
    Returns (amplitude, mean, sigma, window).
    """
    from scipy.optimize import curve_fit

    if window is None:
        mad = np.median(np.abs(values - np.median(values)))
        window = 2.0 * 1.4826 * mad
        if not np.isfinite(window) or window <= 0:
            raise RuntimeError("Degenerate distribution; cannot fit a Gaussian")

    counts, edges = np.histogram(values, bins=bins, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = centers[np.argmax(counts)]

    # Restrict to bins in the window instead of rebinning, to avoid sparse bins.
    in_window = np.abs(centers - peak) <= window
    fit_centers, fit_counts = centers[in_window], counts[in_window]
    if fit_centers.size < 3 or fit_counts.max() <= 0:
        raise RuntimeError("Not enough data near peak to fit a Gaussian")

    p0 = [fit_counts.max(), peak, window / 2.0]
    bounds = ([0, peak - window, 1e-12], [np.inf, peak + window, window * 5])
    popt, _ = curve_fit(gaussian, fit_centers, fit_counts, p0=p0, bounds=bounds, maxfev=10000)
    return (*popt, window)


# ----------------------------------------------------------------- plotting
def save_fig(fig, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path)
    print(f"Saved plot to: {path}")


def _fit_axis(axis, values, label, xlabel, window=None, bins=40, hist_range=None, log=False):
    axis.hist(values, bins=bins, range=hist_range, color="steelblue", alpha=0.8)
    axis.axvline(0.0, color="black", linewidth=1)
    title = f"{label}\nbias={np.mean(values):.3g}, rms={np.std(values):.3g}, N={values.size}"
    try:
        amplitude, mean, sigma, win = fit_gaussian_near_peak(
            values, window=window, bins=bins, hist_range=hist_range
        )
        x_fit = np.linspace(mean - win, mean + win, 200)
        axis.plot(x_fit, gaussian(x_fit, amplitude, mean, sigma), color="crimson", linewidth=2)
        title += f"\nfit: mean={mean:.3g}, sigma={sigma:.3g}"
    except RuntimeError:
        pass
    if log:
        axis.set_yscale("log")
        axis.set_ylim(bottom=0.1)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Events")


def plot_residual_grid(residuals, xlabel="Fitted - truth", ncols=3, window=None,
                       bins=40, hist_range=None, log=False):
    """Histogram + Gaussian fit for each {label: residual array}."""
    nrows = -(-len(residuals) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4 * nrows), squeeze=False)
    for axis, (label, values) in zip(axes.flat, residuals.items()):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        _fit_axis(axis, values, label, xlabel, window, bins, hist_range, log)
    for axis in list(axes.flat)[len(residuals):]:
        axis.set_visible(False)
    fig.tight_layout()
    return fig, axes


def plot_distributions(columns, bins=50, log=True):
    """Plain histograms for each {label: array}."""
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4), squeeze=False)
    for axis, (label, values) in zip(axes[0], columns.items()):
        values = np.asarray(values, dtype=float)
        axis.hist(values[np.isfinite(values)], bins=bins)
        if log:
            axis.set_yscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel("counts")
    fig.tight_layout()
    return fig, axes[0]


def plot_truth_vs_reco_2d(truth, reco, label, truth_name="MC truth", bins=60, zoom_range=None):
    """2D histogram of truth (x axis) vs reco (y axis).

    zoom_range follows hist2d's format [[xmin, xmax], [ymin, ymax]].
    """
    truth = np.asarray(truth, dtype=float)
    reco = np.asarray(reco, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(reco)
    truth, reco = truth[valid], reco[valid]

    if zoom_range is None:
        lo, hi = min(truth.min(), reco.min()), max(truth.max(), reco.max())
        zoom_range = [[lo, hi], [lo, hi]]
    x_range, y_range = zoom_range

    fig, axis = plt.subplots(figsize=(7, 6))
    hist = axis.hist2d(truth, reco, bins=bins, range=zoom_range, cmap="viridis")
    fig.colorbar(hist[3], ax=axis, label="Events")

    diag = [max(x_range[0], y_range[0]), min(x_range[1], y_range[1])]
    axis.plot(diag, diag, color="crimson", linewidth=1, linestyle="--", label="y = x")
    axis.legend()

    axis.text(
        0.03, 0.97, f"r = {np.corrcoef(truth, reco)[0, 1]:.4f}",
        transform=axis.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    axis.set_xlim(x_range)
    axis.set_ylim(y_range)
    axis.set_xlabel(f"{truth_name} {label}")
    axis.set_ylabel(f"Reconstructed {label}")
    axis.set_title(f"{label}: {truth_name} vs reconstructed (N={truth.size})")
    fig.tight_layout()
    return fig, axis


# --------------------------------------------------------------- simulation
def simulation_residuals(data, pairs=PAIRS) -> dict:
    """{label: fitted - truth} for the active-water-entry truth columns."""
    return {
        label: as_float(data, fit_key) - as_float(data, truth_key)
        for label, (fit_key, truth_key) in pairs.items()
    }


def normalized_simulation_residuals(data, pairs=PAIRS, error_keys=ERROR_KEYS):
    """{label: (fitted - truth) / sigma}; needs data["errors"] (detailed results)."""
    if "errors" not in data:
        return None
    residuals = {}
    for label, (fit_key, truth_key) in pairs.items():
        err_key = error_keys.get(fit_key, fit_key)
        sigma = np.asarray([e.get(err_key, np.nan) for e in data["errors"]], dtype=float)
        residual = (as_float(data, fit_key) - as_float(data, truth_key)) / sigma
        residual[~(sigma > 0)] = np.nan
        residuals[label] = residual
    return residuals


def compare_wcsim_vertices(root_file, data, tree_name=WCSIM_TREE_NAME,
                           vertex_scale_to_mm=1.0, root_entry_indices=None):
    """Compare WCSim fVtxs with the saved active-water truth.

    vertex_scale_to_mm converts fVtxs to mm (10.0 if the ROOT file is in cm).
    root_entry_indices maps output rows to WCSim tree entries; without it both
    must have the same number of rows in the same order.
    """
    import awkward as ak
    import uproot

    if not np.isfinite(vertex_scale_to_mm) or vertex_scale_to_mm <= 0.0:
        raise ValueError("vertex_scale_to_mm must be a positive finite number")

    with uproot.open(root_file) as f:
        tree = f[tree_name]
        branch = tree["wcsimrootevent"]
        names = [
            str(n).split(";")[0]
            for n in branch.keys(recursive=True)
            if str(n).split(";")[0].lower().endswith("fvtxs")
        ]
        if len(names) != 1:
            raise KeyError(f"Expected one fVtxs branch in {tree_name!r}; found {names}")
        raw = ak.to_list(tree[names[0]].array(library="ak"))

    vertices = np.full((len(raw), 3), np.nan)
    for i, value in enumerate(raw):
        # fVtxs may be wrapped by the event-list and fixed-capacity dimensions.
        while isinstance(value, list) and value and isinstance(value[0], list):
            value = value[0]
        if isinstance(value, list) and len(value) >= 3:
            try:
                vertices[i] = np.asarray(value[:3], dtype=float) * vertex_scale_to_mm
            except (TypeError, ValueError):
                continue

    truth_keys = [f"truth_active_water_entry_{a}_mm" for a in "xyz"]
    if all(k in data for k in truth_keys):
        truth = np.column_stack([as_float(data, k) for k in truth_keys])
    elif isinstance(data.get("wcsim_truth"), list):
        truth = np.asarray(
            [[r.get("x_mm", np.nan), r.get("y_mm", np.nan), r.get("z_mm", np.nan)]
             for r in data["wcsim_truth"]],
            dtype=float,
        )
    else:
        raise KeyError("Output has no active-water truth; enable WCSim truth in the fit.")

    if root_entry_indices is None:
        if len(vertices) != len(truth):
            raise ValueError(
                f"ROOT has {len(vertices)} entries but output has {len(truth)} truth "
                "rows; provide root_entry_indices to align them."
            )
        root_entry_indices = np.arange(len(truth))
    else:
        root_entry_indices = np.asarray(root_entry_indices, dtype=int)
        if root_entry_indices.shape != (len(truth),):
            raise ValueError("root_entry_indices must have one index per output row")
        if np.any(root_entry_indices < 0) or np.any(root_entry_indices >= len(vertices)):
            raise IndexError("root_entry_indices contains an invalid ROOT entry")

    vertices = vertices[root_entry_indices]
    result = pd.DataFrame({"root_entry": root_entry_indices})
    for i, axis in enumerate("xyz"):
        result[f"wcsim_vertex_{axis}_mm"] = vertices[:, i]
        result[f"truth_entry_{axis}_mm"] = truth[:, i]
        result[f"truth_minus_wcsim_{axis}_mm"] = truth[:, i] - vertices[:, i]
    return result


# --------------------------------------------------------------------- data
def extract_t5_truth(root_file, event_numbers, tree_name=T5_TREE_NAME):
    """T5 hit (x, y) [mm] for the entries whose event_number is in event_numbers."""
    import awkward as ak
    import uproot

    with uproot.open(root_file) as f:
        tree = f[tree_name]
        all_numbers = tree["event_number"].array(library="np")
        # T5 branches are vector<double> with one entry per event.
        x = ak.to_numpy(ak.firsts(tree[T5_X_BRANCH].array(library="ak")))
        y = ak.to_numpy(ak.firsts(tree[T5_Y_BRANCH].array(library="ak")))
    mask = np.isin(all_numbers, event_numbers)
    return x[mask], y[mask]


def data_residuals(df, t5_x, t5_y, z_mm=T5_Z_MM):
    """{label: T5 reference - reco} for x, y, z."""
    return {
        "x [mm]": t5_x - df["x"].to_numpy(),
        "y [mm]": t5_y - df["y"].to_numpy(),
        "z [mm]": z_mm - df["z"].to_numpy(),
    }


# --------------------------------------------------------------------- main -- dump output format
def main():
    """Dump the output format for inspection."""
    import pprint
    import sys

    if len(sys.argv) < 2:
        print("Usage: python lf_analysis.py <output_dict_file>")
        return

    output_dict_file = sys.argv[1]
    data = load_output_dict(output_dict_file)
    pprint.pprint(data)

if __name__ == "__main__":
    main()
