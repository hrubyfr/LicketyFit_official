import argparse
import glob
import os
import pickle
import pprint
import sys

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

TREE_NAME = "WCTEReadoutWindows"
T5_X_BRANCH = "T5_hit_pos_x"
T5_Y_BRANCH = "T5_hit_pos_y"



def load_output_dict(run_number, outputs_dir="outputs"):
    """
    Locate and load the .dict file matching
    estimates_run<run_number>_*p_muon_general_charge_time_relEff-none.dict
    in the given outputs directory.
    """
    pattern = os.path.join(
        outputs_dir,
        f"estimates_run{run_number}_*p_muon_general_charge_time_relEff-none.dict",
    )
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"No file matching pattern '{pattern}' found in '{outputs_dir}'."
        )
    if len(matches) > 1:
        print(f"Warning: multiple files matched, using the first one: {matches}")

    filepath = matches[0]

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return data, filepath


def dump_dict_to_console(data, filepath=None):
    """
    Dump the contents of a dictionary to the console in a readable format.
    """
    if filepath:
        print(f"Contents of: {filepath}")
        print("-" * 60)
    pprint.pprint(data)


def open_file(run_number, outputs_dir="outputs"):
    """
    Load a run's output .dict file into a DataFrame of its per-event columns.
    The source ROOT file path is kept on df.attrs["root_file"].
    """

    filepath = os.path.join(
        outputs_dir,
        f"estimates_run{run_number}_780p_muon_general_charge_time_relEff-none.dict",
    )
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    skip_keys = {"metadata", "event_failures", "mcs_failures"}
    df = pd.DataFrame({k: v for k, v in data.items() if k not in skip_keys})
    df.attrs["root_file"] = data["metadata"]["config_root_file"]
    return df


def collect_event_indices(df):
    """Extract df['event_number'] as a numpy array."""
    return df["event_number"].to_numpy()


def extract_true_pos(root_file, event_indices, tree_name=TREE_NAME):
    """
    Open root_file with uproot and return the T5 hit positions (true_pos_x,
    true_pos_y) for only the entries whose event_number is in event_indices.
    """
    with uproot.open(root_file) as f:
        tree = f[tree_name]
        event_number_all = tree["event_number"].array(library="np")
        # T5_hit_pos_x/y are vector<double> branches with a single entry per event.
        true_x_all = ak.to_numpy(ak.firsts(tree[T5_X_BRANCH].array(library="ak")))
        true_y_all = ak.to_numpy(ak.firsts(tree[T5_Y_BRANCH].array(library="ak")))

    mask = np.isin(event_number_all, event_indices)
    return true_x_all[mask], true_y_all[mask]


def extract_results(df):
    """Return df['x'], df['y'], df['z'], df['cx'], df['cy'], df['cz'] as a numpy array."""
    return df[["x", "y", "z", "cx", "cy", "cz"]].to_numpy()


def plot_vertex_residuals(true_pos_x, true_pos_y, results, bins=50):
    """
    Plot histograms of (true - reconstructed) vertex position for x, y, and z,
    where results is the array returned by extract_results.
    """
    import matplotlib.pyplot as plt

    rec_x = results[:, 0]
    rec_y = results[:, 1]
    rec_z = results[:, 2]
    residual_x = true_pos_x - rec_x
    residual_y = true_pos_y - rec_y
    residual_z = -1348.76 - rec_z
    from scipy.optimize import curve_fit

    def gaussian(x, amplitude, mean, std):
        return amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2)

    def fit_histogram(values):
        values_in_range = values[(values >= -250) & (values <= 250)]
        counts, edges = np.histogram(values_in_range, bins=bins, range=(-250, 250))
        centers = (edges[:-1] + edges[1:]) / 2
        initial_std = max(np.std(values_in_range), np.finfo(float).eps)
        initial_parameters = (np.max(counts), np.median(values_in_range), initial_std)
        parameters, _ = curve_fit(
            gaussian,
            centers,
            counts,
            p0=initial_parameters,
            bounds=([0, -250, np.finfo(float).eps], [np.inf, 250, np.inf]),
            maxfev=10000,
        )
        return centers, counts, parameters

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    centers_x, counts_x, parameters_x = fit_histogram(residual_x)
    axes[0].bar(
        centers_x,
        counts_x,
        width=centers_x[1] - centers_x[0],
        align="center",
    )
    x_fit = np.linspace(-250, 250, 500)
    axes[0].plot(
        x_fit, gaussian(x_fit, *parameters_x), color="red", label="Gaussian fit"
    )
    axes[0].set_xlim(-200, 200)
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=1e-1)
    axes[0].set_xlabel("true x - reconstructed x [mm]")
    axes[0].set_ylabel("counts")
    axes[0].text(
        0.03,
        0.97,
        f"mean = {parameters_x[1]:.2f} mm\nstddev = {parameters_x[2]:.2f} mm",
        transform=axes[0].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    centers_y, counts_y, parameters_y = fit_histogram(residual_y)
    axes[1].bar(
        centers_y,
        counts_y,
        width=centers_y[1] - centers_y[0],
        align="center",
    )
    axes[1].plot(
        x_fit, gaussian(x_fit, *parameters_y), color="red", label="Gaussian fit"
    )
    axes[1].set_xlim(-200, 200)
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=1e-1)
    axes[1].set_xlabel("true y - reconstructed y [mm]")
    axes[1].set_ylabel("counts")
    axes[1].text(
        0.03,
        0.97,
        f"mean = {parameters_y[1]:.2f} mm\nstddev = {parameters_y[2]:.2f} mm",
        transform=axes[1].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    centers_z, counts_z, parameters_z = fit_histogram(residual_z)
    axes[2].bar(
        centers_z,
        counts_z,
        width=centers_z[1] - centers_z[0],
        align="center",
    )
    axes[2].plot(
        x_fit, gaussian(x_fit, *parameters_z), color="red", label="Gaussian fit"
    )
    axes[2].set_xlim(-200, 200)
    axes[2].set_yscale("log")
    axes[2].set_ylim(bottom=1e-1)
    axes[2].set_xlabel("true z - reconstructed z [mm]")
    axes[2].set_ylabel("counts")
    axes[2].text(
        0.03,
        0.97,
        f"mean = {parameters_z[1]:.2f} mm\nstddev = {parameters_z[2]:.2f} mm",
        transform=axes[2].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    fig.tight_layout()
    return fig, axes


def plot_directions(results, bins=50):
    """
    Plot histograms of the reconstructed direction cosines (cx, cy, cz),
    where results is the array returned by extract_results.
    """
    cx = results[:, 3]
    cy = results[:, 4]
    cz = results[:, 5]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(cx, bins=bins)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("cx")
    axes[0].set_ylabel("counts")

    axes[1].hist(cy, bins=bins)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("cy")
    axes[1].set_ylabel("counts")

    axes[2].hist(cz, bins=bins)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("cz")
    axes[2].set_ylabel("counts")

    fig.tight_layout()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description="Load and dump an output .dict file for a given run number."
    )
    parser.add_argument(
        "run_number",
        type=str,
        help="Run number used to identify the estimates_run<run_number>_*.dict file",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Path to the outputs folder (default: 'outputs')",
    )
    args = parser.parse_args()

    # try:
    #     data, filepath = load_output_dict(args.run_number, args.outputs_dir)
    # except FileNotFoundError as e:
    #     print(f"Error: {e}", file=sys.stderr)
    #     sys.exit(1)

    #dump_dict_to_console(data, filepath)
    
    df = open_file(args.run_number)
    event_indices = collect_event_indices(df)

    EOS_ORIGINAL_FILE_PATH = df.attrs["root_file"]

    true_pos_x, true_pos_y = extract_true_pos(
        EOS_ORIGINAL_FILE_PATH, event_indices
    )
    results = extract_results(df)
    fig, axes = plot_vertex_residuals(true_pos_x, true_pos_y, results)
    fig.savefig(f"vertex_residuals_run{args.run_number}.png")

    fig, axes = plot_directions(results)
    fig.savefig(f"directions_run{args.run_number}.png")


if __name__ == "__main__":
    main()
