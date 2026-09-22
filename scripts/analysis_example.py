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
    Plot histograms of (true - reconstructed) vertex position for x and y,
    where results is the array returned by extract_results.
    """
    import matplotlib.pyplot as plt

    rec_x = results[:, 0]
    rec_y = results[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(true_pos_x - rec_x, bins=bins)
    axes[0].set_xlabel("true x - reconstructed x [mm]")
    axes[0].set_ylabel("counts")

    axes[1].hist(true_pos_y - rec_y, bins=bins)
    axes[1].set_xlabel("true y - reconstructed y [mm]")
    axes[1].set_ylabel("counts")

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


if __name__ == "__main__":
    main()
