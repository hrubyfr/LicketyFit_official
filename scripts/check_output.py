import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python check_output.py <filepath>")
    sys.exit(1)

dict_file = Path(sys.argv[1])

pairs = {
    "x [mm]": ("x", "truth_active_water_entry_x_mm"),
    "y [mm]": ("y", "truth_active_water_entry_y_mm"),
    "z [mm]": ("z", "truth_active_water_entry_z_mm"),
    "cx": ("cx", "truth_active_water_entry_dir_x"),
    "cy": ("cy", "truth_active_water_entry_dir_y"),
    "cz": ("cz", "truth_active_water_entry_dir_z"),
}


if dict_file.exists():
    print(f"Loading dictionary from: {dict_file}")
    
    # Open and load the dictionary
    with open(dict_file, 'rb') as f:
        data = pickle.load(f)
    
    # Dump everything to console
    print("\n" + "="*50)
    print("Dictionary Contents:")
    print("="*50 + "\n")
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {value}")
    else:
        print(data)
    print("\n" + "="*50)
    print("Plotting out residuals:")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (label, (fit_key, truth_key)) in zip(axes.flat, pairs.items()):
        fitted = np.asarray(data[fit_key], dtype=float)
        truth = np.asarray(data[truth_key], dtype=float)

        valid = np.isfinite(fitted) & np.isfinite(truth)
        residual = fitted[valid] - truth[valid]

        axis.hist(residual, bins=40, color="steelblue", alpha=0.8)
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_title(
            f"{label}\n"
            f"bias={np.mean(residual):.3g}, "
            f"resolution={np.std(residual):.3g}, "
            f"N={residual.size}"
        )
        axis.set_xlabel("Fitted - truth")
        axis.set_ylabel("Events")

    fig.tight_layout()
    plt.show()


else:
    print(f"File not found: {dict_file}")
