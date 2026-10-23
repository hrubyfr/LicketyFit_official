"""Steering macro for LicketyFit output on simulation (MC truth available)."""

import os

import lf_analysis as lf

# Config: edit in place.
DICT_FILE = "outputs/estimates_wcte_muon_680MeV_general_charge_time.dict"
PLOT_DIR = "scripts/plots/sim"
PRINT_DICT = False

# Optional cross-check of saved truth against WCSim fVtxs.
RUN_WCSIM_VERTEX_CHECK = False
WCSIM_ROOT_FILE = "/eos/user/f/fhruby/MC/wcsim.root"
VERTEX_SCALE_TO_MM = 1.0


def main():
    if not os.path.exists(DICT_FILE):
        print(f"File not found: {DICT_FILE}")
        return

    data = lf.load_output_dict(DICT_FILE)
    if PRINT_DICT:
        lf.dump_dict(data)

    residuals = lf.simulation_residuals(data)
    fig, _ = lf.plot_residual_grid(residuals, window=None)
    lf.save_fig(fig, f"{PLOT_DIR}/sim_residuals.png")

    normalized = lf.normalized_simulation_residuals(data)
    if normalized is None:
        print("No 'errors' entry in dictionary; skipping normalized residuals.")
    else:
        fig, _ = lf.plot_residual_grid(
            normalized, xlabel="(Fitted - truth) / uncertainty", window=20.0
        )
        lf.save_fig(fig, f"{PLOT_DIR}/sim_normalized_residuals.png")

    for label, (fit_key, truth_key) in lf.PAIRS.items():
        fig, _ = lf.plot_truth_vs_reco_2d(
            lf.as_float(data, truth_key), lf.as_float(data, fit_key), label
        )
        lf.save_fig(fig, f"{PLOT_DIR}/{fit_key}_true_vs_reco.png")

    if RUN_WCSIM_VERTEX_CHECK:
        comparison = lf.compare_wcsim_vertices(
            WCSIM_ROOT_FILE, data, vertex_scale_to_mm=VERTEX_SCALE_TO_MM
        )
        print(comparison.describe())


if __name__ == "__main__":
    main()
