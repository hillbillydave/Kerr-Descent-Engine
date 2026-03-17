# PIPELINE 7.2 – MEMORY-OPTIMIZED + WHITE HOLE ONLY IF MATH DECIDES
# 10M steps – logs written directly to disk – almost zero RAM usage

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUTPUT_DIR = "pipeline_7_2_memory_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

M = 0.05
a = 0.01
delta_tau = 0.15
evap_rate = 1e-6
r0 = 10.0
np.random.seed(42)

from src import compute_metrics, kerr_metric

def classify_region_extended(r: float, throat: float, white_hole_active: bool = False) -> str:
    if white_hole_active: return "WH"
    if r < 0.0: return "OS"
    if r > 1.0: return "I"
    if r > throat: return "II"
    if r > 0.03: return "III"
    return "IV"

def entropy_flux(step: int, M_cur: float) -> float:
    T_H = 1.0 / (8.0 * np.pi * M_cur)
    return np.sin(0.0009 * step) + 0.15 * np.random.randn() * T_H

def curvature_invariant(r: float, depth: float, M_cur: float):
    metric = kerr_metric(r, np.pi/2, M_cur, a)
    Delta = metric.get("Delta", 1.0)
    r6 = abs(r)**6 + 1e-12
    return depth * (1.0 + r**2) * (48 * M_cur**2 / r6) * (1.0 / (abs(Delta) + 1e-9))

# Pre-throat (unchanged)
def run_pre_throat():
    # ... (same as previous version – short pre-throat)
    # (I kept it identical for brevity – copy from your last working pre-throat if needed)
    # For now using the same logic as 7.1/7.2
    r = r0
    M_cur = M
    throat = 2.0 * M_cur
    region_log = []
    prev_region = "I"
    r_throat = r0
    odim_lines = []  # only keep pre-throat (tiny)

    for step in range(40000):
        dr_dtau = -np.sqrt(2.0 * M_cur / max(abs(r), 1e-8))
        r += dr_dtau * delta_tau + 0.05 * np.sin(0.0003 * step)
        if r < 0.02: break

        depth = (1.0 / (abs(r) + 0.05))**2
        region = classify_region_extended(r, throat)
        if region != prev_region:
            region_log.append(f"[PRE] Step {step}: {prev_region} to {region}")
            prev_region = region
        if region == "III": break

    return region_log, r, odim_lines   # we only need r_throat and pre-log

# Post-throat – MEMORY EFFICIENT + WHITE HOLE ONLY IF MATH DECIDES
def continue_post_throat(mode: str, r_start: float, step_offset: int, odim_file, region_file):
    r = r_start
    M_cur = M
    white_hole_active = False
    downsample_step = 1000          # only keep 1/1000 points for plots (saves RAM)
    plot_steps = []
    plot_r = []
    plot_depth = []
    plot_flux = []
    plot_curv = []

    MAX_STEPS = 10_000_000

    for i in range(MAX_STEPS):
        step = step_offset + i
        dr_dtau = -np.sqrt(2.0 * M_cur / max(abs(r), 1e-8))
        if white_hole_active:
            dr_dtau = -dr_dtau

        M_cur = max(M_cur - evap_rate, 0.001)
        throat = 2.0 * M_cur

        osc = 0.00003 * np.sin(0.00001 * step) if abs(r) > 2.0 else 0.0
        extra_in = -0.00015 * r if r < -8.0 and not white_hole_active else 0.0

        flux = entropy_flux(step, M_cur)
        quantum = 0.0
        if mode == "B" and abs(r) < 0.001:
            quantum = 0.01 * (0.001 - abs(r))
        elif mode == "C" and abs(r) < 0.003 and abs(flux) > 10.0:
            quantum = 0.06 * np.sign(flux)

        r += dr_dtau * delta_tau + osc + quantum + extra_in
        depth = (1.0 / (abs(r) + 0.05))**2
        curv = curvature_invariant(r, depth, M_cur)

        # === WHITE HOLE TRIGGER – ONLY IF MATH SCREAMS FOR IT ===
        proxy_entropy = 0.0
        if abs(r) < 0.005:
            proxy_entropy = 25.0 / (abs(r) + 5e-6)**4.2

        if (not white_hole_active and
            abs(r) < 0.01 and curv > 1e18 and flux > 15.0 and proxy_entropy > 1e6):
            white_hole_active = True
            region_file.write(f"[{mode}] Step {step}: WHITE HOLE EXIT TRIGGERED (r={r:.3e}, curv={curv:.3e}, flux={flux:.2f})\n")
            dr_dtau = -dr_dtau
            flux = -flux * 1.5

        region = "WH" if white_hole_active else classify_region_extended(r, throat, white_hole_active)

        # Write ODIM line directly to disk (no memory buildup)
        odim_file.write(
            f"{mode} {step:8d} | r={r:14.6e} | τ={step*delta_tau:12.4f} | "
            f"real_info=0.0000e+00 | proxy_info=0.0000e+00 | "
            f"real_ent=0.0000e+00 | proxy_ent={proxy_entropy:12.4e} | curv={curv:12.4e}\n"
        )

        # Downsample for plotting (keeps RAM tiny)
        if i % downsample_step == 0 or white_hole_active:
            plot_steps.append(step)
            plot_r.append(r)
            plot_depth.append(depth)
            plot_flux.append(flux)
            plot_curv.append(curv)

        if abs(r) < 1e-6 or curv > 1e23:
            region_file.write(f"[{mode}] Step {step}: SINGULARITY / CORE REACHED (r={r:.3e}, curv={curv:.3e})\n")
            break

        if i % 500000 == 0 and i > 0:
            print(f"[{mode}] Step {step}: r={r:.4e}  curv={curv:.4e}  WH={white_hole_active}")

    return np.array(plot_steps), np.array(plot_r), np.array(plot_depth), np.array(plot_flux), np.array(plot_curv)

# Main – now memory-safe
def main():
    print("Starting Pipeline 7.2 – Memory Optimized (10M steps)")

    pre_log, r_throat, _ = run_pre_throat()
    throat_step = 40000  # approximate

    odim_path = os.path.join(OUTPUT_DIR, "odim_evolution.txt")
    region_path = os.path.join(OUTPUT_DIR, "region_log_7_2.txt")

    with open(odim_path, "w", encoding="utf-8") as odim_f, open(region_path, "w", encoding="utf-8") as region_f:
        for line in pre_log:
            region_f.write(line + "\n")

        paths = {}
        for mode in ["A", "B", "C"]:
            print(f"\nRunning path {mode} (memory-safe)...")
            steps, r, d, f, curv = continue_post_throat(mode, r_throat, throat_step, odim_f, region_f)
            paths[mode] = (steps, r, d, f, curv)

    # === Final plots from downsampled data (tiny memory) ===
    colors = {"A": "cyan", "B": "lime", "C": "magenta"}
    for mode in ["A", "B", "C"]:
        steps, r, d, fl, curv = paths[mode]
        save_plot(steps, r, f"Radius {mode}", f"radius_{mode}.png", "Radius (signed)", colors[mode])
        save_plot(steps, d, f"Depth {mode}", f"depth_{mode}.png", "Depth", colors[mode])
        save_plot(steps, fl, f"Entropy Flux {mode}", f"entropy_{mode}.png", "Entropy Flux", colors[mode])
        save_plot(steps, curv, f"Curvature {mode}", f"curvature_{mode}.png", "Curvature", colors[mode])

    # Combined radius
    plt.figure(figsize=(12, 6))
    for mode in ["A", "B", "C"]:
        steps, r, _, _, _ = paths[mode]
        plt.plot(steps, r, label=mode, color=colors[mode])
    plt.title("Combined Radius – 7.2 Memory-Optimized")
    plt.xlabel("Step")
    plt.ylabel("Radius (signed)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "radius_combined.png"))
    plt.close()

    print("\nPipeline 7.2 finished without MemoryError!")
    print(f"ODIM log: {odim_path}")
    print(f"Region log: {region_path}")
    print(f"Plots in: {OUTPUT_DIR}/")

def save_plot(x, y, title, filename, ylabel, color):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, color=color, lw=1.2)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()