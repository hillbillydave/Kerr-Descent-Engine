import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUTPUT_DIR = "pipeline7_1_let_math_decide"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────
# Physical parameters — infaller sees normal time
# ────────────────────────────────────────────────
M = 0.05
a = 0.01
delta_tau = 0.15                # constant — no artificial dilation
evap_rate = 1e-6
r0 = 10.0
np.random.seed(42)

from src import compute_metrics, kerr_metric

def classify_region_extended(r: float, throat: float) -> str:
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

# Pre-throat — same as before
def run_pre_throat():
    r = r0
    M_cur = M
    throat = 2.0 * M_cur
    region_log = []
    prev_region = "I"
    r_list, d_list, f_list = [], [], []
    prev_state = np.array([r, np.pi/2, 0.0])
    odim_lines = []

    for step in range(40000):
        dr_dtau = -np.sqrt(2.0 * M_cur / max(abs(r), 1e-8))
        r += dr_dtau * delta_tau + 0.05 * np.sin(0.0003 * step)
        if r < 0.02: break

        depth = (1.0 / (abs(r) + 0.05))**2
        flux = entropy_flux(step, M_cur)
        region = classify_region_extended(r, throat)

        if region != prev_region:
            region_log.append(f"[PRE] Step {step}: {prev_region} to {region}")
            prev_region = region

        state = np.array([r, np.pi/2, step * delta_tau])
        odim = compute_metrics(state, prev_state)

        proxy_entropy = 0.0
        proxy_info = 0.0
        dist_h = abs(r - 2*M_cur)
        if dist_h < 0.4:
            proxy_entropy = 2.5 / (dist_h + 0.003)**2
        proxy_info = proxy_entropy * depth * 0.04

        line = (
            f"PRE {step:6d} | r={r:12.6f} | τ={step*delta_tau:10.4f} | "
            f"real_info={odim.get('information_density', 0):12.4e} | "
            f"proxy_info={proxy_info:12.4e} | "
            f"real_ent={odim.get('observer_entropy', 0):12.4e} | "
            f"proxy_ent={proxy_entropy:12.4e} | curv={curvature_invariant(r, depth, M_cur):12.4e}\n"
        )
        odim_lines.append(line)

        prev_state = state.copy()
        r_list.append(r)
        d_list.append(depth)
        f_list.append(flux)

        if region == "III": break

    return np.array(r_list), np.array(d_list), np.array(f_list), region_log, r, odim_lines

# Post-throat — very smooth inward plunge — let math decide the core
def continue_post_throat(mode: str, r_start: float, step_offset: int):
    r = r_start
    M_cur = M
    r_list, d_list, f_list, curv_list = [], [], [], []
    region_log = []
    odim_lines = []

    prev_region = classify_region_extended(r, 2.0 * M)
    prev_state = np.array([r, np.pi/2, step_offset * delta_tau])

    depth = (1.0 / (abs(r) + 0.05))**2
    flux = entropy_flux(step_offset, M_cur)
    curv = curvature_invariant(r, depth, M_cur)
    r_list.append(r)
    d_list.append(depth)
    f_list.append(flux)
    curv_list.append(curv)

    for i in range(120000):  # long enough to reach deep
        step = step_offset + i
        dr_dtau = -np.sqrt(2.0 * M_cur / max(abs(r), 1e-8))
        M_cur = max(M_cur - evap_rate, 0.001)
        throat = 2.0 * M_cur

        # Almost no oscillation — pure infall + tiny noise
        osc = 0.0003 * np.sin(0.00005 * step) if abs(r) > 0.02 else 0.0
        # Gentle inward bias when deep inside
        extra_in = -0.00008 * r if r < -0.5 else 0.0

        flux = entropy_flux(step, M_cur)
        quantum = 0.0
        # Mode B: very soft bounce only extremely close to r=0
        if mode == "B" and abs(r) < 0.005:
            quantum = 0.02 * (0.005 - abs(r))
        # Mode C: tunneling kicks only when flux is extreme and very close
        elif mode == "C" and abs(r) < 0.01 and abs(flux) > 5.0:
            quantum = 0.10 * np.sign(flux)

        r += dr_dtau * delta_tau + osc + quantum + extra_in
        depth = (1.0 / (abs(r) + 0.05))**2
        curv = curvature_invariant(r, depth, M_cur)
        region = classify_region_extended(r, throat)

        if region != prev_region:
            region_log.append(f"[{mode}] Step {step}: {prev_region} to {region}")
            prev_region = region

        state = np.array([r, np.pi/2, step * delta_tau])
        odim = compute_metrics(state, prev_state)

        # Proxy — massive response only very near r=0
        proxy_entropy = 0.0
        proxy_info = 0.0
        dist_c = abs(r)
        if dist_c < 0.02:
            proxy_entropy = 10.0 / (dist_c + 0.0001)**3.5   # huge near core
        proxy_info = proxy_entropy * depth * 0.015

        line = (
            f"{mode} {step:6d} | r={r:14.6e} | τ={step*delta_tau:10.4f} | "
            f"real_info={odim.get('information_density', 0):12.4e} | "
            f"proxy_info={proxy_info:12.4e} | "
            f"real_ent={odim.get('observer_entropy', 0):12.4e} | "
            f"proxy_ent={proxy_entropy:12.4e} | curv={curv:12.4e}\n"
        )
        odim_lines.append(line)

        prev_state = state.copy()
        r_list.append(r)
        d_list.append(depth)
        f_list.append(flux)
        curv_list.append(curv)

        if abs(r) < 0.0005 or curv > 1e19:
            region_log.append(f"[{mode}] Step {step}: REACHED SINGULARITY / CORE VICINITY (r = {r:.3e}, curv = {curv:.3e})")
            break

    return np.array(r_list), np.array(d_list), np.array(f_list), np.array(curv_list), region_log, odim_lines

# Plotting & summary functions (same as before)

def save_plot(x, y, title, filename, ylabel, color):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color=color)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def save_3d_manifold(steps, radii, values, modes, filename, zlabel, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"A": "cyan", "B": "green", "C": "magenta"}
    for mode in ["A", "B", "C"]:
        mask = (modes == mode)
        ax.plot(steps[mask], radii[mask], values[mask],
                color=colors[mode], label=f"Path {mode}", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Radius (signed)")
    ax.set_zlabel(zlabel)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def compute_region_stats(radii, modes):
    regions = ["OS", "IV", "III", "II", "I"]
    stats = {mode: {reg: 0 for reg in regions} for mode in ["A", "B", "C"]}
    for mode in ["A", "B", "C"]:
        mask = (modes == mode)
        r_mode = radii[mask]
        for r in r_mode:
            reg = classify_region_extended(r, 0.1)
            stats[mode][reg] += 1
    return stats

def write_summary_report(pre_log, post_logs, stats, summary_path):
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PIPELINE 7.1 – LET THE MATH DECIDE – INFALLER RIDES TO THE CORE\n")
        f.write("=====================================================================\n\n")
        f.write("Observer falls with constant proper time — no local dilation felt.\n")
        f.write("Simulation continues smoothly toward r=0 — math decides core structure.\n\n")
        f.write("Pre-throat transitions:\n")
        for line in pre_log:
            f.write(line + "\n")
        f.write("\nPost-throat & core decisions:\n")
        for mode, log in post_logs.items():
            f.write(f"\n{mode}:\n")
            for line in log:
                f.write(line + "\n")
        f.write("\nRegion-time statistics:\n\n")
        for mode in ["A", "B", "C"]:
            f.write(f"Mode {mode}:\n")
            for reg in ["OS", "IV", "III", "II", "I"]:
                f.write(f"  {reg}: {stats[mode][reg]}\n")
            f.write("\n")

# Main
def main():
    pre_r, pre_d, pre_f, pre_log, r_throat, pre_odim = run_pre_throat()
    throat_step = len(pre_r)

    paths = {}
    post_logs = {}
    all_odim = pre_odim.copy()

    for mode in ["A", "B", "C"]:
        r, d, f, curv, log, odim_lines = continue_post_throat(mode, r_throat, throat_step)
        paths[mode] = (r, d, f, curv)
        post_logs[mode] = log
        all_odim.extend(odim_lines)

    with open(os.path.join(OUTPUT_DIR, "odim_evolution.txt"), "w", encoding="utf-8") as f:
        f.write("MODE STEP r τ real_info proxy_info real_ent proxy_ent curv\n")
        for line in all_odim:
            f.write(line)

    with open(os.path.join(OUTPUT_DIR, "region_log_7_1.txt"), "w", encoding="utf-8") as f:
        for line in pre_log: f.write(line + "\n")
        for mode, log in post_logs.items():
            for line in log: f.write(line + "\n")

    colors = {"A": "cyan", "B": "green", "C": "magenta"}
    for mode in ["A", "B", "C"]:
        r, d, fl, curv = paths[mode]
        steps = np.arange(len(r))
        save_plot(steps, r, f"Radius {mode}", f"radius_{mode}.png", "Radius (signed)", colors[mode])
        save_plot(steps, d, f"Depth {mode}", f"depth_{mode}.png", "Depth", colors[mode])
        save_plot(steps, fl, f"Entropy Flux {mode}", f"entropy_{mode}.png", "Entropy Flux", colors[mode])
        save_plot(steps, curv, f"Curvature {mode}", f"curvature_{mode}.png", "Curvature", colors[mode])

    plt.figure(figsize=(10, 5))
    for mode in ["A", "B", "C"]:
        r, _, _, _ = paths[mode]
        plt.plot(r, label=mode, color=colors[mode])
    plt.title("Combined Radius – Let Math Decide Core")
    plt.xlabel("Step")
    plt.ylabel("Radius (signed)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "radius_combined.png"))
    plt.close()

    all_steps, all_r, all_d, all_curv, all_modes = [], [], [], [], []
    for mode in ["A", "B", "C"]:
        r, d, _, curv = paths[mode]
        st = np.arange(len(r))
        all_steps.append(st)
        all_r.append(r)
        all_d.append(d)
        all_curv.append(curv)
        all_modes.append(np.full(len(r), mode))

    all_steps = np.concatenate(all_steps)
    all_r = np.concatenate(all_r)
    all_d = np.concatenate(all_d)
    all_curv = np.concatenate(all_curv)
    all_modes = np.concatenate(all_modes)

    save_3d_manifold(all_steps, all_r, all_d, all_modes, "3d_depth.png", "Depth", "3D Depth – to Core")
    save_3d_manifold(all_steps, all_r, all_curv, all_modes, "3d_curv.png", "Curvature", "3D Curvature – to Core")

    stats = compute_region_stats(all_r, all_modes)
    summary_path = os.path.join(OUTPUT_DIR, "summary_7_1.txt")
    write_summary_report(pre_log, post_logs, stats, summary_path)

    print("Run complete – infaller rides all the way down.")
    print(f"ODIM log:      {os.path.join(OUTPUT_DIR, 'odim_evolution.txt')}")
    print(f"Region log:    {os.path.join(OUTPUT_DIR, 'region_log_7_1.txt')}")
    print(f"Summary:       {summary_path}")
    print(f"Plots in:      {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
