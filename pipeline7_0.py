import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
#  PIPELINE 7.0 — Full Descent Engine
#
#  - Pre-throat descent
#  - Post-throat multi-mode continuation:
#       A: Free continuation (OS allowed)
#       B: Soft bounce (resists OS, but can cross)
#       C: Tunneling (flux-driven kicks, OS flicker)
#  - Extended region classifier with OTHER SIDE (OS)
#  - Depth + toy curvature invariant
#  - 3D manifold maps:
#       (step, radius, depth)
#       (step, radius, curvature)
#  - Region-time statistics per mode
#  - Summary report
# ============================================================

OUTPUT_DIR = "pipeline7_0_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
#  Shared parameters
# ============================================================
steps_pre = 20000        # steps before throat
steps_post = 30000       # long post-throat exploration
r0 = 10.0                # initial radius

alpha = 0.00035          # inward drift strength
beta = 0.00012           # reserved for future coupling

flux_freq = 0.0009
flux_amp = 1.0
flux_noise = 0.15

throat_radius = 0.1
forbidden_radius = 0.03  # classical forbidden band

# ============================================================
#  Region classifier (extended)
#
#  I   : r > 1.0
#  II  : 0.1 < r <= 1.0
#  III : 0.03 < r <= 0.1
#  IV  : 0 <= r <= 0.03
#  OS  : r < 0  (OTHER SIDE)
# ============================================================
def classify_region_extended(r: float) -> str:
    if r < 0.0:
        return "OS"
    if r > 1.0:
        return "I"
    if r > 0.1:
        return "II"
    if r > 0.03:
        return "III"
    return "IV"


# ============================================================
#  Entropy flux function
# ============================================================
def entropy_flux(step: int) -> float:
    return flux_amp * np.sin(flux_freq * step) + flux_noise * np.random.randn()


# ============================================================
#  Toy curvature invariant
#
#  Not a real Kerr invariant, but a diagnostic:
#      K ~ depth * (1 + r^2)
# ============================================================
def curvature_invariant(r: float, depth: float) -> float:
    return depth * (1.0 + r**2)


# ============================================================
#  Pre-throat descent
# ============================================================
def run_pre_throat():
    r_vals = []
    depth_vals = []
    flux_vals = []
    region_log = []

    prev_region = "I"

    for step in range(steps_pre):
        # Exponential descent with mild oscillation
        r = r0 * np.exp(-alpha * step) + 0.15 * np.sin(0.0005 * step)
        if r < forbidden_radius:
            # stop when we hit the classical forbidden band
            break

        depth = (1.0 / (r + 0.05))**2
        flux = entropy_flux(step)

        region = classify_region_extended(r)
        if region != prev_region:
            region_log.append(f"[PRE] Step {step}: Region {prev_region} -> Region {region}")
            prev_region = region

        r_vals.append(r)
        depth_vals.append(depth)
        flux_vals.append(flux)

        if region == "III":
            # hand off to post-throat once we’re near the throat
            break

    return np.array(r_vals), np.array(depth_vals), np.array(flux_vals), region_log


# ============================================================
#  Post-throat continuation rules
#
#  A: free continuation, OS allowed
#  B: softened bounce — resists deep IV/OS but can cross
#  C: tunneling — flux-driven kicks, OS flicker
# ============================================================
def continue_post_throat(mode: str,
                         r_start: float,
                         depth_start: float,
                         flux_start: float,
                         step_offset: int):
    r_vals = [r_start]
    depth_vals = [depth_start]
    flux_vals = [flux_start]
    curv_vals = [curvature_invariant(r_start, depth_start)]
    region_log = []

    prev_region = classify_region_extended(r_start)
    r = r_start

    for i in range(steps_post):
        step = step_offset + i

        # Baseline drift: inward pull + mild oscillation
        r = r + (-alpha * r + 0.02 * np.sin(0.0007 * step))

        # Mode B: softened bounce — push outward if deep in IV and not yet OS
        if mode == "B" and 0.0 <= r < throat_radius:
            r = r + 0.015

        # Entropy flux
        flux = entropy_flux(step)

        # Mode C: tunneling — strong flux can kick across the throat and into OS
        if mode == "C" and abs(flux) > 1.0:
            r = r + 0.06 * np.sign(flux)

        # Depth uses |r| but we keep sign for region classification
        depth = (1.0 / (abs(r) + 0.05))**2
        curv = curvature_invariant(r, depth)

        region = classify_region_extended(r)
        if region != prev_region:
            region_log.append(f"[{mode}] Step {step}: Region {prev_region} -> Region {region}")
            prev_region = region

        r_vals.append(r)
        depth_vals.append(depth)
        flux_vals.append(flux)
        curv_vals.append(curv)

    return (
        np.array(r_vals),
        np.array(depth_vals),
        np.array(flux_vals),
        np.array(curv_vals),
        region_log,
    )


# ============================================================
#  Plotting helpers
# ============================================================
def save_plot(x, y, title, filename, ylabel, color):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color=color)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def save_3d_manifold(steps, radii, depths, modes, filename, zlabel, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"A": "cyan", "B": "green", "C": "magenta"}

    for mode in ["A", "B", "C"]:
        mask = (modes == mode)
        ax.plot(
            steps[mask],
            radii[mask],
            depths[mask],
            color=colors[mode],
            label=f"Path {mode}",
            linewidth=1.0,
        )

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Radius (signed)")
    ax.set_zlabel(zlabel)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


# ============================================================
#  Region-time statistics
# ============================================================
def compute_region_stats(radii, modes):
    regions = ["OS", "IV", "III", "II", "I"]
    stats = {mode: {reg: 0 for reg in regions} for mode in ["A", "B", "C"]}

    for mode in ["A", "B", "C"]:
        mask = (modes == mode)
        r_mode = radii[mask]
        for r in r_mode:
            reg = classify_region_extended(r)
            stats[mode][reg] += 1

    return stats


def write_summary_report(pre_log, region_log_all, stats, summary_path):
    with open(summary_path, "w") as f:
        f.write("PIPELINE 7.0 — FULL DESCENT ENGINE SUMMARY\n")
        f.write("===========================================\n\n")

        f.write("Pre-throat region transitions:\n")
        for line in pre_log:
            f.write(line + "\n")
        f.write("\n")

        f.write("Post-throat region transitions:\n")
        for line in region_log_all:
            f.write(line + "\n")
        f.write("\n")

        f.write("Region-time statistics (counts of steps per region):\n\n")
        for mode in ["A", "B", "C"]:
            f.write(f"Mode {mode}:\n")
            for reg in ["OS", "IV", "III", "II", "I"]:
                f.write(f"  Region {reg}: {stats[mode][reg]}\n")
            f.write("\n")


# ============================================================
#  Main runner
# ============================================================
def main():
    # -------------------------
    # Pre-throat
    # -------------------------
    pre_r, pre_d, pre_f, pre_log = run_pre_throat()
    throat_step = len(pre_r)
    r_throat = pre_r[-1]
    d_throat = pre_d[-1]
    f_throat = pre_f[-1]

    paths = {}
    logs = pre_log.copy()

    # -------------------------
    # Post-throat: modes A, B, C
    # -------------------------
    for mode in ["A", "B", "C"]:
        r, d, f, curv, log = continue_post_throat(
            mode, r_throat, d_throat, f_throat, throat_step
        )
        paths[mode] = (r, d, f, curv)
        logs.extend(log)

    # -------------------------
    # Save region log
    # -------------------------
    region_log_path = os.path.join(OUTPUT_DIR, "region_log_7_0.txt")
    with open(region_log_path, "w") as f:
        for line in logs:
            f.write(line + "\n")

    # -------------------------
    # 1D plots per mode
    # -------------------------
    colors = {"A": "cyan", "B": "green", "C": "magenta"}

    for mode in ["A", "B", "C"]:
        r, d, fl, curv = paths[mode]
        steps_arr = np.arange(len(r))

        save_plot(
            steps_arr,
            r,
            f"Pipeline 7.0 — Radius ({mode})",
            f"radius_{mode}.png",
            "Radius (signed)",
            colors[mode],
        )
        save_plot(
            steps_arr,
            d,
            f"Pipeline 7.0 — Depth ({mode})",
            f"depth_{mode}.png",
            "Depth",
            colors[mode],
        )
        save_plot(
            steps_arr,
            fl,
            f"Pipeline 7.0 — Entropy Flux ({mode})",
            f"entropy_{mode}.png",
            "Entropy Flux",
            colors[mode],
        )
        save_plot(
            steps_arr,
            curv,
            f"Pipeline 7.0 — Curvature ({mode})",
            f"curvature_{mode}.png",
            "Curvature (toy invariant)",
            colors[mode],
        )

    # -------------------------
    # Combined 2D radius map
    # -------------------------
    plt.figure(figsize=(10, 5))
    for mode in ["A", "B", "C"]:
        r, _, _, _ = paths[mode]
        plt.plot(r, label=f"Path {mode}", color=colors[mode])
    plt.title("Pipeline 7.0 — Combined Radius Map")
    plt.xlabel("Step")
    plt.ylabel("Radius (signed)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "radius_combined.png"))
    plt.close()

    # -------------------------
    # Build 3D manifold data
    # -------------------------
    all_steps = []
    all_r = []
    all_d = []
    all_curv = []
    all_modes = []

    for mode in ["A", "B", "C"]:
        r, d, _, curv = paths[mode]
        steps_arr = np.arange(len(r))
        all_steps.append(steps_arr)
        all_r.append(r)
        all_d.append(d)
        all_curv.append(curv)
        all_modes.append(np.array([mode] * len(r), dtype=object))

    all_steps = np.concatenate(all_steps)
    all_r = np.concatenate(all_r)
    all_d = np.concatenate(all_d)
    all_curv = np.concatenate(all_curv)
    all_modes = np.concatenate(all_modes)

    # -------------------------
    # 3D manifold maps
    # -------------------------
    save_3d_manifold(
        all_steps,
        all_r,
        all_d,
        all_modes,
        "manifold_3d_depth_7_0.png",
        "Depth",
        "Pipeline 7.0 — 3D Interior Manifold (Depth)",
    )

    save_3d_manifold(
        all_steps,
        all_r,
        all_curv,
        all_modes,
        "manifold_3d_curvature_7_0.png",
        "Curvature (toy invariant)",
        "Pipeline 7.0 — 3D Interior Manifold (Curvature)",
    )

    # -------------------------
    # Region-time statistics + summary
    # -------------------------
    stats = compute_region_stats(all_r, all_modes)
    summary_path = os.path.join(OUTPUT_DIR, "summary_7_0.txt")
    write_summary_report(pre_log, logs, stats, summary_path)

    # -------------------------
    # Final verdict
    # -------------------------
    print("Pipeline 7.0 complete.")
    print(f"Region transitions logged in {region_log_path}")
    print(f"Summary written to {summary_path}")
    print(f"All plots and 3D manifolds saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
