import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
#  PIPELINE 6.4 — Interior Mapper, Other-Side Probe
#  A: Free continuation (extended)
#  B: Bounce model (softened)
#  C: Tunneling model (enhanced)
#  Adds:
#    - true interior (r can go negative)
#    - OTHER-SIDE region (r < 0)
#    - 3D manifold map: (step, r, depth)
# ============================================================

OUTPUT_DIR = "pipeline6_4_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
#  Shared parameters (building on 6.3)
# ============================================================
steps_pre = 20000       # steps before throat
steps_post = 25000      # longer post-throat exploration
r0 = 10.0

alpha = 0.00035
beta = 0.00012  # reserved

flux_freq = 0.0009
flux_amp = 1.0
flux_noise = 0.15

throat_radius = 0.1
forbidden_radius = 0.03  # classical forbidden band

# ============================================================
#  Region classifier (extended)
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
#  Entropy flux function (same as 6.3)
# ============================================================
def entropy_flux(step: int) -> float:
    return flux_amp * np.sin(flux_freq * step) + flux_noise * np.random.randn()


# ============================================================
#  Pre-throat descent (same baseline as 6.3)
# ============================================================
def run_pre_throat():
    r_vals = []
    depth_vals = []
    flux_vals = []
    region_log = []

    prev_region = "I"

    for step in range(steps_pre):
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
#  Post-throat continuation rules (6.4 extensions)
#
#  A: free continuation, now allowed to cross r=0
#  B: softened bounce — allows brief IV, resists OS
#  C: tunneling — flux-driven kicks that can push to OS
# ============================================================
def continue_post_throat(mode: str,
                         r_start: float,
                         depth_start: float,
                         flux_start: float,
                         step_offset: int):
    r_vals = [r_start]
    depth_vals = [depth_start]
    flux_vals = [flux_start]
    region_log = []

    prev_region = classify_region_extended(r_start)
    r = r_start

    for i in range(steps_post):
        step = step_offset + i

        # Baseline drift: inward pull + mild oscillation
        r = r + (-alpha * r + 0.02 * np.sin(0.0007 * step))

        # Mode B: softened bounce — only push outward if deep in IV and not yet OS
        if mode == "B" and 0.0 <= r < throat_radius:
            r = r + 0.015

        # Entropy flux
        flux = entropy_flux(step)

        # Mode C: tunneling — strong flux can kick across the throat and even to OS
        if mode == "C" and abs(flux) > 1.0:
            r = r + 0.06 * np.sign(flux)

        # Depth uses |r| but we keep sign for region classification
        depth = (1.0 / (abs(r) + 0.05))**2

        region = classify_region_extended(r)
        if region != prev_region:
            region_log.append(f"[{mode}] Step {step}: Region {prev_region} -> Region {region}")
            prev_region = region

        r_vals.append(r)
        depth_vals.append(depth)
        flux_vals.append(flux)

    return np.array(r_vals), np.array(depth_vals), np.array(flux_vals), region_log


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


def save_3d_manifold(steps, radii, depths, modes, filename):
    """
    3D manifold map:
      x = step
      y = radius (signed)
      z = depth
      color by mode (A/B/C)
    """
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

    ax.set_title("Pipeline 6.4 — 3D Interior Manifold Map")
    ax.set_xlabel("Step")
    ax.set_ylabel("Radius (signed)")
    ax.set_zlabel("Depth")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


# ============================================================
#  Main runner
# ============================================================
def main():
    # Pre-throat
    pre_r, pre_d, pre_f, pre_log = run_pre_throat()
    throat_step = len(pre_r)
    r_throat = pre_r[-1]
    d_throat = pre_d[-1]
    f_throat = pre_f[-1]

    paths = {}
    logs = pre_log.copy()

    # Run all three post-throat paths
    for mode in ["A", "B", "C"]:
        r, d, f, log = continue_post_throat(mode, r_throat, d_throat, f_throat, throat_step)
        paths[mode] = (r, d, f)
        logs.extend(log)

    # Save region log
    region_log_path = os.path.join(OUTPUT_DIR, "region_log_6_4.txt")
    with open(region_log_path, "w") as f:
        for line in logs:
            f.write(line + "\n")

    # Save 1D plots per path
    colors = {"A": "cyan", "B": "green", "C": "magenta"}

    for mode in ["A", "B", "C"]:
        r, d, fl = paths[mode]
        steps_arr = np.arange(len(r))

        save_plot(
            steps_arr,
            r,
            f"Pipeline 6.4 — Radius ({mode})",
            f"radius_{mode}.png",
            "Radius (signed)",
            colors[mode],
        )
        save_plot(
            steps_arr,
            d,
            f"Pipeline 6.4 — Depth ({mode})",
            f"depth_{mode}.png",
            "Depth",
            colors[mode],
        )
        save_plot(
            steps_arr,
            fl,
            f"Pipeline 6.4 — Entropy Flux ({mode})",
            f"entropy_{mode}.png",
            "Entropy Flux",
            colors[mode],
        )

    # Combined 2D radius map
    plt.figure(figsize=(10, 5))
    for mode in ["A", "B", "C"]:
        r, _, _ = paths[mode]
        plt.plot(r, label=f"Path {mode}", color=colors[mode])
    plt.title("Pipeline 6.4 — Combined Radius Map")
    plt.xlabel("Step")
    plt.ylabel("Radius (signed)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "radius_combined.png"))
    plt.close()

    # Build 3D manifold data
    all_steps = []
    all_r = []
    all_d = []
    all_modes = []

    for mode in ["A", "B", "C"]:
        r, d, _ = paths[mode]
        steps_arr = np.arange(len(r))
        all_steps.append(steps_arr)
        all_r.append(r)
        all_d.append(d)
        all_modes.append(np.array([mode] * len(r), dtype=object))

    all_steps = np.concatenate(all_steps)
    all_r = np.concatenate(all_r)
    all_d = np.concatenate(all_d)
    all_modes = np.concatenate(all_modes)

    # 3D manifold map
    save_3d_manifold(all_steps, all_r, all_d, all_modes, "manifold_3d_6_4.png")

    # Final verdict
    print("Pipeline 6.4 complete.")
    print(f"Region transitions logged in {region_log_path}")
    print(f"All plots and 3D manifold saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
