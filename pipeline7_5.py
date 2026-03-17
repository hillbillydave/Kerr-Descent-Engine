# PIPELINE 7.5 – INWARD THRUSTERS + GRAV BUBBLE
# Thrusters push WITH gravity to drive deep
# Grav bubble activates automatically when deep (protection + extra inward push)

import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "pipeline_7_5_inward_bubble"
os.makedirs(OUTPUT_DIR, exist_ok=True)

M = 0.05
a = 0.01
delta_tau = 0.15
evap_rate = 1e-6
r0 = 10.0
THRUST_INWARD = 0.035          # constant inward push (with gravity)
np.random.seed(42)

from src import compute_metrics, kerr_metric

def classify_region_extended(r: float, throat: float, wh_active=False):
    if wh_active: return "WH"
    if r < 0.0: return "OS"
    if r > 1.0: return "I"
    if r > throat: return "II"
    if r > 0.03: return "III"
    return "IV"

def entropy_flux(step, M_cur):
    T_H = 1.0 / (8.0 * np.pi * M_cur)
    return np.sin(0.0009 * step) + 0.15 * np.random.randn() * T_H

def curvature_invariant(r, depth, M_cur):
    metric = kerr_metric(r, np.pi/2, M_cur, a)
    Delta = metric.get("Delta", 1.0)
    r6 = abs(r)**6 + 1e-12
    return depth * (1.0 + r**2) * (48 * M_cur**2 / r6) * (1.0 / (abs(Delta) + 1e-9))

# Pre-throat
def run_pre_throat():
    r = r0
    M_cur = M
    throat = 2.0 * M_cur
    region_log = []
    prev = "I"
    for step in range(40000):
        dr = -np.sqrt(2.0 * M_cur / max(abs(r), 1e-8))
        r += dr * delta_tau + 0.05 * np.sin(0.0003 * step)
        if r < 0.02: break
        reg = classify_region_extended(r, throat)
        if reg != prev:
            region_log.append(f"[PRE] Step {step}: {prev} to {reg}")
            prev = reg
        if reg == "III": break
    return region_log, r

# Post-throat with inward thrusters + grav bubble
def continue_post_throat(mode, r_start, step_offset, odim_f, region_f):
    r = r_start
    M_cur = M
    wh_active = False
    bubble_active = False
    plot_steps, plot_r, plot_d, plot_f, plot_c = [], [], [], [], []

    MAX = 10_000_000
    for i in range(MAX):
        step = step_offset + i
        dr_dtau = -np.sqrt(2.0 * M_cur / max(abs(r), 1e-8))
        if wh_active:
            dr_dtau = -dr_dtau

        M_cur = max(M_cur - evap_rate, 0.001)

        # === INWARD THRUSTERS (pushing with gravity) ===
        thrust = THRUST_INWARD if not wh_active else -THRUST_INWARD

        osc = 0.00001 * np.sin(0.000005 * step) if abs(r) > 5 else 0.0
        extra_in = -0.0008 * r if r < -5 and not wh_active else 0.0

        flux = entropy_flux(step, M_cur)
        quantum = 0.0
        if mode == "B" and abs(r) < 0.001: quantum = 0.01 * (0.001 - abs(r))
        if mode == "C" and abs(r) < 0.003 and abs(flux) > 10: quantum = 0.06 * np.sign(flux)

        # === GRAV BUBBLE – activates automatically when deep ===
        bubble_active = abs(r) < 5.0
        bubble_boost = -0.002 * r if bubble_active else 0.0   # extra inward help inside bubble
        bubble_shield = 0.1 if bubble_active else 1.0         # reduces felt curvature

        r += (dr_dtau + thrust + quantum + bubble_boost) * delta_tau + osc + extra_in

        depth = (1.0 / (abs(r) + 0.05))**2
        curv = curvature_invariant(r, depth, M_cur) * bubble_shield   # bubble shields curvature

        # White-hole trigger – only if math strongly decides
        proxy = 0.0
        if abs(r) < 0.005:
            proxy = 30.0 / (abs(r) + 1e-6)**4.5

        if (not wh_active and abs(r) < 0.01 and curv > 1e19 and flux > 18.0 and proxy > 5e6):
            wh_active = True
            region_f.write(f"[{mode}] Step {step}: WHITE HOLE EXIT TRIGGERED (r={r:.3e}, curv={curv:.3e})\n")
            flux = -flux * 2.0

        region = "WH" if wh_active else classify_region_extended(r, 2*M_cur, wh_active)

        odim_f.write(f"{mode} {step:8d} | r={r:14.6e} | tau={step*delta_tau:12.4f} | "
                     f"proxy_ent={proxy:12.4e} | curv={curv:12.4e} | bubble={bubble_active}\n")

        if i % 500 == 0 or wh_active or bubble_active:
            plot_steps.append(step)
            plot_r.append(r)
            plot_d.append(depth)
            plot_f.append(flux)
            plot_c.append(curv)

        if abs(r) < 1e-6 or curv > 1e24:
            region_f.write(f"[{mode}] Step {step}: SINGULARITY / CORE REACHED (r={r:.3e})\n")
            break

        if i % 500000 == 0 and i > 0:
            print(f"[{mode}] Step {step}: r={r:.4e}  curv={curv:.4e}  bubble={bubble_active}  WH={wh_active}")

    return np.array(plot_steps), np.array(plot_r), np.array(plot_d), np.array(plot_f), np.array(plot_c)

def main():
    print("Pipeline 7.5 – Inward Thrusters + Automatic Grav Bubble")
    pre_log, r_throat = run_pre_throat()

    with open(os.path.join(OUTPUT_DIR, "odim_evolution.txt"), "w", encoding="utf-8") as odim_f, \
         open(os.path.join(OUTPUT_DIR, "region_log_7_5.txt"), "w", encoding="utf-8") as region_f:

        for line in pre_log: region_f.write(line + "\n")

        for mode in ["A", "B", "C"]:
            print(f"Running {mode} with inward thrusters + grav bubble...")
            steps, r, d, f, c = continue_post_throat(mode, r_throat, 40000, odim_f, region_f)

            plt.figure(figsize=(10,5))
            plt.plot(steps, r, label=mode)
            plt.title(f"Radius {mode} – 7.5 Inward + Bubble")
            plt.xlabel("Step")
            plt.ylabel("Radius (signed)")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, f"radius_{mode}.png"))
            plt.close()

    print("\nFinished! Check region_log_7_5.txt for WHITE HOLE or SINGULARITY messages.")
    print("All files in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()