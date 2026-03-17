# PIPELINE 7.8 FIXED – MAXIMUM INWARD DRIVE + ESCAPE PREVENTION + STABILITY GUARDS
# Overflows fixed + critical event log + all your requests

import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "pipeline_7_8_max_inward_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

M = 0.05
a = 0.01
delta_tau = 0.15
evap_rate = 1e-6
r0 = 10.0
THRUST_BASE = 0.035
ESCAPE_BOOST = 0.25
INWARD_BIAS = 0.005
np.random.seed(42)

# =============================================================
# SAFE PHYSICS FUNCTIONS (overflow protection added)
# =============================================================
def kerr_metric(r, M_cur, a):
    if abs(r) > 1e8:
        return {"Delta": 1.0}
    Delta = r**2 - 2 * M_cur * r + a**2
    return {"Delta": Delta}

def compute_odim_metrics(r, curv, flux):
    if abs(r) > 1e8:
        return 0.0, 0.0
    info_density = np.exp(-abs(r) / 5.0) * (1.0 + 20.0 * min(curv, 1e10))
    observer_ent = (1.0 / (abs(r) + 0.001)**2.5) * info_density * (1.0 + abs(flux))
    return info_density, observer_ent

def quiet_scalar_time(r):
    if abs(r) > 1e8:
        return 1.0, 1.0
    S_smooth = 1.0 / (1.0 + np.sqrt(abs(r) + 0.01))
    B_S = 1.0 + 0.15 * S_smooth
    return S_smooth, B_S

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
    if abs(r) > 1e8:
        return 0.0
    metric = kerr_metric(r, M_cur, a)
    Delta = metric.get("Delta", 1.0)
    r6 = abs(r)**6 + 1e-12
    return depth * (1.0 + r**2) * (48 * M_cur**2 / r6) * (1.0 / (abs(Delta) + 1e-9))

# Pre-throat (unchanged)
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

# Post-throat – STABLE MAX INWARD
def continue_post_throat(mode, r_start, step_offset, odim_f, region_f, event_f):
    r = r_start
    M_cur = M
    wh_active = False
    reversed_polarity = False
    reverse_timer = 0
    last_r = r_start
    plot_steps, plot_r, plot_d, plot_f, plot_c = [], [], [], [], []

    MAX = 10_000_000
    for i in range(MAX):
        step = step_offset + i
        dr_dtau = -np.sqrt(2.0 * M_cur / max(abs(r), 1e-8))
        if wh_active:
            dr_dtau = -dr_dtau

        M_cur = max(M_cur - evap_rate, 0.001)
        _, B_S = quiet_scalar_time(r)
        M_eff = M_cur * B_S
        dr_dtau = -np.sqrt(2.0 * M_eff / max(abs(r), 1e-8))

        flux = entropy_flux(step, M_cur)
        real_info, proxy_ent = compute_odim_metrics(r, 0.0, flux)
        depth = (1.0 / (abs(r) + 0.05))**2
        curv = curvature_invariant(r, depth, M_cur)

        bubble_active = abs(r) < 8.0
        odim_feedback = 1.0 + 12.0 * proxy_ent
        bubble_boost = -0.003 * r * odim_feedback if bubble_active else 0.0

        velocity = (r - last_r) / delta_tau
        escape_boost = 0.0
        if velocity > 0.0 and r < 0.0:
            escape_boost = -ESCAPE_BOOST * abs(r) * (1.0 + 5 * proxy_ent)
            event_f.write(f"[{mode}] Step {step}: ESCAPE VELOCITY – STRONG INWARD BOOST (vel={velocity:.4f})\n")

        inward_bias = -INWARD_BIAS * abs(r) if r < 0 else 0.0
        if r < 0 and i % 100000 == 0:
            event_f.write(f"[{mode}] Step {step}: INWARD BIAS ACTIVE (r={r:.3e})\n")

        # === RUNAWAY SAFETY CAP (prevents explosion) ===
        if abs(r) > 1e10:
            r = np.sign(r) * 1e10
            event_f.write(f"[{mode}] Step {step}: RUNAWAY DETECTED – CAPPING r to 1e10\n")

        if proxy_ent > 5e6 and abs(r) < 0.0005 and not reversed_polarity:
            reversed_polarity = True
            reverse_timer = 3000
            event_f.write(f"[{mode}] Step {step}: REVERSE PLATES AT CORE (r={r:.3e})\n")

        if reversed_polarity:
            thrust = +THRUST_BASE * 1.5
            reverse_timer -= 1
            if reverse_timer <= 0:
                reversed_polarity = False
        else:
            thrust = THRUST_BASE

        quantum = 0.0
        if mode == "B" and abs(r) < 0.001: quantum = 0.01 * (0.001 - abs(r))
        if mode == "C" and abs(r) < 0.003 and abs(flux) > 10: quantum = 0.06 * np.sign(flux)

        osc = 0.000005 * np.sin(0.000005 * step) if abs(r) > 3 else 0.0

        r += (dr_dtau + thrust + quantum + bubble_boost + escape_boost + inward_bias) * delta_tau + osc

        if (not wh_active and abs(r) < 0.01 and curv > 1e19 and flux > 18.0 and proxy_ent > 5e6):
            wh_active = True
            event_f.write(f"[{mode}] Step {step}: WHITE HOLE EXIT TRIGGERED\n")

        region = "WH" if wh_active else classify_region_extended(r, 2*M_cur, wh_active)

        odim_f.write(f"{mode} {step:8d} | r={r:14.6e} | tau={step*delta_tau:12.4f} | "
                     f"real_info={real_info:12.4e} | proxy_ent={proxy_ent:12.4e} | "
                     f"curv={curv:12.4e} | bubble={bubble_active} | reversed={reversed_polarity}\n")

        if i % 500 == 0 or escape_boost != 0 or reversed_polarity or abs(r) > 1e9:
            plot_steps.append(step)
            plot_r.append(r)
            plot_d.append(depth)
            plot_f.append(flux)
            plot_c.append(curv)

        if abs(r) < 1e-6 or curv > 1e24:
            event_f.write(f"[{mode}] Step {step}: SINGULARITY / CORE REACHED\n")
            break

        last_r = r

        if i % 500000 == 0 and i > 0:
            print(f"[{mode}] Step {step}: r={r:.4e}  curv={curv:.4e}")

    return np.array(plot_steps), np.array(plot_r), np.array(plot_d), np.array(plot_f), np.array(plot_c)

def main():
    print("Pipeline 7.8 FIXED – MAXIMUM INWARD DRIVE + STABILITY GUARDS")
    pre_log, r_throat = run_pre_throat()

    with open(os.path.join(OUTPUT_DIR, "odim_evolution.txt"), "w", encoding="utf-8") as odim_f, \
         open(os.path.join(OUTPUT_DIR, "region_log_7_8.txt"), "w", encoding="utf-8") as region_f, \
         open(os.path.join(OUTPUT_DIR, "critical_events_7_8.txt"), "w", encoding="utf-8") as event_f:

        for line in pre_log: region_f.write(line + "\n")

        for mode in ["A", "B", "C"]:
            print(f"\nRunning {mode} with maximum inward drive (stable)...")
            steps, r, d, f, c = continue_post_throat(mode, r_throat, 40000, odim_f, region_f, event_f)

            plt.figure(figsize=(10,5))
            plt.plot(steps, r, label=mode)
            plt.title(f"Radius {mode} – 7.8 Fixed")
            plt.xlabel("Step")
            plt.ylabel("Radius (signed)")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, f"radius_{mode}.png"))
            plt.close()

    print("\n=== 7.8 FIXED COMPLETE ===")
    print("Check critical_events_7_8.txt for every inward bias, escape boost, and runaway cap!")
    print("Files saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()