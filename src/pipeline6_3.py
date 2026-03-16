import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "pipeline6_3_output"

steps_pre = 20000
steps_post = 20000
r0 = 10.0

alpha = 0.00035
beta = 0.00012

flux_freq = 0.0009
flux_amp = 1.0
flux_noise = 0.15

throat_radius = 0.1
forbidden_radius = 0.03

def classify_region(r: float) -> str:
    if r > 1.0:
        return "I"
    if r > 0.1:
        return "II"
    if r > 0.03:
        return "III"
    return "IV"

def entropy_flux(step: int) -> float:
    return flux_amp * np.sin(flux_freq * step) + flux_noise * np.random.randn()

def run_pre_throat():
    r_vals = []
    depth_vals = []
    flux_vals = []
    region_log = []

    prev_region = "I"

    for step in range(steps_pre):
        r = r0 * np.exp(-alpha * step) + 0.15 * np.sin(0.0005 * step)
        if r < forbidden_radius:
            break

        depth = (1.0 / (r + 0.05))**2
        flux = entropy_flux(step)

        region = classify_region(r)
        if region != prev_region:
            region_log.append(f"[PRE] Step {step}: Region {prev_region} -> Region {region}")
            prev_region = region

        r_vals.append(r)
        depth_vals.append(depth)
        flux_vals.append(flux)

        if region == "III":
            break

    return np.array(r_vals), np.array(depth_vals), np.array(flux_vals), region_log

def continue_post_throat(mode: str,
                         r_start: float,
                         depth_start: float,
                         flux_start: float,
                         step_offset: int):
    r_vals = [r_start]
    depth_vals = [depth_start]
    flux_vals = [flux_start]
    region_log = []

    prev_region = "III"
    r = r_start

    for i in range(steps_post):
        step = step_offset + i

        r = r + (-alpha * r + 0.02 * np.sin(0.0007 * step))

        if mode == "B" and r < throat_radius:
            r = r + abs(0.02)

        flux = entropy_flux(step)
        if mode == "C" and abs(flux) > 1.0:
            r = r + 0.05

        depth = (1.0 / (abs(r) + 0.05))**2

        region = classify_region(abs(r))
        if region != prev_region:
            region_log.append(f"[{mode}] Step {step}: Region {prev_region} -> Region {region}")
            prev_region = region

        r_vals.append(r)
        depth_vals.append(depth)
        flux_vals.append(flux)

    return np.array(r_vals), np.array(depth_vals), np.array(flux_vals), region_log

def save_plot(x, y, title, filename, color):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color=color)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(title.split("—")[-1].strip())
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def run_pipeline_6_3():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pre_r, pre_d, pre_f, pre_log = run_pre_throat()
    throat_step = len(pre_r)
    r_throat = pre_r[-1]
    d_throat = pre_d[-1]
    f_throat = pre_f[-1]

    paths = {}
    logs = pre_log.copy()

    for mode in ["A", "B", "C"]:
        r, d, f, log = continue_post_throat(mode, r_throat, d_throat, f_throat, throat_step)
        paths[mode] = (r, d, f)
        logs.extend(log)

    with open(os.path.join(OUTPUT_DIR, "region_log.txt"), "w") as f:
        for line in logs:
            f.write(line + "\n")

    colors = {"A": "cyan", "B": "green", "C": "magenta"}

    for mode in ["A", "B", "C"]:
        r, d, fl = paths[mode]
        steps_arr = np.arange(len(r))

        save_plot(steps_arr, r, f"Pipeline 6.3 — Radius ({mode})", f"radius_{mode}.png", colors[mode])
        save_plot(steps_arr, d, f"Pipeline 6.3 — Depth ({mode})", f"depth_{mode}.png", colors[mode])
        save_plot(steps_arr, fl, f"Pipeline 6.3 — Entropy Flux ({mode})", f"entropy_{mode}.png", colors[mode])

    plt.figure(figsize=(10, 5))
    for mode in ["A", "B", "C"]:
        r, _, _ = paths[mode]
        plt.plot(r, label=f"Path {mode}", color=colors[mode])

    plt.title("Pipeline 6.3 — Combined Manifold Continuation Map")
    plt.xlabel("Step")
    plt.ylabel("Radius")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "manifold_map.png"))
    plt.close()

    print("Pipeline 6.3 complete.")
    print("Region transitions logged in pipeline6_3_output/region_log.txt")
    print("All plots saved in pipeline6_3_output/")
