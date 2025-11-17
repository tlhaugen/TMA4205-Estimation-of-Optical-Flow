import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from image_gen import mycomputeColor, mycolorwheel
import time

def init_results(methods, Ns):
    """
    Create a results dictionary for storing numerical experiment data.

    methods : list of method names
    Ns      : list of problem sizes
    """
    Ns_array = np.array(Ns, dtype=int)

    results = {}

    for method in methods:
        results[method] = {
            "N": Ns_array.copy(),
            "iterations": np.zeros(len(Ns)),
            "time": np.zeros(len(Ns)),
            "residual_history": [[] for _ in Ns],
        }

    return results



def run_all_methods(methods, Ns, results, 
                    generate_test_image, image_preprocess,
                    solvers_dict,lam = 1.0, testcase=2, tol=1e-8, maxit=2000):
    """
    Run all solvers on different grid sizes and fill results dict.

    methods: list of method names (matching keys in results)
    Ns: list of problem sizes
    results:dictionary from init_results()
    tol, maxit: solver parameters
    generate_test_image: function(N, testcase) → (Im_0, Im_1)
    image_preprocess: preprocess function
    solvers_dict: {"cg": of_cg, "vc": of_vc, "pcg": run_pcg}
    testcase: which synthetic test to run
    """

    for k, N in enumerate(Ns):

        # Regularisation parameter λ = 4^(log2(N) - 4)
        lam = 4 ** (int(np.log2(N)) - 4)

        #Generate and preprocess images 
        Im_0, Im_1 = generate_test_image(N, testcase=testcase)
        u0, v0, Ix, Iy, rhsu, rhsv, I0, I1 = image_preprocess(Im_0, Im_1)

    
        for method in methods:
            solver = solvers_dict[method]

            start = time.time()
            u, v, it, rel_res, res_hist = solver(
                u0, v0, Ix, Iy, lam, rhsu, rhsv, tol=tol, maxit=maxit
            )
            elapsed = time.time() - start

            #Store results
            results[method]["iterations"][k] = it
            results[method]["time"][k] = elapsed
            results[method]["residual_history"][k] = res_hist

    return results



def plot_flow_field(I0, I1, u, v, method='cg'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax0, ax1, ax2 = axes  # unpack for clarity

    flow_mag = np.sqrt(u**2 + v**2)

    # 1) Original grayscale image
    ax0.imshow(I0, cmap="gray")
    ax0.set_title(f"First input image ({u.shape[0]}x{u.shape[1]})")
    ax0.axis("off")

    # 2) Second grayscale image
    ax1.imshow(I1, cmap="gray")
    ax1.set_title(f"Second input image ({u.shape[0]}x{u.shape[1]})")
    ax1.axis("off")

    # 3) Flow magnitude
    im = ax2.imshow(flow_mag, cmap="inferno")
    ax2.set_title("Flow magnitude")
    ax2.axis("off")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="|u,v|")

    plt.suptitle(f"Optical Flow Field using {method} method")
    plt.tight_layout()
    plt.show()


def plot_quiver(I0, u1, v1, u2, v2, u3, v3, step=20):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax0, ax1, ax2 = axes  # unpack for clarity
    
    # grid for quiver
    Y, X = np.mgrid[0:u1.shape[0], 0:u1.shape[1]]
    mask = (X % step == 0) & (Y % step == 0)

    # 1) Quiver overlay on the image
    ax0.imshow(I0, cmap="gray")
    ax0.quiver(
        X[mask], Y[mask],
        u1[mask], v1[mask],
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
    )
    ax0.set_title("Sparse flow vectors (CG)")
    ax0.axis("off")

    Y, X = np.mgrid[0:u2.shape[0], 0:u2.shape[1]]
    mask = (X % step == 0) & (Y % step == 0)

    # 2) Quiver overlay on the image
    ax1.imshow(I0, cmap="gray")
    ax1.quiver(
        X[mask], Y[mask],
        u2[mask], v2[mask],
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
    )
    ax1.set_title("Sparse flow vectors (VC)")
    ax1.axis("off")

    Y, X = np.mgrid[0:u3.shape[0], 0:u3.shape[1]]
    mask = (X % step == 0) & (Y % step == 0)

    # 2) Quiver overlay on the image
    ax2.imshow(I0, cmap="gray")
    ax2.quiver(
        X[mask], Y[mask],
        u3[mask], v3[mask],
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
    )
    ax2.set_title("Sparse flow vectors (PCG)")
    ax2.axis("off")

    plt.suptitle(f"Optical Flow Field Comparison")
    plt.tight_layout()
    plt.show()

def plot_performance(results, method="cg"):
    data = results[method]
    N = data["N"]
    iters = data["iterations"]
    times = data["time"]
    histories = data["residual_history"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax0, ax1, ax2 = axes

    # 1) Convergence histories
    for Ni, res in zip(N, histories):
        if len(res) == 0:
            continue
        ax0.semilogy(res, label=f"N={int(Ni)}")

    ax0.set_title(f"Convergence history ({method})")
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Relative residual")
    ax0.legend()
    ax0.grid(True, which="both", ls="--", lw=0.5)

    # 2) Iterations vs image size
    ax1.plot(N, iters, "o-", lw=2)
    ax1.set_title(f"Iterations vs image size ({method})")
    ax1.set_xlabel("Image size N")
    ax1.set_ylabel("Number of iterations")
    ax1.grid(True)

    # 3) Computation time vs image size
    ax2.plot(N, times, "o-", lw=2)
    ax2.set_title(f"Computation time vs image size ({method})")
    ax2.set_xlabel("Image size N")
    ax2.set_ylabel("Time [s]")
    ax2.grid(True)

    plt.suptitle(f"Performance metrics for {method} method")
    plt.tight_layout()
    plt.show()

def plot_performance_lambda(results, method="cg"):
    data = results[method]
    lams = data["lam"]
    iters = data["iterations"]
    times = data["time"]
    histories = data["residual_history"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax0, ax1, ax2 = axes

    # 1) Convergence histories for different lambda values
    for lam, res in zip(lams, histories):
        if len(res) == 0:
            continue
        ax0.semilogy(res, label=f"λ={lam:.0f}")

    ax0.set_title(f"Convergence history ({method})")
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Relative residual")
    ax0.legend()
    ax0.grid(True, which="both", ls="--", lw=0.5)

    # 2) Iterations vs lambda
    ax1.plot(lams, iters, "o-", lw=2)
    ax1.set_title(f"Iterations vs λ ({method})")
    ax1.set_xlabel("λ")
    ax1.set_ylabel("Number of iterations")
    ax1.grid(True)

    # 3) Computation time vs lambda
    ax2.plot(lams, times, "o-", lw=2)
    ax2.set_title(f"Computation time vs λ ({method})")
    ax2.set_xlabel("λ")
    ax2.set_ylabel("Time [s]")
    ax2.grid(True)

    plt.suptitle(f"Performance metrics for {method} method")
    plt.tight_layout()
    plt.show()



def run_vc_param_grid(
    N,
    s1_list,
    s2_list,
    level_list,
    lambda_list,
    generate_test_image,
    image_preprocess,
    of_vc,
    testcase=2,
    tol=1e-8,
    maxit=2000,
):
    
    '''
    run V_cycles for a grid of parameters and store results
    returns flat list of records, each a dict with keys:
    N, s1,s1, levels, lambda
    iterations, time, residual_history
    '''

    # Generate and preprocess images
    Im_0, Im_1 = generate_test_image(N, testcase)
    u0, v0, Ix, Iy, rhsu, rhsv, I0, I1 = image_preprocess(Im_0, Im_1)

    records = []
    for lam in lambda_list:
        for levels in level_list:
            for s1 in s1_list:
                for s2 in s2_list:
                    start = time.time()
                    u, v, it, rel_res, res_hist = of_vc(
                        u0,
                        v0,
                        Ix,
                        Iy,
                        lam,
                        rhsu,
                        rhsv,
                        s1=s1,
                        s2=s2,
                        max_level=levels,
                        tol=tol,
                        maxit=maxit,
                    )
                    elapsed = time.time() - start

                    record = {
                        "N": N,
                        "s1": s1,
                        "s2": s2,
                        "levels": levels,
                        "lambda": lam,
                        "iterations": it,
                        "time": elapsed,
                        "residual_history": res_hist,
                    }
                    records.append(record)
    return records

def filter_records(records, **conds):
    """
    Filter a list of record dicts by matching key=value pairs.
    Example: filter_records(records, levels=4, lambda=1.0)
    """
    out = []
    for r in records:
        ok = True
        for key, val in conds.items():
            if r.get(key) != val:
                ok = False
                break
        if ok:
            out.append(r)
    return out

def plot_vc_param_comparison(
    records,
    x_param,
    y_param = 'iterations',
    group_by = None,
    filters=None,
    title = "V-cycle Parameter Comparison",
):
    """
    Comparison potter for V_cycle parameter numerical experiments
    records: list of record dicts from run_vc_param_grid
    x_param: parameter name for x-axis
    y_param: parameter name for y-axis (default: iterations)
    group_by: parameter name to create separate lines
    filters: dict of key=value pairs to filter records
    title: plot title
"""
    if filters is not None:
        records_f = filter_records(records, **filters)
    else: 
        records_f = list(records)

    if group_by is None:
        groups = {None : records_f}   

    else: 
        groups = {}
        for r in records_f:
            key = r[group_by]
            if key not in groups:
                groups[key] = []
            groups[key].append(r)


    plt.figure(figsize=(8, 6))
    for label, group_records in groups.items():
        group_sorted = sorted(group_records, key=lambda r: r[x_param])
        x_vals = [r[x_param] for r in group_sorted]
        y_vals = [r[y_param] for r in group_sorted]
        if group_by is None:
            plt.plot(x_vals, y_vals, marker='o', label='Data')
        else:
            plt.plot(x_vals, y_vals, marker='o', label=f"{group_by}={label}")
    plt.title(title)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.legend()
    plt.grid(True)
    plt.show()

  

def summarize_results(results):
    """
    Build a table from the 'results' dict with:
      N, method, iterations, time [s], iterations/time, mean conv. factor, final rel. residual.
    """
    rows = []

    for method, data in results.items():
        Ns = data["N"]
        its = data["iterations"]
        times = data["time"]
        res_hists = data["residual_history"]

        for N, it, t, hist in zip(Ns, its, times, res_hists):
            it = int(it)
            hist = np.asarray(hist, dtype=float)

            if hist.size > 0:
                final_rel = float(hist[-1])  # last entry in residual history
            else:
                final_rel = np.nan

            #mean convergence factor: rho ≈ (||r_k||/||r_0||)^(1/k)
            if it > 0 and final_rel > 0:
                rho = final_rel ** (1.0 / it)
            else:
                rho = np.nan

            rows.append(
                {
                    "N": int(N),
                    "method": method,
                    "iterations": it,
                    "time_s": t,
                    "it_per_s": it / t if t > 0 else np.nan,
                    "mean_conv_factor": rho,
                    "final_rel_res": final_rel,
                }
            )

    df = pd.DataFrame(rows).sort_values(["N", "method"])

    #print(df.to_markdown(index=False, floatfmt=".3g"))
    return df




def plot_gaussian_gradients(Im_0, Im_1, Ix, Iy, title_prefix="Gaussian level"):
    """
    Visualise Ix, Iy, It for the Gaussian / pyramid level images.

    Im_0, Im_1 : preprocessed (e.g. Gaussian-smoothed, downscaled) frames
    Ix, Iy     : corresponding spatial derivatives (from image_preprocess)
    """

    Im_0 = Im_0.astype(float)
    Im_1 = Im_1.astype(float)

    # Temporal derivative at this (Gaussian) level
    It = Im_1 - Im_0

    plt.figure(figsize=(15, 4))

    # Ix
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(Ix, cmap="viridis")
    plt.title(f"{title_prefix}: Ix = ∂I/∂x")
    plt.colorbar(im1)

    # Iy
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(Iy, cmap="viridis")
    plt.title(f"{title_prefix}: Iy = ∂I/∂y")
    plt.colorbar(im2)

    # It
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(It, cmap="viridis")
    plt.title(f"{title_prefix}: It = ∂I/∂t")
    plt.colorbar(im3)

    plt.tight_layout()
    plt.show()


def init_lambda_results(methods, lams):
    """
    Create a results dict indexed by method, with 'lam', 'iterations',
    'time', and 'residual_history', matching plot_performance_lambda.
    """
    lams_arr = np.array(lams, dtype=float)
    results = {}
    for method in methods:
        results[method] = {
            "lam": lams_arr.copy(),
            "iterations": np.zeros(len(lams_arr), dtype=int),
            "time": np.zeros(len(lams_arr), dtype=float),
            "residual_history": [[] for _ in lams_arr],
        }
    return results


def run_lambda_experiment(
    lams,
    frame0_path,
    frame1_path,
    N_synth,
    generate_test_image,
    image_preprocess,
    of_cg,
    of_vc,
    V_cycle,
    tol=1e-8,
    maxit_cg=3000,
    maxit_vc=10,
    s1_vc=5,
    s2_vc=5,
    max_level_vc=2,
    s1_pcg=4,
    s2_pcg=4,
    max_level_pcg=2,
    gauss_scale=0.5,
):
    """
    Run CG, VC, and PCG for several lambda values.

    - CG and PCG are run on the real image pair (frame0_path, frame1_path).
    - VC is run on a synthetic Gaussian test image of size N_synth.

    Returns:
        results : dict in the format expected by plot_performance_lambda
        vis_data: (Im0_real, Im1_real, u_cg, v_cg, u_vc, v_vc, u_pcg, v_pcg)
                  using the last lambda in `lams` (for plotting flow fields).
    """

    methods = ["cg", "vc", "pcg"]
    results = init_lambda_results(methods, lams)

    # --- Real image pair: preprocess once ---
    Im_0_raw = plt.imread(frame0_path)
    Im_1_raw = plt.imread(frame1_path)
    u0_real, v0_real, Ix_real, Iy_real, rhsu_real, rhsv_real, Im0_real, Im1_real = \
        image_preprocess(Im_0_raw, Im_1_raw, gauss_scale)

    # # --- Synthetic Gaussian test image: preprocess once ---
    # Im_0_synth, Im_1_synth = generate_test_image(N_synth, testcase=2)
    # u0_synth, v0_synth, Ix_synth, Iy_synth, rhsu_synth, rhsv_synth, Im0_s, Im1_s = \
    #     image_preprocess(Im_0_synth, Im_1_synth, gauss_scale)

    # placeholders for last flows (for plotting)
    u_cg = v_cg = u_vc = v_vc = u_pcg = v_pcg = None

    for k, lam in enumerate(lams):
        print(f"\n=== λ = {lam} ===")

        # CG
        start = time.time()
        u_cg, v_cg, it_cg, rel_cg, res_cg = of_cg(
            u0_real, v0_real, Ix_real, Iy_real, lam,
            rhsu_real, rhsv_real,
            tol=tol, maxit=maxit_cg
        )
        t_cg = time.time() - start
        results["cg"]["iterations"][k] = it_cg
        results["cg"]["time"][k] = t_cg
        results["cg"]["residual_history"][k] = res_cg
        print(f"CG:  it={it_cg:4d}, time={t_cg:.3f}s, final rel={rel_cg:.2e}")

        # Vcycle
        start = time.time()
        u_vc, v_vc, it_vc, rel_vc, res_vc = of_vc(
            u0_real, v0_real, Ix_real, Iy_real, lam,
            rhsu_real, rhsv_real,
            s1=s1_vc, s2=s2_vc, max_level=max_level_vc,
            tol=tol, maxit=maxit_vc
        )
        t_vc = time.time() - start
        results["vc"]["iterations"][k] = it_vc
        results["vc"]["time"][k] = t_vc
        results["vc"]["residual_history"][k] = res_vc
        print(f"VC:  it={it_vc:4d}, time={t_vc:.3f}s, final rel={rel_vc:.2e}")

        # PCG
        start = time.time()
        u_pcg, v_pcg, it_pcg, rel_pcg, res_pcg = of_cg(
            u0_real, v0_real, Ix_real, Iy_real, lam,
            rhsu_real, rhsv_real,
            tol=tol, maxit=maxit_cg,
            preconditioner=V_cycle,
            s1=s1_pcg, s2=s2_pcg, level=0, max_level=max_level_pcg
        )
        t_pcg = time.time() - start
        results["pcg"]["iterations"][k] = it_pcg
        results["pcg"]["time"][k] = t_pcg
        results["pcg"]["residual_history"][k] = res_pcg
        print(f"PCG: it={it_pcg:4d}, time={t_pcg:.3f}s, final rel={rel_pcg:.2e}")

    vis_data = (Im0_real, Im1_real, u_cg, v_cg, u_vc, v_vc, u_pcg, v_pcg)
    return results, vis_data

