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

            print(f"{method.upper():>4} → iter={it:4d}, time={elapsed:.3f}s")

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