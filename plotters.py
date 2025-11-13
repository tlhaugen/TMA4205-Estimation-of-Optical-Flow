import numpy as np
import matplotlib.pyplot as plt


def plot_flow_field(I0, u, v, method='cg'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax0, ax1, ax2 = axes  # unpack for clarity

    flow_mag = np.sqrt(u**2 + v**2)
    step = 40  # spacing for quiver arrows

    # grid for quiver
    Y, X = np.mgrid[0:u.shape[0], 0:u.shape[1]]
    mask = (X % step == 0) & (Y % step == 0)

    # 1) Original grayscale image
    ax0.imshow(I0, cmap="gray")
    ax0.set_title(f"Input image ({u.shape[0]}x{u.shape[1]})")
    ax0.axis("off")

    # 2) Quiver overlay on the image
    ax1.imshow(I0, cmap="gray")
    ax1.quiver(
        X[mask], Y[mask],
        u[mask], v[mask],
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
    )
    ax1.set_title("Sparse flow vectors")
    ax1.axis("off")

    # 3) Flow magnitude
    im = ax2.imshow(flow_mag, cmap="inferno")
    ax2.set_title("Flow magnitude")
    ax2.axis("off")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="|u,v|")

    plt.suptitle(f"Optical Flow Field using {method} method")
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


# def plot_flow_field(I0, u, v, method = 'cg'):

#     fig, axes = plt.subplots(1, 3, figsize=(15, 15))

#     flow_mag = np.sqrt(u**2 + v**2)
#     step = 40  # spacing for quiver arrows
#     Y, X = np.mgrid[0:u.shape[0], 0:u.shape[1]]
#     mask = (X % step == 0) & (Y % step == 0)

#     # Original grayscale image
#     axes[0].imshow(I0, cmap="gray")
#     axes[0].set_title(f"Input image ({u.shape[0]}x{u.shape[1]})")
#     axes[0].axis("off")

#     #  Quiver overlay
#     axes[0].imshow(I0, cmap="gray")
#     axes[0].quiver(X[mask], Y[mask], u[mask], v[mask], color="red", angles="xy", scale_units="xy", scale=1.0, width=0.004)
#     axes[0].set_title("Sparse flow vectors")
#     axes[0].axis("off")

#     # Flow magnitude
#     im = axes[0].imshow(flow_mag, cmap="inferno")
#     axes[0].set_title("Flow magnitude")
#     axes[0].axis("off")
#     fig.colorbar(im, ax=axes[0,2], fraction=0.046, pad=0.04, label="|u,v|")
#     plt.suptitle(f"Optical Flow Field using {method} method")

# # def plot_performance(results, method="cg"):
#     data = results[method]
#     N = data["N"]
#     iters = data["iterations"]
#     times = data["time"]
#     histories = data["residual_history"]

#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#     # Convergence histories
#     for Ni, res in zip(N, histories):
#         if len(res) == 0:
#             continue
#         axes[0].semilogy(res, label=f"N={int(Ni)}")

#     axes[0].set_title(f"Convergence history ({method})")
#     axes[0].set_xlabel("Iteration")
#     axes[0].set_ylabel("Relative residual")
#     axes[0].legend()
#     axes[0].grid(True, which="both", ls="--", lw=0.5)

#     # Iterations vs image size
#     axes[1].plot(N, iters, "o-", lw=2)
#     axes[1].set_title(f"Iterations vs image size ({method})")
#     axes[1].set_xlabel("Image size N")
#     axes[1].set_ylabel("Number of iterations")
#     axes[1].grid(True)

#     # Computation time vs image size
#     axes[2].plot(N, times, "o-", lw=2)
#     axes[2].set_title(f"Computation time vs image size ({method})")
#     axes[2].set_xlabel("Image size N")
#     axes[2].set_ylabel("Time [s]")
#     axes[2].grid(True)
#     plt.suptitle(f"Performance metrics for {method} method")
#     plt.show()