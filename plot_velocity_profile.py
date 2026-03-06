import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_velocity_triplet(v_true, v_pred, title_prefix="Velocity", dx=10.0, dz=10.0, save_path=None):
    v_true = np.asarray(v_true)
    v_pred = np.asarray(v_pred)
    if v_true.ndim != 2 or v_pred.ndim != 2:
        raise ValueError("v_true and v_pred must be 2D arrays [nz, nx].")

    err = v_pred - v_true
    nz, nx = v_true.shape
    extent = [0, nx * dx, nz * dz, 0]

    vmin = min(v_true.min(), v_pred.min())
    vmax = max(v_true.max(), v_pred.max())
    eabs = float(np.max(np.abs(err)) + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(v_true, cmap="turbo", aspect="auto", origin="upper", extent=extent, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix} - Ground Truth")
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Depth (m)")
    c0 = fig.colorbar(im0, ax=axes[0])
    c0.set_label("Velocity")

    im1 = axes[1].imshow(v_pred, cmap="turbo", aspect="auto", origin="upper", extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title_prefix} - Prediction")
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Depth (m)")
    c1 = fig.colorbar(im1, ax=axes[1])
    c1.set_label("Velocity")

    im2 = axes[2].imshow(err, cmap="seismic", aspect="auto", origin="upper", extent=extent, vmin=-eabs, vmax=eabs)
    axes[2].set_title(f"{title_prefix} - Error (Pred - GT)")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylabel("Depth (m)")
    c2 = fig.colorbar(im2, ax=axes[2])
    c2.set_label("Velocity Error")

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot ground-truth/prediction/error velocity profiles.")
    parser.add_argument("--true", required=True, help="Path to ground-truth .npy (2D) or array with leading dims.")
    parser.add_argument("--pred", required=True, help="Path to prediction .npy (2D) or array with leading dims.")
    parser.add_argument("--dx", type=float, default=10.0, help="Grid spacing in x (meters).")
    parser.add_argument("--dz", type=float, default=10.0, help="Grid spacing in z (meters).")
    parser.add_argument("--title", default="FWI", help="Figure title prefix.")
    parser.add_argument("--save", default="results/velocity_triplet.png", help="Output figure path.")
    parser.add_argument("--denorm", action="store_true", help="Apply de-normalization with --vmin and --vmax.")
    parser.add_argument("--vmin", type=float, default=1500.0, help="Minimum velocity for de-normalization.")
    parser.add_argument("--vmax", type=float, default=4500.0, help="Maximum velocity for de-normalization.")
    args = parser.parse_args()

    v_true = np.load(args.true)
    v_pred = np.load(args.pred)

    if v_true.ndim > 2:
        v_true = np.squeeze(v_true)
    if v_pred.ndim > 2:
        v_pred = np.squeeze(v_pred)

    if args.denorm:
        scale = args.vmax - args.vmin
        v_true = args.vmin + v_true * scale
        v_pred = args.vmin + v_pred * scale

    plot_velocity_triplet(v_true, v_pred, title_prefix=args.title, dx=args.dx, dz=args.dz, save_path=args.save)


if __name__ == "__main__":
    main()
