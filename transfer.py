from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from cnn_results import cnn_image_parameters, connectivity_from_CNN, load_model_for_eval


def main():
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "figure.figsize": (3.5, 3.5),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.constrained_layout.use": True,
        },
    )

    model_path = Path("models/best.ckpt")
    data_path = Path("data/")

    model = load_model_for_eval(model_path)
    assert model is not None

    data_files = list(data_path.glob("*.hdf5"))

    with sns.axes_style("ticks"):
        fig, axs = plt.subplots(2, len(data_files), figsize=(7.5, 3.5))

    for data_file in data_files:
        with h5py.File(data_file, "r") as f:
            task_img_dataset = f["/test/task_img"]
            task_config_dataset = f["/test/task_config"]
            assert isinstance(task_img_dataset, h5py.Dataset)
            assert isinstance(task_config_dataset, h5py.Dataset)

            idx = 0
            task_img = task_img_dataset[idx]
            comm_img = model.inference(task_img)
            task_config = task_config_dataset[idx]

            img_size = comm_img.shape[0]
            width = img_size * 1.25
            scale = img_size // 256
            extent = [-width / 2, width / 2, -width / 2, width / 2]

            # task image
            task_ax = axs[0, scale - 1]
            task_ax.set_title(f"{width:.0f}m wide area")
            task_ax.imshow(task_img.T, extent=extent, cmap="gray_r", origin="lower")
            task_ax.set_xticks([])
            task_ax.set_yticks([])

            # comm image (cnn output)
            comm_ax = axs[1, scale - 1]
            comm_ax.imshow(comm_img.T, extent=extent, cmap="gray_r", origin="lower")
            comm_plot = comm_ax.plot(
                *task_config.T,
                "r.",
                markersize=10 / scale,
                label="Task Agent Positions",
            )
            comm_ax.set_xticks([])
            comm_ax.set_yticks([])

            if scale == 1:
                task_ax.set_ylabel("Task Image")
                comm_ax.set_ylabel("Comm Image")
                fig.legend(
                    [comm_plot],
                    loc="lower center",
                    ncol=2,
                    bbox_to_anchor=(0.5, 1.05),
                )
    plt.savefig("figures/generalization.pdf")
    plt.show()


if __name__ == "__main__":
    main()
