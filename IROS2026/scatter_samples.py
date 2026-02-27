import numpy as np
import matplotlib as mpl
import matplotlib
# matplotlib.use('Qt5Agg')  # Use Qt5 interactive backend
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

# === IEEE-like font family and sizes (10pt doc) ===
S = {
    "normalsize": 18,      # body text
    "small": 16,            # axis labels / lane labels
    "footnotesize": 14,     # tick labels / legend
    "large": 22,           # figure title
}
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",                   # Times-like math
    "axes.titlesize": S["large"],
    "axes.labelsize": S["normalsize"],
    "xtick.labelsize": S["large"],
    "ytick.labelsize": S["large"],
    "legend.fontsize": S["large"],
    "pdf.fonttype": 42, "ps.fonttype": 42,        # keep text searchable
})

# Pick an available Times-like font so figures stay consistent on systems
# without proprietary Times faces.
_FONT_CANDIDATES = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "DejaVu Serif",
    "STIXGeneral",
]


def _select_font_property():
    for family in _FONT_CANDIDATES:
        prop = font_manager.FontProperties(family=family)
        try:
            font_manager.findfont(prop, fallback_to_default=False)
        except ValueError:
            continue
        return prop
    return font_manager.FontProperties(family="serif")


_BASE_FONT = _select_font_property()


def font_prop(size_key: str) -> font_manager.FontProperties:
    prop = _BASE_FONT.copy()
    prop.set_size(S[size_key])
    return prop

method = "fop" # "fop" or "sparse"
time_step = 26
samples_file = f"data/output/sampling_parameters/{method}/DEU_Schwetzingen-12_1_T-2.csv"
save_path = f"/home/kareem/fiss_plus_planner/IROS2026/imgs/DEU_Schwetzingen-12_1_T-2_{method}_time_{time_step}.pdf"

file = pd.read_csv(samples_file)
# reps = [5 if val == time_step else 1 for val in file["time_step"]]
# file = file.loc[np.repeat(file.index.values, reps)]
# file = file.append(file[file["time_step"] == time_step]*5, ignore_index=True)
samples_at_time_step = file[file["time_step"] == time_step]
samples_at_time_step_np = samples_at_time_step.to_numpy()
print(f"Samples at time step {time_step}:")
print(samples_at_time_step_np)

d_min_max = (samples_at_time_step["d"].min(), samples_at_time_step["d"].max())
sv_min_max = (samples_at_time_step["s_d"].min(), samples_at_time_step["s_d"].max())
t_min_max = (samples_at_time_step["t"].min(), samples_at_time_step["t"].max())

d_range = np.linspace(d_min_max[0]-1, d_min_max[1]+1, 5)
sv_range = np.linspace(sv_min_max[0]-1, sv_min_max[1]+1, 5)
t_range = np.linspace(t_min_max[0]-1, t_min_max[1]+1, 5)

def plot_3d_scatter(samples, elev=18, azim=45, save_path=None):
    """
    Plot FOP structured sampling grid in 3D space.
    d: lateral offset [m]
    sv: longitudinal velocity [m/s]
    t: time horizon [s]
    """

    t, d, sv, costs = samples[:, 1], samples[:, 2], samples[:, 3], samples[:, 4]

    # --- Plot ---
    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    
    vmin, vmax = 0, 90

    # FOP grid (blue, structured)
    ax.scatter(
        d, sv, t,
        s=20,
        alpha=1.0,
        c=costs, 
        cmap="RdYlGn_r",
        # these should be the same as the scenario plot to have the same range for color bar
        vmin=vmin, 
        vmax=vmax,
        marker="o"
    )
    # ax.cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.1)
    # ax.cbar.set_label("Cost", fontproperties=font_prop("small"))

    ax.set_xlabel(r"$d$ [m]", fontproperties=font_prop("large"), labelpad=15)
    ax.set_ylabel(r"$\dot{s}$ [m/s]", fontproperties=font_prop("large"), labelpad=15)
    ax.set_zlabel(r"$t$ [s]", fontproperties=font_prop("large"), labelpad=10)

    # ax.set_xticks(np.round(d_range, 1))
    # ax.set_yticks(np.round(sv_range, 1))
    # ax.set_zticks(np.round(t_range, 1))
    ax.set_xticks(np.linspace(-1.8, 1.8, 5))
    ax.set_yticks(np.linspace(-1, 15, 5))
    ax.set_zticks(np.linspace(2, 6, 5))

    ax.view_init(elev=elev, azim=azim)

    # Apply tick label fonts
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_prop("large"))
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_prop("large"))
    for tick in ax.get_zticklabels():
        tick.set_fontproperties(font_prop("large"))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# run as: python ../IROS2026/scatter_samples.py
plot_3d_scatter(samples_at_time_step.to_numpy(), elev=18, azim=45, save_path=save_path)