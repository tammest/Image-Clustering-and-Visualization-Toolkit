# ============================================================
# GT1-GT6 3D SURFACE:
# ANATOMY × POD5->POD10 TRAJECTORY × LOCAL GT COMPOSITION
# ============================================================
#
# X-axis:
#   normalized anatomy
#   -2 = Rostral tissue boundary
#   -1 = Rostral wound edge
#    0 = Wound center
#   +1 = Caudal wound edge
#   +2 = Caudal tissue boundary
#
# Y-axis:
#   consensus-reference pseudotime
#
#   0.0 = POD5-like
#   1.0 = POD10-like
#
# Z-axis / surface color:
#
#   For each SAMPLE and SPATIAL BIN:
#
#       GT-state tiles in this pseudotime bin
#       ------------------------------------- × 100
#       ALL GT1-GT6 tiles in this spatial bin
#
#   Then average those percentages across samples.
#
# IMPORTANT:
#
# - If a sample HAS GT tissue in a spatial bin but has
#   zero tiles for a particular GT/pseudotime cell:
#       abundance = 0%
#
# - If a sample has NO GT tissue at all in that spatial bin:
#       abundance = NaN / no information
#
# Therefore samples do not contribute artificial zeroes to
# anatomical locations where they have no GT representation.
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import Normalize

from scipy.ndimage import gaussian_filter


# ============================================================
# COLUMN SETTINGS
# ============================================================

SAMPLE_COLUMN = "sample_name"
POSITION_COLUMN = "normalized_wound_and_granulation_position"
CLUSTER_COLUMN = "cluster_ids"

PSEUDOTIME_COLUMN = "consensus_reference_pseudotime"
ASSIGNMENT_STATUS_COLUMN = "consensus_reference_assignment_status"


# ============================================================
# INPUT
# ============================================================

PSEUDOTIME_CSV = Path("data/tiles_with_pseudotime_annotation.csv")


# ============================================================
# RECOMMENDED BINNING FROM QC
# ============================================================
#
# 3 spatial bins per anatomical interval:
#
# -2 -> -1   Rostral outer tissue
# -1 ->  0   Rostral wound
#  0 -> +1   Caudal wound
# +1 -> +2   Caudal outer tissue
#
# ============================================================

N_SPATIAL_BINS = 12


# 8 bins across POD5 -> POD10 trajectory
N_PSEUDOTIME_BINS = 8


# Moderate visualization-only smoothing
SMOOTH_SIGMA = 0.8


# ============================================================
# COLOR SETTINGS
# ============================================================

SURFACE_CMAP = cm.turbo


# ------------------------------------------------------------
# OPTIONAL DISPLAY-ONLY COLOR CLIPPING
# ------------------------------------------------------------
#
# False = use true global maximum.
#
# True = use percentile max for COLOR only.
# Surface height remains true.
#
# ------------------------------------------------------------

USE_PERCENTILE_COLOR_MAX = True

COLOR_MAX_PERCENTILE = 99.0


# ============================================================
# OUTPUT
# ============================================================

if "VISUAL_OUTPUT_DIR" not in globals():

    VISUAL_OUTPUT_DIR = Path(
        "visual_outputs"
    )

else:

    VISUAL_OUTPUT_DIR = Path(
        VISUAL_OUTPUT_DIR
    )


OUT_DIR = (
    VISUAL_OUTPUT_DIR
    / "analysis_outputs"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUT_PNG = (
    OUT_DIR
    / "gt_surface_plot.png"
)


OUT_PDF = (
    OUT_DIR
    / "gt_surface_plot.pdf"
)


OUT_CSV = (
    OUT_DIR
    / "gt_surface_summary.csv"
)


# ============================================================
# FLIPPED SAMPLES
# ============================================================

FLIPPED_SAMPLES = {
    # Add sample identifiers here only if needed locally.
    # Keep this empty in the public GitHub version if sample names are sensitive.
}


# ============================================================
# OPTIONAL SAMPLE EXCLUSIONS
# ============================================================

EXCLUDED_SAMPLES = {
    # Add sample identifiers here only if needed locally.
}


# ============================================================
# GT STATES
# ============================================================

PLOT_CLUSTER_IDS = {

    "Granulation Tissue 1": True,
    "Granulation Tissue 2": True,
    "Granulation Tissue 3": True,
    "Granulation Tissue 4": True,
    "Granulation Tissue 5": True,
    "Granulation Tissue 6": True,
}


cluster_order = [
    cluster_name
    for cluster_name, keep in PLOT_CLUSTER_IDS.items()
    if keep
]


cluster_short_labels = {

    "Granulation Tissue 1": "GT1",
    "Granulation Tissue 2": "GT2",
    "Granulation Tissue 3": "GT3",
    "Granulation Tissue 4": "GT4",
    "Granulation Tissue 5": "GT5",
    "Granulation Tissue 6": "GT6",
}


# ============================================================
# CHECK merged
# ============================================================

if "merged" not in globals():

    raise NameError(
        "`merged` was not found in memory."
    )


required_merged_columns = {

    SAMPLE_COLUMN,
    POSITION_COLUMN,
    CLUSTER_COLUMN,

    "is_granulation_tissue",
    "sample_group",
    "superclusters",
    "UMAP1",
    "UMAP2",
}


missing_merged_columns = (
    required_merged_columns
    - set(merged.columns)
)


if missing_merged_columns:

    raise ValueError(
        "Missing required columns from `merged`:\n"
        f"{sorted(missing_merged_columns)}"
    )


# ============================================================
# LOAD PSEUDOTIME
# ============================================================

if not PSEUDOTIME_CSV.exists():

    raise FileNotFoundError(
        f"Pseudotime CSV not found:\n{PSEUDOTIME_CSV}"
    )


pseudotime_df = pd.read_csv(
    PSEUDOTIME_CSV,
    low_memory=False,
)


required_pt_columns = {

    SAMPLE_COLUMN,
    "sample_group",

    PSEUDOTIME_COLUMN,
    ASSIGNMENT_STATUS_COLUMN,

    "superclusters",
    "UMAP1",
    "UMAP2",
}


missing_pt_columns = (
    required_pt_columns
    - set(pseudotime_df.columns)
)


if missing_pt_columns:

    raise ValueError(
        "Missing required pseudotime columns:\n"
        f"{sorted(missing_pt_columns)}"
    )


# ============================================================
# CLEAN BASE TABLES
# ============================================================

plot_base = merged.copy()


plot_base[
    POSITION_COLUMN
] = pd.to_numeric(

    plot_base[
        POSITION_COLUMN
    ],

    errors="coerce",
)


pseudotime_df[
    PSEUDOTIME_COLUMN
] = pd.to_numeric(

    pseudotime_df[
        PSEUDOTIME_COLUMN
    ],

    errors="coerce",
)


for col in [
    "UMAP1",
    "UMAP2",
]:

    plot_base[
        col
    ] = pd.to_numeric(

        plot_base[
            col
        ],

        errors="coerce",
    )


    pseudotime_df[
        col
    ] = pd.to_numeric(

        pseudotime_df[
            col
        ],

        errors="coerce",
    )


# ============================================================
# PREPARE MERGE
# ============================================================

join_columns = [

    "sample_group",
    SAMPLE_COLUMN,
    "superclusters",
    "UMAP1",
    "UMAP2",
]


pseudotime_for_merge = pseudotime_df[

    join_columns
    + [
        PSEUDOTIME_COLUMN,
        ASSIGNMENT_STATUS_COLUMN,
    ]

].copy()


# ============================================================
# DUPLICATE KEY CHECK
# ============================================================

duplicate_pt_keys = (

    pseudotime_for_merge

    .duplicated(
        subset=join_columns,
        keep=False,
    )
)


if duplicate_pt_keys.any():

    print(
        "\nWARNING: duplicate pseudotime merge keys detected."
    )

    print(

        pseudotime_for_merge.loc[

            duplicate_pt_keys,

            join_columns

        ]

        .head(20)

        .to_string(
            index=False
        )
    )

    raise ValueError(
        "Pseudotime merge keys are not unique."
    )


# ============================================================
# MERGE
# ============================================================

plot_df = plot_base.merge(

    pseudotime_for_merge,

    on=join_columns,

    how="left",

    validate="many_to_one",
)


# ============================================================
# CLEAN MERGED FIELDS
# ============================================================

plot_df[
    CLUSTER_COLUMN
] = (

    plot_df[
        CLUSTER_COLUMN
    ]

    .astype("string")

    .str.replace(
        "\ufeff",
        "",
        regex=False,
    )

    .str.strip()
)


plot_df[
    SAMPLE_COLUMN
] = (

    plot_df[
        SAMPLE_COLUMN
    ]

    .astype("string")

    .str.replace(
        "\ufeff",
        "",
        regex=False,
    )

    .str.strip()
)


plot_df[
    POSITION_COLUMN
] = pd.to_numeric(

    plot_df[
        POSITION_COLUMN
    ],

    errors="coerce",
)


plot_df[
    PSEUDOTIME_COLUMN
] = pd.to_numeric(

    plot_df[
        PSEUDOTIME_COLUMN
    ],

    errors="coerce",
)


# ============================================================
# OPTIONAL SAMPLE EXCLUSION
# ============================================================

plot_df = plot_df.loc[

    ~plot_df[
        SAMPLE_COLUMN
    ].isin(
        EXCLUDED_SAMPLES
    )

].copy()


# ============================================================
# MERGE QC
# ============================================================

candidate_gt_mask = (

    plot_df[
        "is_granulation_tissue"
    ].fillna(False)

    & plot_df[
        CLUSTER_COLUMN
    ].isin(
        cluster_order
    )

    & plot_df[
        SAMPLE_COLUMN
    ].notna()
)


n_candidate_gt = int(
    candidate_gt_mask.sum()
)


n_candidate_with_pt = int(

    (
        candidate_gt_mask

        & plot_df[
            PSEUDOTIME_COLUMN
        ].notna()

    ).sum()
)


pt_match_rate = (

    100.0

    * n_candidate_with_pt

    / n_candidate_gt

    if n_candidate_gt > 0

    else np.nan
)


# ============================================================
# KEEP VALID GT TILES
# ============================================================

plot_df = plot_df.loc[

    plot_df[
        "is_granulation_tissue"
    ].fillna(False)

    & plot_df[
        CLUSTER_COLUMN
    ].isin(
        cluster_order
    )

    & plot_df[
        SAMPLE_COLUMN
    ].notna()

    & plot_df[
        POSITION_COLUMN
    ].notna()

    & plot_df[
        PSEUDOTIME_COLUMN
    ].notna()

].copy()


# ============================================================
# KEEP DISPLAYED ANATOMICAL DOMAIN
# ============================================================

plot_df = plot_df.loc[

    plot_df[
        POSITION_COLUMN
    ]

    .between(
        -2.0,
        2.0,
        inclusive="both",
    )

].copy()


if plot_df.empty:

    raise ValueError(
        "No valid GT tiles remained after filtering."
    )


# ============================================================
# ORIENTATION FLIP
# ============================================================

flip_mask = (

    plot_df[
        SAMPLE_COLUMN
    ]

    .isin(
        FLIPPED_SAMPLES
    )
)


plot_df.loc[

    flip_mask,

    POSITION_COLUMN

] *= -1.0


# ============================================================
# PSEUDOTIME RANGE
# ============================================================

pt_min_observed = float(

    plot_df[
        PSEUDOTIME_COLUMN
    ].min()
)


pt_max_observed = float(

    plot_df[
        PSEUDOTIME_COLUMN
    ].max()
)


if np.isclose(
    pt_min_observed,
    pt_max_observed,
):

    raise ValueError(
        "Pseudotime has zero range."
    )


# ============================================================
# PSEUDOTIME DISPLAY RANGE
# ============================================================
#
# Consensus-reference pseudotime is interpreted on 0 -> 1.
#
# We therefore explicitly display:
#
# 0.0 = POD5-like
# 0.2
# 0.4
# 0.6
# 0.8
# 1.0 = POD10-like
#
# ============================================================

PT_DISPLAY_MIN = 0.0
PT_DISPLAY_MAX = 1.0


# ============================================================
# CREATE BIN EDGES
# ============================================================

spatial_edges = np.linspace(

    -2.0,
    2.0,

    N_SPATIAL_BINS + 1,
)


pt_edges = np.linspace(

    PT_DISPLAY_MIN,
    PT_DISPLAY_MAX,

    N_PSEUDOTIME_BINS + 1,
)


# ============================================================
# BIN CENTERS
# ============================================================

spatial_centers = (

    spatial_edges[:-1]
    + spatial_edges[1:]

) / 2.0


pt_centers = (

    pt_edges[:-1]
    + pt_edges[1:]

) / 2.0


# ============================================================
# ASSIGN BINS
# ============================================================

plot_df[
    "spatial_bin"
] = pd.cut(

    plot_df[
        POSITION_COLUMN
    ],

    bins=spatial_edges,

    labels=False,

    include_lowest=True,
)


plot_df[
    "pt_bin"
] = pd.cut(

    plot_df[
        PSEUDOTIME_COLUMN
    ],

    bins=pt_edges,

    labels=False,

    include_lowest=True,
)


plot_df = plot_df.dropna(

    subset=[
        "spatial_bin",
        "pt_bin",
    ]

).copy()


plot_df[
    "spatial_bin"
] = (

    plot_df[
        "spatial_bin"
    ]

    .astype(int)
)


plot_df[
    "pt_bin"
] = (

    plot_df[
        "pt_bin"
    ]

    .astype(int)
)


# ============================================================
# LOCAL SPATIAL GT COMPOSITION
# ============================================================
#
# DENOMINATOR:
#
#   ALL GT1-GT6 tiles in:
#
#       sample × spatial bin
#
#
# NUMERATOR:
#
#   tiles in:
#
#       sample × GT × spatial bin × PT bin
#
#
# Therefore the entire local GT compartment at each spatial
# location is the 100% reference.
#
# ============================================================


# ============================================================
# LOCAL GT TOTAL PER SAMPLE × SPATIAL BIN
# ============================================================

sample_spatial_gt_total = (

    plot_df

    .groupby(
        [
            SAMPLE_COLUMN,
            "spatial_bin",
        ],
        observed=True,
    )

    .size()

    .rename(
        "sample_spatial_total_gt_tiles"
    )

    .reset_index()
)


# ============================================================
# COUNT GT × PT CELLS
# ============================================================

sample_cell_counts = (

    plot_df

    .groupby(
        [
            SAMPLE_COLUMN,
            CLUSTER_COLUMN,
            "spatial_bin",
            "pt_bin",
        ],
        observed=True,
    )

    .size()

    .rename(
        "n_tiles"
    )

    .reset_index()
)


# ============================================================
# MERGE LOCAL DENOMINATOR
# ============================================================

sample_cell_counts = (

    sample_cell_counts

    .merge(

        sample_spatial_gt_total,

        on=[
            SAMPLE_COLUMN,
            "spatial_bin",
        ],

        how="left",
    )
)


# ============================================================
# LOCAL GT COMPOSITION %
# ============================================================

sample_cell_counts[
    "local_gt_composition_percent"
] = (

    100.0

    * sample_cell_counts[
        "n_tiles"
    ]

    / sample_cell_counts[
        "sample_spatial_total_gt_tiles"
    ]
)


# ============================================================
# COMPLETE ONLY INFORMATIVE SAMPLE × SPATIAL LOCATIONS
# ============================================================
#
# If a sample has GT tiles in a spatial bin:
#   missing GT × PT combinations = 0%
#
# If a sample has NO GT tiles in a spatial bin:
#   that sample/spatial location is not added.
#
# ============================================================

informative_sample_spatial = (

    sample_spatial_gt_total[
        [
            SAMPLE_COLUMN,
            "spatial_bin",
        ]
    ]

    .drop_duplicates()
)


complete_rows = []


for _, base_row in (

    informative_sample_spatial.iterrows()
):

    sample_name = base_row[
        SAMPLE_COLUMN
    ]


    spatial_bin = int(
        base_row[
            "spatial_bin"
        ]
    )


    for cluster_name in cluster_order:

        for pt_bin in range(
            N_PSEUDOTIME_BINS
        ):

            complete_rows.append(
                (
                    sample_name,
                    cluster_name,
                    spatial_bin,
                    pt_bin,
                )
            )


complete_grid = pd.DataFrame(

    complete_rows,

    columns=[
        SAMPLE_COLUMN,
        CLUSTER_COLUMN,
        "spatial_bin",
        "pt_bin",
    ],
)


# ============================================================
# MERGE MEASURED LOCAL COMPOSITION
# ============================================================

complete_grid = complete_grid.merge(

    sample_cell_counts[
        [
            SAMPLE_COLUMN,
            CLUSTER_COLUMN,
            "spatial_bin",
            "pt_bin",
            "local_gt_composition_percent",
        ]
    ],

    on=[
        SAMPLE_COLUMN,
        CLUSTER_COLUMN,
        "spatial_bin",
        "pt_bin",
    ],

    how="left",
)


# ============================================================
# TRUE ZERO IF GT EXISTS THERE BUT THIS GT/PT CELL DOES NOT
# ============================================================

complete_grid[
    "local_gt_composition_percent"
] = (

    complete_grid[
        "local_gt_composition_percent"
    ]

    .fillna(
        0.0
    )
)


# ============================================================
# AVERAGE ACROSS INFORMATIVE SAMPLES
# ============================================================

summary = (

    complete_grid

    .groupby(
        [
            CLUSTER_COLUMN,
            "spatial_bin",
            "pt_bin",
        ],
        observed=True,
    )

    .agg(

        mean_local_gt_composition_percent=(
            "local_gt_composition_percent",
            "mean",
        ),

        median_local_gt_composition_percent=(
            "local_gt_composition_percent",
            "median",
        ),

        n_informative_samples=(
            SAMPLE_COLUMN,
            "nunique",
        ),
    )

    .reset_index()
)


# ============================================================
# ADD BIN CENTERS
# ============================================================

summary[
    "spatial_bin_center"
] = (

    summary[
        "spatial_bin"
    ]

    .map(
        dict(
            enumerate(
                spatial_centers
            )
        )
    )
)


summary[
    "pseudotime_bin_center"
] = (

    summary[
        "pt_bin"
    ]

    .map(
        dict(
            enumerate(
                pt_centers
            )
        )
    )
)


# ============================================================
# SAVE UNSMOOTHED SUMMARY
# ============================================================

summary.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# BUILD RAW + SMOOTHED SURFACE MATRICES
# ============================================================

surface_matrices_raw = {}

surface_matrices_smooth = {}

global_z_max = 0.0


for cluster_name in cluster_order:

    cluster_summary = summary.loc[

        summary[
            CLUSTER_COLUMN
        ]
        == cluster_name

    ].copy()


    Z_raw = np.full(
        (
            N_PSEUDOTIME_BINS,
            N_SPATIAL_BINS,
        ),
        np.nan,
        dtype=float,
    )


    for _, row in (
        cluster_summary.iterrows()
    ):

        sbin = int(
            row[
                "spatial_bin"
            ]
        )


        pbin = int(
            row[
                "pt_bin"
            ]
        )


        Z_raw[
            pbin,
            sbin
        ] = row[
            "mean_local_gt_composition_percent"
        ]


    surface_matrices_raw[
        cluster_name
    ] = Z_raw.copy()


    # ========================================================
    # NaN-AWARE SMOOTHING
    # ========================================================

    if (
        SMOOTH_SIGMA is not None
        and SMOOTH_SIGMA > 0
    ):

        valid_mask = np.isfinite(
            Z_raw
        ).astype(
            float
        )


        Z_filled = np.nan_to_num(
            Z_raw,
            nan=0.0,
        )


        smooth_values = gaussian_filter(

            Z_filled,

            sigma=SMOOTH_SIGMA,

            mode="nearest",
        )


        smooth_weights = gaussian_filter(

            valid_mask,

            sigma=SMOOTH_SIGMA,

            mode="nearest",
        )


        with np.errstate(
            invalid="ignore",
            divide="ignore",
        ):

            Z_smooth = (
                smooth_values
                / smooth_weights
            )


        Z_smooth[
            smooth_weights
            <= 1e-12
        ] = np.nan


    else:

        Z_smooth = Z_raw.copy()


    surface_matrices_smooth[
        cluster_name
    ] = Z_smooth


    if np.isfinite(
        Z_smooth
    ).any():

        global_z_max = max(

            global_z_max,

            float(
                np.nanmax(
                    Z_smooth
                )
            )
        )


# ============================================================
# MESH GRID
# ============================================================

X, Y = np.meshgrid(
    spatial_centers,
    pt_centers,
)


# ============================================================
# COLOR SCALE
# ============================================================

all_surface_values = np.concatenate(
    [
        Z[
            np.isfinite(Z)
        ]

        for Z
        in surface_matrices_smooth.values()

        if np.isfinite(Z).any()
    ]
)


if (
    USE_PERCENTILE_COLOR_MAX
    and len(
        all_surface_values
    ) > 0
):

    color_max = float(

        np.nanpercentile(
            all_surface_values,
            COLOR_MAX_PERCENTILE,
        )
    )


else:

    color_max = float(
        global_z_max
    )


if (
    not np.isfinite(
        color_max
    )
    or color_max <= 0
):

    color_max = 1.0


surface_norm = Normalize(
    vmin=0.0,
    vmax=color_max,
    clip=True,
)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(
        18,
        11,
    )
)


# ============================================================
# DRAW GT SURFACES
# ============================================================

for idx, cluster_name in enumerate(
    cluster_order,
    start=1,
):

    ax = fig.add_subplot(
        2,
        3,
        idx,
        projection="3d",
    )


    Z = surface_matrices_smooth[
        cluster_name
    ]


    facecolors = SURFACE_CMAP(

        surface_norm(

            np.nan_to_num(
                Z,
                nan=0.0,
            )
        )
    )


    # No-information regions transparent
    facecolors[..., 3] = (

        np.isfinite(
            Z
        )

        .astype(float)

        * 0.98
    )


    ax.plot_surface(

        X,
        Y,

        np.ma.masked_invalid(
            Z
        ),

        facecolors=facecolors,

        rstride=1,
        cstride=1,

        linewidth=0,

        antialiased=True,

        shade=False,
    )


    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(

        cluster_short_labels.get(
            cluster_name,
            cluster_name,
        ),

        fontsize=14,

        weight="bold",

        pad=10,
    )


    # ========================================================
    # X = ANATOMY
    # ========================================================

    ax.set_xlim(
        -2.0,
        2.0,
    )


    ax.set_xticks(
        [
            -2,
            -1,
            0,
            1,
            2,
        ]
    )


    ax.set_xticklabels(
        [
            "Rostral Tis.\nBoundary",
            "Rostral Wound\nEdge",
            "Wound\nCenter",
            "Caudal Wound\nEdge",
            "Caudal Tis.\nBoundary",
        ],
        fontsize=8,
    )


    ax.set_xlabel(
        "Normalized wound position",
        fontsize=9,
        labelpad=10,
    )


    # ========================================================
    # Y = POD5 -> POD10 PSEUDOTIME
    # ========================================================

    ax.set_ylim(
        0.0,
        1.0,
    )


    ax.set_yticks(
        [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ]
    )


    ax.set_yticklabels(
        [
            "0.0\nPOD5-like",
            "0.2",
            "0.4",
            "0.6",
            "0.8",
            "1.0\nPOD10-like",
        ],
        fontsize=8,
    )


    ax.set_ylabel(
        "POD5 → POD10 pseudotime",
        fontsize=9,
        labelpad=10,
    )


    # ========================================================
    # Z = LOCAL GT COMPOSITION
    # ========================================================

    if global_z_max > 0:

        ax.set_zlim(
            0.0,
            global_z_max * 1.08,
        )

    else:

        ax.set_zlim(
            0.0,
            1.0,
        )


    ax.set_zlabel(
        "Mean local GT composition (%)",
        fontsize=9,
        labelpad=8,
    )


    # ========================================================
    # VIEW
    # ========================================================

    ax.view_init(
        elev=28,
        azim=-58,
    )


    try:

        ax.xaxis.pane.set_alpha(
            0.03
        )

        ax.yaxis.pane.set_alpha(
            0.03
        )

        ax.zaxis.pane.set_alpha(
            0.03
        )

    except Exception:

        pass


    ax.grid(
        True
    )


# ============================================================
# SHARED COLORBAR
# ============================================================

scalar_mappable = cm.ScalarMappable(
    norm=surface_norm,
    cmap=SURFACE_CMAP,
)


scalar_mappable.set_array([])


cbar_ax = fig.add_axes(
    [
        0.92,
        0.24,
        0.018,
        0.52,
    ]
)


cbar = fig.colorbar(
    scalar_mappable,
    cax=cbar_ax,
)


cbar.set_label(
    "Mean local GT composition (%)",
    fontsize=10,
)


cbar.ax.tick_params(
    labelsize=8
)


# ============================================================
# TITLE
# ============================================================

n_samples = int(
    plot_df[
        SAMPLE_COLUMN
    ].nunique()
)


n_tiles = int(
    len(
        plot_df
    )
)


fig.suptitle(

    "Local GT composition across anatomy and the POD5 → POD10 trajectory\n"

    "At each anatomical position: "
    "height/color = mean % of the local GT compartment\n"

    "represented by each GT state at each trajectory position\n"

    f"{n_samples} samples | "
    f"{n_tiles:,} assigned GT tiles",

    fontsize=16,

    weight="bold",

    y=0.98,
)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(

    left=0.03,

    right=0.89,

    top=0.86,

    bottom=0.05,

    wspace=0.03,

    hspace=0.05,
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(

    OUT_PNG,

    dpi=400,

    bbox_inches="tight",

    facecolor="white",
)


fig.savefig(

    OUT_PDF,

    dpi=400,

    bbox_inches="tight",

    facecolor="white",
)


plt.show()

plt.close(
    fig
)


# ============================================================
# QC
# ============================================================

print(
    "\n"
    + "=" * 100
)


print(
    "3D LOCAL GT COMPOSITION × ANATOMY × POD5->POD10"
)


print(
    "=" * 100
)


print(
    f"\nCandidate selected GT tiles before PT filter: "
    f"{n_candidate_gt:,}"
)


print(
    f"Candidate GT tiles with pseudotime: "
    f"{n_candidate_with_pt:,}"
)


print(
    f"Pseudotime match rate: "
    f"{pt_match_rate:.2f}%"
)


print(
    f"\nSamples included: "
    f"{n_samples}"
)


print(
    f"Assigned GT tiles included: "
    f"{n_tiles:,}"
)


print(
    f"Observed pseudotime range: "
    f"{pt_min_observed:.4f} -> "
    f"{pt_max_observed:.4f}"
)


print(
    f"Displayed pseudotime range: "
    f"{PT_DISPLAY_MIN:.1f} -> "
    f"{PT_DISPLAY_MAX:.1f}"
)


print(
    f"Spatial bins: "
    f"{N_SPATIAL_BINS}"
)


print(
    f"Pseudotime bins: "
    f"{N_PSEUDOTIME_BINS}"
)


print(
    f"Smoothing sigma: "
    f"{SMOOTH_SIGMA}"
)


# ============================================================
# ABUNDANCE DEFINITION
# ============================================================

print(
    "\nABUNDANCE DEFINITION:"
)


print(
    "  For each sample × spatial bin:"
)


print(
    "      100 × "
    "(GT-state tiles in a pseudotime bin)"
)


print(
    "      ---------------------------------------------"
)


print(
    "      ALL GT1-GT6 tiles in that spatial bin"
)


print(
    "\n  Then values are averaged across samples that "
    "actually contain GT tiles in that spatial bin."
)


print(
    "\n  Sample with GT there but no specific GT/PT cell = 0%"
)


print(
    "  Sample with no GT there at all = excluded / NaN"
)


# ============================================================
# CHECK LOCAL TOTALS
# ============================================================

local_sum_check = (

    complete_grid

    .groupby(
        [
            SAMPLE_COLUMN,
            "spatial_bin",
        ],
        observed=True,
    )[
        "local_gt_composition_percent"
    ]

    .sum()

    .reset_index(
        name="sum_percent"
    )
)


print(
    "\nPer-sample spatial-bin composition sum QC:"
)


print(

    local_sum_check[
        "sum_percent"
    ]

    .describe()

    .round(
        4
    )

    .to_string()
)


max_sum_error = float(

    np.nanmax(

        np.abs(

            local_sum_check[
                "sum_percent"
            ]

            - 100.0
        )
    )
)


print(
    f"\nMaximum deviation from 100%: "
    f"{max_sum_error:.8f}"
)


# ============================================================
# INFORMATIVE SAMPLE SUPPORT PER SPATIAL BIN
# ============================================================

spatial_sample_support = (

    sample_spatial_gt_total

    .groupby(
        "spatial_bin",
        observed=True,
    )[
        SAMPLE_COLUMN
    ]

    .nunique()

    .reindex(
        range(
            N_SPATIAL_BINS
        )
    )
)


print(
    "\nInformative samples per spatial bin:"
)


for spatial_bin, count in (
    spatial_sample_support.items()
):

    center = spatial_centers[
        int(
            spatial_bin
        )
    ]


    print(

        f"  bin {int(spatial_bin):2d} "

        f"(center {center:+.3f}): "

        f"{int(count) if pd.notna(count) else 0} samples"
    )


# ============================================================
# RAW GT COUNTS
# ============================================================

print(
    "\nRaw assigned tile counts per GT:"
)


tile_counts = (

    plot_df[
        CLUSTER_COLUMN
    ]

    .value_counts()

    .reindex(
        cluster_order
    )
)


print(
    tile_counts.to_string()
)


# ============================================================
# GLOBAL Z
# ============================================================

print(
    f"\nGlobal maximum plotted local GT composition: "
    f"{global_z_max:.4f}%"
)


print(
    f"Color maximum: "
    f"{color_max:.4f}%"
)


if USE_PERCENTILE_COLOR_MAX:

    print(

        f"Color scale clipped at "
        f"{COLOR_MAX_PERCENTILE:g}th percentile."
    )

else:

    print(
        "Color scale uses true global maximum."
    )


# ============================================================
# SAVE PATHS
# ============================================================

print(
    "\n[✓] Saved unsmoothed summary CSV:"
)


print(
    f"    {OUT_CSV}"
)


print(
    "\n[✓] Saved PNG:"
)


print(
    f"    {OUT_PNG}"
)


print(
    "\n[✓] Saved PDF:"
)


print(
    f"    {OUT_PDF}"
)


print(
    "\n[✓] Local spatial GT-composition surface complete."
)
