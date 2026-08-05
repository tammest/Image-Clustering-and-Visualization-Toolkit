from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

MAIN_CSV = Path(
    "rabbanilab/wsi_analysis/input_csv_files/"
    "n_of_3_with_rotated_coords.csv"
)

WOUND_CSV = Path(
    "rabbanilab/wsi_analysis/input_csv_files/"
    "wound_coordinates.csv"
)

OUT_CSV = Path(
    "rabbanilab/wsi_analysis/input_csv_files/"
    "n_of_3_with_rotated_coords_and_wound_info.csv"
)

VISUAL_OUTPUT_DIR = Path(
    "rabbanilab/wsi_analysis/output/"
    "wound_normalization_qc"
)


# ============================================================
# VISUAL QC CONFIGURATION
# ============================================================

# Visual QC plots are generated automatically for every sample
# found in the merged tile dataset.


# ============================================================
# CLEANING HELPERS
# ============================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove hidden byte-order marks and whitespace from column names.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return df


def clean_sample_names(series: pd.Series) -> pd.Series:
    """
    Standardize sample-name strings.
    """
    return (
        series
        .astype("string")
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )


def convert_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Convert selected columns to numeric.

    Values such as Na, NA, blank strings, and invalid strings
    become np.nan.
    """
    df = df.copy()

    missing_strings = {
        "Na": np.nan,
        "NA": np.nan,
        "N/A": np.nan,
        "na": np.nan,
        "n/a": np.nan,
        "Nan": np.nan,
        "nan": np.nan,
        "": np.nan,
        " ": np.nan,
    }

    for column in columns:
        if column not in df.columns:
            continue

        df[column] = df[column].replace(
            missing_strings
        )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def safe_filename(name: str) -> str:
    """
    Convert a sample name into a safe filename.
    """
    safe = str(name)

    invalid_characters = [
        "/",
        "\\",
        ":",
        "*",
        "?",
        '"',
        "<",
        ">",
        "|",
        " ",
    ]

    for character in invalid_characters:
        safe = safe.replace(
            character,
            "_",
        )

    while "__" in safe:
        safe = safe.replace(
            "__",
            "_",
        )

    return safe.strip("_")


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def normalize_wound_position(
    rotated_w: pd.Series,
    left_edge: pd.Series,
    wound_center: pd.Series,
    right_edge: pd.Series,
) -> pd.Series:
    """
    Normalize each tile's position relative to the wound.

    Mapping:
        left wound edge  -> -1
        wound center     ->  0
        right wound edge -> +1

    Each side is normalized independently.

    If the left edge is missing:
        left-side positions remain NaN;
        right-side positions are still normalized.

    If the right edge is missing:
        right-side positions remain NaN;
        left-side positions are still normalized.

    Values outside the wound may extend below -1 or above +1.
    """
    normalized = pd.Series(
        np.nan,
        index=rotated_w.index,
        dtype=float,
    )

    valid_x = rotated_w.notna()
    valid_center = wound_center.notna()

    # --------------------------------------------------------
    # LEFT SIDE
    # --------------------------------------------------------

    valid_left = (
        valid_x
        & valid_center
        & left_edge.notna()
        & (wound_center > left_edge)
        & (rotated_w <= wound_center)
    )

    normalized.loc[valid_left] = (
        rotated_w.loc[valid_left]
        - wound_center.loc[valid_left]
    ) / (
        wound_center.loc[valid_left]
        - left_edge.loc[valid_left]
    )

    # --------------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------------

    valid_right = (
        valid_x
        & valid_center
        & right_edge.notna()
        & (right_edge > wound_center)
        & (rotated_w >= wound_center)
    )

    normalized.loc[valid_right] = (
        rotated_w.loc[valid_right]
        - wound_center.loc[valid_right]
    ) / (
        right_edge.loc[valid_right]
        - wound_center.loc[valid_right]
    )

    # --------------------------------------------------------
    # EXACT CENTER
    # --------------------------------------------------------

    at_center = (
        valid_x
        & valid_center
        & np.isclose(
            rotated_w,
            wound_center,
            equal_nan=False,
        )
    )

    normalized.loc[at_center] = 0.0

    return normalized


def normalize_wound_and_granulation_position(df):
    """
    Extended biological normalization:

        left granulation boundary  -> -2
        left wound edge            -> -1
        wound center               ->  0
        right wound edge           -> +1
        right granulation boundary -> +2

    Only tiles whose superclusters value is exactly
    "Granulation Tissue" receive any value from -2 through +2,
    including the wound interior between the left and right edges.
    """
    normalized = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    x = df["rotated_w"]
    center = df["wound_center"]
    left_edge = df["left_wound_edge"]
    right_edge = df["right_wound_edge"]
    left_granulation = df["left_granulation_tissue_range"]
    right_granulation = df["right_granulation_tissue_range"]
    is_granulation = df["is_granulation_tissue"].fillna(False)

    # Left wound: -1 to 0
    mask = (
        x.notna()
        & is_granulation
        & center.notna()
        & left_edge.notna()
        & (center > left_edge)
        & (x >= left_edge)
        & (x <= center)
    )

    normalized.loc[mask] = (
        x.loc[mask] - center.loc[mask]
    ) / (
        center.loc[mask] - left_edge.loc[mask]
    )

    # Right wound: 0 to +1
    mask = (
        x.notna()
        & is_granulation
        & center.notna()
        & right_edge.notna()
        & (right_edge > center)
        & (x >= center)
        & (x <= right_edge)
    )

    normalized.loc[mask] = (
        x.loc[mask] - center.loc[mask]
    ) / (
        right_edge.loc[mask] - center.loc[mask]
    )

    # Left granulation: -2 to -1
    mask = (
        x.notna()
        & is_granulation
        & left_granulation.notna()
        & left_edge.notna()
        & (left_edge > left_granulation)
        & (x >= left_granulation)
        & (x < left_edge)
    )

    normalized.loc[mask] = (
        -2.0
        + (
            x.loc[mask]
            - left_granulation.loc[mask]
        )
        / (
            left_edge.loc[mask]
            - left_granulation.loc[mask]
        )
    )

    # Right granulation: +1 to +2
    mask = (
        x.notna()
        & is_granulation
        & right_edge.notna()
        & right_granulation.notna()
        & (right_granulation > right_edge)
        & (x > right_edge)
        & (x <= right_granulation)
    )

    normalized.loc[mask] = (
        1.0
        + (
            x.loc[mask]
            - right_edge.loc[mask]
        )
        / (
            right_granulation.loc[mask]
            - right_edge.loc[mask]
        )
    )

    # Exact wound center.
    mask = (
        x.notna()
        & is_granulation
        & center.notna()
        & np.isclose(
            x,
            center,
            equal_nan=False,
        )
    )

    normalized.loc[mask] = 0.0

    return normalized


def assign_normalization_status(
    left_edge: pd.Series,
    wound_center: pd.Series,
    right_edge: pd.Series,
) -> pd.Series:
    """
    Describe which wound sides are available for normalization.
    """
    status = pd.Series(
        "not_normalized_missing_wound_center",
        index=wound_center.index,
        dtype="string",
    )

    center_exists = wound_center.notna()
    left_exists = left_edge.notna()
    right_exists = right_edge.notna()

    status.loc[
        center_exists
        & left_exists
        & right_exists
    ] = "both_sides_normalized"

    status.loc[
        center_exists
        & ~left_exists
        & right_exists
    ] = "right_side_only_missing_left_edge"

    status.loc[
        center_exists
        & left_exists
        & ~right_exists
    ] = "left_side_only_missing_right_edge"

    status.loc[
        center_exists
        & ~left_exists
        & ~right_exists
    ] = "not_normalized_missing_both_edges"

    return status


def assign_wound_side(
    rotated_w: pd.Series,
    wound_center: pd.Series,
) -> pd.Series:
    """
    Assign each tile to the left, center, or right side.
    """
    side = pd.Series(
        pd.NA,
        index=rotated_w.index,
        dtype="string",
    )

    valid = (
        rotated_w.notna()
        & wound_center.notna()
    )

    side.loc[
        valid
        & (rotated_w < wound_center)
    ] = "left"

    side.loc[
        valid
        & (rotated_w > wound_center)
    ] = "right"

    center_mask = (
        valid
        & np.isclose(
            rotated_w,
            wound_center,
            equal_nan=False,
        )
    )

    side.loc[
        center_mask
    ] = "center"

    return side


# ============================================================
# WOUND-REGION FUNCTION
# ============================================================

def assign_wound_region(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Classify tile locations using available wound landmarks.

    Possible regions include:
        left_outer_tissue
        left_granulation_tissue
        left_wound
        wound_center
        right_wound
        right_granulation_tissue
        right_outer_tissue

    Fallback labels are used when one boundary is unavailable.
    """
    region = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    x = df["rotated_w"]
    center = df["wound_center"]

    left_granulation = (
        df["left_granulation_tissue_range"]
    )

    left_edge = (
        df["left_wound_edge"]
    )

    right_edge = (
        df["right_wound_edge"]
    )

    right_granulation = (
        df["right_granulation_tissue_range"]
    )

    valid = (
        x.notna()
        & center.notna()
    )

    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    center_mask = (
        valid
        & np.isclose(
            x,
            center,
            equal_nan=False,
        )
    )

    region.loc[
        center_mask
    ] = "wound_center"

    # --------------------------------------------------------
    # LEFT SIDE
    # --------------------------------------------------------

    left_side = (
        valid
        & (x < center)
    )

    # Left half of wound
    mask = (
        left_side
        & left_edge.notna()
        & (x >= left_edge)
    )

    region.loc[
        mask
    ] = "left_wound"

    # Left granulation tissue
    mask = (
        left_side
        & left_edge.notna()
        & left_granulation.notna()
        & df["is_granulation_tissue"].fillna(False)
        & (x >= left_granulation)
        & (x < left_edge)
    )

    region.loc[
        mask
    ] = "left_granulation_tissue"

    # Tissue beyond left granulation boundary
    mask = (
        left_side
        & left_granulation.notna()
        & (x < left_granulation)
    )

    region.loc[
        mask
    ] = "left_outer_tissue"

    # Left edge unavailable
    mask = (
        left_side
        & left_edge.isna()
        & region.isna()
    )

    region.loc[
        mask
    ] = "left_side_boundary_unavailable"

    # Left edge exists but granulation range is unavailable
    mask = (
        left_side
        & left_edge.notna()
        & (x < left_edge)
        & left_granulation.isna()
        & region.isna()
    )

    region.loc[
        mask
    ] = "left_nonwound_tissue"

    # --------------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------------

    right_side = (
        valid
        & (x > center)
    )

    # Right half of wound
    mask = (
        right_side
        & right_edge.notna()
        & (x <= right_edge)
    )

    region.loc[
        mask
    ] = "right_wound"

    # Right granulation tissue
    mask = (
        right_side
        & right_edge.notna()
        & right_granulation.notna()
        & df["is_granulation_tissue"].fillna(False)
        & (x > right_edge)
        & (x <= right_granulation)
    )

    region.loc[
        mask
    ] = "right_granulation_tissue"

    # Tissue beyond right granulation boundary
    mask = (
        right_side
        & right_granulation.notna()
        & (x > right_granulation)
    )

    region.loc[
        mask
    ] = "right_outer_tissue"

    # Right edge unavailable
    mask = (
        right_side
        & right_edge.isna()
        & region.isna()
    )

    region.loc[
        mask
    ] = "right_side_boundary_unavailable"

    # Right edge exists but granulation range is unavailable
    mask = (
        right_side
        & right_edge.notna()
        & (x > right_edge)
        & right_granulation.isna()
        & region.isna()
    )

    region.loc[
        mask
    ] = "right_nonwound_tissue"

    return region


# ============================================================
# WOUND-TABLE VALIDATION
# ============================================================

def validate_wound_table(
    wounds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate each available wound landmark independently.
    """
    validation = wounds.copy()

    validation[
        "left_side_landmarks_valid"
    ] = (
        validation["left_wound_edge"].isna()
        | (
            validation["wound_center"].notna()
            & (
                validation["left_wound_edge"]
                < validation["wound_center"]
            )
        )
    )

    validation[
        "right_side_landmarks_valid"
    ] = (
        validation["right_wound_edge"].isna()
        | (
            validation["wound_center"].notna()
            & (
                validation["wound_center"]
                < validation["right_wound_edge"]
            )
        )
    )

    validation[
        "left_granulation_landmarks_valid"
    ] = (
        validation[
            "left_granulation_tissue_range"
        ].isna()
        | validation["left_wound_edge"].isna()
        | (
            validation[
                "left_granulation_tissue_range"
            ]
            < validation["left_wound_edge"]
        )
    )

    validation[
        "right_granulation_landmarks_valid"
    ] = (
        validation[
            "right_granulation_tissue_range"
        ].isna()
        | validation["right_wound_edge"].isna()
        | (
            validation["right_wound_edge"]
            < validation[
                "right_granulation_tissue_range"
            ]
        )
    )

    validation[
        "all_available_landmarks_valid"
    ] = (
        validation["wound_center"].notna()
        & validation[
            "left_side_landmarks_valid"
        ]
        & validation[
            "right_side_landmarks_valid"
        ]
        & validation[
            "left_granulation_landmarks_valid"
        ]
        & validation[
            "right_granulation_landmarks_valid"
        ]
    )

    return validation


# ============================================================
# VISUAL QC FUNCTIONS
# ============================================================

def plot_rotated_coordinate_check(
    sample_df: pd.DataFrame,
    sample_name: str,
    output_dir: Path,
) -> Path:
    """
    Plot the rotated tile coordinates with wound-landmark lines.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_df = sample_df.dropna(
        subset=[
            "rotated_w",
            "rotated_h",
        ]
    )

    if plot_df.empty:
        raise ValueError(
            f"No rotated coordinates available for "
            f"{sample_name}"
        )

    figure, axis = plt.subplots(
        figsize=(12, 8)
    )

    axis.scatter(
        plot_df["rotated_w"],
        plot_df["rotated_h"],
        s=6,
        alpha=0.35,
    )

    landmark_columns = [
        (
            "left_granulation_tissue_range",
            "Left granulation boundary",
            ":",
        ),
        (
            "left_wound_edge",
            "Left wound edge",
            "--",
        ),
        (
            "wound_center",
            "Wound center",
            "-",
        ),
        (
            "right_wound_edge",
            "Right wound edge",
            "--",
        ),
        (
            "right_granulation_tissue_range",
            "Right granulation boundary",
            ":",
        ),
    ]

    for column, label, line_style in landmark_columns:
        values = (
            plot_df[column]
            .dropna()
            .unique()
        )

        if len(values) == 1:
            value = float(
                values[0]
            )

            axis.axvline(
                value,
                linestyle=line_style,
                linewidth=2,
                label=f"{label}: {value:.1f}",
            )

    axis.set_title(
        "Rotated tile coordinates and wound landmarks\n"
        f"{sample_name}"
    )

    axis.set_xlabel(
        "rotated_w"
    )

    axis.set_ylabel(
        "rotated_h"
    )

    axis.legend(
        loc="best"
    )

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    output_path = (
        output_dir
        / (
            f"{safe_filename(sample_name)}"
            "_rotated_coordinate_check.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def plot_normalization_check(
    sample_df: pd.DataFrame,
    sample_name: str,
    output_dir: Path,
) -> Path:
    """
    Show the full spatial tile layout and color only the wound interior.

    Every tile with valid rotated_w and rotated_h coordinates is plotted
    first in light gray. Tiles inside the normalized interval [-1, +1]
    are then overlaid with the wound-position color scale.

    This guarantees that all available rotated tile coordinates remain
    visible, including tissue outside the wound and tiles whose wound
    normalization is unavailable.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_sample_rows = len(sample_df)

    valid_coordinate_mask = (
        sample_df["rotated_w"].notna()
        & sample_df["rotated_h"].notna()
    )

    plot_df = sample_df.loc[
        valid_coordinate_mask
    ].copy()

    missing_coordinate_count = int(
        (~valid_coordinate_mask).sum()
    )

    if plot_df.empty:
        raise ValueError(
            f"No rotated coordinates available for {sample_name}"
        )

    figure, axis = plt.subplots(
        figsize=(14, 9)
    )

    # --------------------------------------------------------
    # PLOT EVERY AVAILABLE TILE COORDINATE AS A GRAY BASE LAYER
    # --------------------------------------------------------

    axis.scatter(
        plot_df["rotated_w"],
        plot_df["rotated_h"],
        s=10,
        alpha=0.40,
        color="lightgray",
        label="All tiles outside wound or normalization unavailable",
        zorder=1,
    )

    # --------------------------------------------------------
    # OVERLAY ONLY THE WOUND INTERIOR WITH NORMALIZED COLORS
    # --------------------------------------------------------

    normalized_position = plot_df[
        "normalized_wound_and_granulation_position"
    ]

    wound_interior = normalized_position.between(
        -2,
        2,
        inclusive="both",
    )

    colored_count = int(
        wound_interior.sum()
    )

    if wound_interior.any():
        normalized_values = plot_df.loc[
            wound_interior,
            "normalized_wound_position",
        ]

        scatter = axis.scatter(
            plot_df.loc[
                wound_interior,
                "rotated_w",
            ],
            plot_df.loc[
                wound_interior,
                "rotated_h",
            ],
            c=normalized_values,
            cmap="coolwarm",
            vmin=-2,
            vmax=2,
            s=10,
            alpha=0.80,
            zorder=2,
        )

        colorbar = figure.colorbar(
            scatter,
            ax=axis,
            pad=0.02,
        )

        colorbar.set_label(
            "Normalized wound and granulation position",
            fontsize=12,
        )

        colorbar.set_ticks(
            [-2, -1, 0, 1, 2]
        )

        colorbar.set_ticklabels(
            [
                "Left granulation boundary (-2)",
                "Left wound edge (-1)",
                "Center (0)",
                "Right wound edge (+1)",
                "Right granulation boundary (+2)",
            ]
        )

    # --------------------------------------------------------
    # EXTRACT AND DRAW LANDMARKS
    # --------------------------------------------------------

    landmark_styles = [
        (
            "left_granulation_tissue_range",
            "Left granulation boundary",
            ":",
            1.5,
        ),
        (
            "left_wound_edge",
            "Left wound edge = -1",
            "--",
            2.5,
        ),
        (
            "wound_center",
            "Wound center = 0",
            "-",
            3.0,
        ),
        (
            "right_wound_edge",
            "Right wound edge = +1",
            "--",
            2.5,
        ),
        (
            "right_granulation_tissue_range",
            "Right granulation boundary",
            ":",
            1.5,
        ),
    ]

    for column, label, line_style, line_width in landmark_styles:
        values = (
            plot_df[column]
            .dropna()
            .unique()
        )

        if len(values) == 1:
            value = float(values[0])

            axis.axvline(
                value,
                linestyle=line_style,
                linewidth=line_width,
                label=f"{label}: {value:.1f}",
                zorder=3,
            )

    # --------------------------------------------------------
    # SHADE THE AVAILABLE WOUND HALVES
    # --------------------------------------------------------

    left_values = (
        plot_df["left_wound_edge"]
        .dropna()
        .unique()
    )

    center_values = (
        plot_df["wound_center"]
        .dropna()
        .unique()
    )

    right_values = (
        plot_df["right_wound_edge"]
        .dropna()
        .unique()
    )

    left_edge = (
        float(left_values[0])
        if len(left_values) == 1
        else None
    )

    wound_center = (
        float(center_values[0])
        if len(center_values) == 1
        else None
    )

    right_edge = (
        float(right_values[0])
        if len(right_values) == 1
        else None
    )

    if left_edge is not None and wound_center is not None:
        axis.axvspan(
            left_edge,
            wound_center,
            alpha=0.06,
            label="Left wound half: -1 to 0",
            zorder=0,
        )

    if wound_center is not None and right_edge is not None:
        axis.axvspan(
            wound_center,
            right_edge,
            alpha=0.06,
            label="Right wound half: 0 to +1",
            zorder=0,
        )

    # --------------------------------------------------------
    # FORCE LIMITS TO INCLUDE ALL AVAILABLE TILE COORDINATES
    # --------------------------------------------------------

    x_min = float(plot_df["rotated_w"].min())
    x_max = float(plot_df["rotated_w"].max())
    y_min = float(plot_df["rotated_h"].min())
    y_max = float(plot_df["rotated_h"].max())

    x_padding = max((x_max - x_min) * 0.03, 1.0)
    y_padding = max((y_max - y_min) * 0.03, 1.0)

    axis.set_xlim(
        x_min - x_padding,
        x_max + x_padding,
    )

    axis.set_ylim(
        y_min - y_padding,
        y_max + y_padding,
    )

    # Preserve the geometry of the rotated tissue.
    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    # --------------------------------------------------------
    # TITLE AND LABELS
    # --------------------------------------------------------

    normalization_status_values = (
        plot_df["normalization_status"]
        .dropna()
        .unique()
    )

    if len(normalization_status_values) == 1:
        normalization_status = normalization_status_values[0]
    else:
        normalization_status = "unknown"

    plotted_coordinate_count = len(plot_df)
    gray_count = plotted_coordinate_count - colored_count

    axis.set_title(
        "Spatial wound normalization check\n"
        f"{sample_name}\n"
        f"Status: {normalization_status} | "
        f"Total rows: {total_sample_rows} | "
        f"Plotted coordinates: {plotted_coordinate_count} | "
        f"Missing coordinates: {missing_coordinate_count}\n"
        f"Colored wound tiles: {colored_count} | "
        f"Gray tiles: {gray_count}",
        fontsize=14,
    )

    biological_tick_positions = []
    biological_tick_labels = []

    if left_edge is not None:
        biological_tick_positions.append(left_edge)
        biological_tick_labels.append("-1")

    if wound_center is not None:
        biological_tick_positions.append(wound_center)
        biological_tick_labels.append("0")

    if right_edge is not None:
        biological_tick_positions.append(right_edge)
        biological_tick_labels.append("+1")

    if biological_tick_positions:
        axis.set_xticks(
            biological_tick_positions
        )

        axis.set_xticklabels(
            biological_tick_labels
        )

    axis.set_xlabel(
        "Normalized wound position",
        fontsize=12,
    )

    axis.set_ylabel(
        "rotated_h",
        fontsize=12,
    )

    axis.grid(
        alpha=0.2
    )

    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=9,
    )

    figure.tight_layout()

    output_path = (
        output_dir
        / (
            f"{safe_filename(sample_name)}"
            "_spatial_normalization_check.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


# ============================================================
# LOAD DATA
# ============================================================

tiles = clean_columns(
    pd.read_csv(
        MAIN_CSV
    )
)

wounds = clean_columns(
    pd.read_csv(
        WOUND_CSV,
        na_values=[
            "Na",
            "NA",
            "N/A",
            "na",
            "n/a",
            "",
            " ",
        ],
        keep_default_na=True,
    )
)


# ============================================================
# REQUIRED-COLUMN CHECKS
# ============================================================

required_tile_columns = {
    "sample_name",
    "w",
    "h",
    "rotated_w",
    "rotated_h",
    "superclusters",
}

required_wound_columns = {
    "sample_name",
    "rotation",
    "left_granulation_tissue_range",
    "left_wound_edge",
    "wound_center",
    "right_wound_edge",
    "right_granulation_tissue_range",
    "best_edge",
}

missing_tile_columns = (
    required_tile_columns
    - set(tiles.columns)
)

missing_wound_columns = (
    required_wound_columns
    - set(wounds.columns)
)

if missing_tile_columns:
    raise ValueError(
        "Tile CSV is missing required columns: "
        f"{sorted(missing_tile_columns)}"
    )

if missing_wound_columns:
    raise ValueError(
        "Wound CSV is missing required columns: "
        f"{sorted(missing_wound_columns)}"
    )


# ============================================================
# CLEAN VALUES
# ============================================================

tiles["sample_name"] = clean_sample_names(
    tiles["sample_name"]
)

wounds["sample_name"] = clean_sample_names(
    wounds["sample_name"]
)

tiles = convert_numeric_columns(
    tiles,
    [
        "w",
        "h",
        "rotated_w",
        "rotated_h",
    ],
)

# Exact supercluster annotation used for granulation filtering.
tiles["is_granulation_tissue"] = (
    tiles["superclusters"]
    .astype("string")
    .str.strip()
    .str.casefold()
    .eq("granulation tissue")
)

wounds = convert_numeric_columns(
    wounds,
    [
        "rotation",
        "left_granulation_tissue_range",
        "left_wound_edge",
        "wound_center",
        "right_wound_edge",
        "right_granulation_tissue_range",
    ],
)

wounds["best_edge"] = (
    wounds["best_edge"]
    .astype("string")
    .str.strip()
    .str.title()
)


# ============================================================
# DUPLICATE WOUND-ROW CHECK
# ============================================================

duplicate_wound_samples = (
    wounds.loc[
        wounds["sample_name"].duplicated(
            keep=False
        ),
        "sample_name",
    ]
    .dropna()
    .unique()
)

if len(duplicate_wound_samples) > 0:
    duplicate_text = "\n".join(
        f"  - {sample}"
        for sample in duplicate_wound_samples
    )

    raise ValueError(
        "The wound table contains multiple rows for "
        "the following samples:\n"
        f"{duplicate_text}"
    )


# ============================================================
# VALIDATE WOUND LANDMARKS
# ============================================================

wound_validation = validate_wound_table(
    wounds
)

invalid_landmarks = wound_validation.loc[
    ~wound_validation[
        "all_available_landmarks_valid"
    ]
]

if not invalid_landmarks.empty:
    print(
        "\nWARNING: Invalid or incomplete "
        "wound-landmark rows:"
    )

    print(
        invalid_landmarks[
            [
                "sample_name",
                "left_granulation_tissue_range",
                "left_wound_edge",
                "wound_center",
                "right_wound_edge",
                "right_granulation_tissue_range",
                "left_side_landmarks_valid",
                "right_side_landmarks_valid",
                "left_granulation_landmarks_valid",
                "right_granulation_landmarks_valid",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAMPLE-NAME MATCHING CHECK
# ============================================================

tile_sample_names = set(
    tiles["sample_name"].dropna()
)

wound_sample_names = set(
    wounds["sample_name"].dropna()
)

samples_missing_wound_info = sorted(
    tile_sample_names
    - wound_sample_names
)

wound_samples_without_tiles = sorted(
    wound_sample_names
    - tile_sample_names
)

if samples_missing_wound_info:
    print(
        "\nWARNING: Tile samples with no wound row:"
    )

    for sample in samples_missing_wound_info:
        print(
            f"  - {sample}"
        )

if wound_samples_without_tiles:
    print(
        "\nWARNING: Wound rows with no tile data:"
    )

    for sample in wound_samples_without_tiles:
        print(
            f"  - {sample}"
        )


# ============================================================
# MERGE WOUND INFORMATION
# ============================================================

merged = tiles.merge(
    wounds,
    on="sample_name",
    how="left",
    validate="many_to_one",
    suffixes=(
        "",
        "_wound_table",
    ),
    indicator=True,
)

merged["wound_merge_status"] = (
    merged["_merge"]
    .map(
        {
            "both": "matched",
            "left_only": "missing_wound_row",
            "right_only": "unexpected_right_only",
        }
    )
    .astype("string")
)

merged = merged.drop(
    columns=[
        "_merge",
    ]
)


# ============================================================
# RAW DISTANCE FROM WOUND CENTER
# ============================================================

valid_center_distance = (
    merged["rotated_w"].notna()
    & merged["wound_center"].notna()
)

merged[
    "signed_raw_distance_from_wound_center"
] = np.nan

merged.loc[
    valid_center_distance,
    "signed_raw_distance_from_wound_center",
] = (
    merged.loc[
        valid_center_distance,
        "rotated_w",
    ]
    - merged.loc[
        valid_center_distance,
        "wound_center",
    ]
)

merged[
    "absolute_raw_distance_from_wound_center"
] = (
    merged[
        "signed_raw_distance_from_wound_center"
    ].abs()
)


# ============================================================
# NORMALIZE WOUND POSITION
# ============================================================

merged[
    "normalized_wound_position"
] = normalize_wound_position(
    rotated_w=merged["rotated_w"],
    left_edge=merged["left_wound_edge"],
    wound_center=merged["wound_center"],
    right_edge=merged["right_wound_edge"],
)

merged[
    "absolute_normalized_distance"
] = (
    merged[
        "normalized_wound_position"
    ].abs()
)

merged[
    "normalization_status"
] = assign_normalization_status(
    left_edge=merged["left_wound_edge"],
    wound_center=merged["wound_center"],
    right_edge=merged["right_wound_edge"],
)

merged[
    "wound_side"
] = assign_wound_side(
    rotated_w=merged["rotated_w"],
    wound_center=merged["wound_center"],
)

merged[
    "wound_region"
] = assign_wound_region(
    merged
)

merged[
    "normalized_wound_and_granulation_position"
] = normalize_wound_and_granulation_position(
    merged
)


# ============================================================
# BEST-EDGE DISTANCE
# ============================================================

merged[
    "best_edge_coordinate"
] = np.nan

left_best = (
    merged["best_edge"]
    .str.lower()
    .eq("left")
)

right_best = (
    merged["best_edge"]
    .str.lower()
    .eq("right")
)

merged.loc[
    left_best,
    "best_edge_coordinate",
] = merged.loc[
    left_best,
    "left_wound_edge",
]

merged.loc[
    right_best,
    "best_edge_coordinate",
] = merged.loc[
    right_best,
    "right_wound_edge",
]

merged[
    "signed_distance_from_best_edge"
] = (
    merged["rotated_w"]
    - merged["best_edge_coordinate"]
)

merged[
    "absolute_distance_from_best_edge"
] = (
    merged[
        "signed_distance_from_best_edge"
    ].abs()
)


# ============================================================
# ROW-LEVEL VALIDITY FLAGS
# ============================================================

merged[
    "has_valid_rotated_coordinates"
] = (
    merged["rotated_w"].notna()
    & merged["rotated_h"].notna()
)

merged[
    "has_wound_center"
] = (
    merged["wound_center"].notna()
)

merged[
    "has_normalized_value"
] = (
    merged[
        "normalized_wound_position"
    ].notna()
)


# ============================================================
# EXPLAIN WHY NORMALIZATION IS UNAVAILABLE
# ============================================================

merged[
    "normalization_unavailable_reason"
] = pd.Series(
    pd.NA,
    index=merged.index,
    dtype="string",
)

# Missing rotated coordinate
mask = (
    merged["rotated_w"].isna()
)

merged.loc[
    mask,
    "normalization_unavailable_reason",
] = "missing_rotated_w"

# Missing wound center
mask = (
    merged["rotated_w"].notna()
    & merged["wound_center"].isna()
)

merged.loc[
    mask,
    "normalization_unavailable_reason",
] = "missing_wound_center"

# Left side cannot normalize because left edge is missing
mask = (
    merged["rotated_w"].notna()
    & merged["wound_center"].notna()
    & merged["left_wound_edge"].isna()
    & (
        merged["rotated_w"]
        < merged["wound_center"]
    )
)

merged.loc[
    mask,
    "normalization_unavailable_reason",
] = "left_side_missing_left_wound_edge"

# Right side cannot normalize because right edge is missing
mask = (
    merged["rotated_w"].notna()
    & merged["wound_center"].notna()
    & merged["right_wound_edge"].isna()
    & (
        merged["rotated_w"]
        > merged["wound_center"]
    )
)

merged.loc[
    mask,
    "normalization_unavailable_reason",
] = "right_side_missing_right_wound_edge"

# Clear the reason when a value was successfully normalized
mask = (
    merged[
        "normalized_wound_position"
    ].notna()
)

merged.loc[
    mask,
    "normalization_unavailable_reason",
] = pd.NA


# ============================================================
# FORMULA LANDMARK CHECKS
# ============================================================

sample_landmark_checks = (
    merged[
        [
            "sample_name",
            "left_wound_edge",
            "wound_center",
            "right_wound_edge",
        ]
    ]
    .drop_duplicates(
        subset=[
            "sample_name",
        ]
    )
    .copy()
)

sample_landmark_checks[
    "left_edge_expected_normalized"
] = np.where(
    sample_landmark_checks[
        "left_wound_edge"
    ].notna()
    & sample_landmark_checks[
        "wound_center"
    ].notna()
    & (
        sample_landmark_checks[
            "wound_center"
        ]
        > sample_landmark_checks[
            "left_wound_edge"
        ]
    ),
    (
        sample_landmark_checks[
            "left_wound_edge"
        ]
        - sample_landmark_checks[
            "wound_center"
        ]
    )
    / (
        sample_landmark_checks[
            "wound_center"
        ]
        - sample_landmark_checks[
            "left_wound_edge"
        ]
    ),
    np.nan,
)

sample_landmark_checks[
    "center_expected_normalized"
] = np.where(
    sample_landmark_checks[
        "wound_center"
    ].notna(),
    0.0,
    np.nan,
)

sample_landmark_checks[
    "right_edge_expected_normalized"
] = np.where(
    sample_landmark_checks[
        "right_wound_edge"
    ].notna()
    & sample_landmark_checks[
        "wound_center"
    ].notna()
    & (
        sample_landmark_checks[
            "right_wound_edge"
        ]
        > sample_landmark_checks[
            "wound_center"
        ]
    ),
    (
        sample_landmark_checks[
            "right_wound_edge"
        ]
        - sample_landmark_checks[
            "wound_center"
        ]
    )
    / (
        sample_landmark_checks[
            "right_wound_edge"
        ]
        - sample_landmark_checks[
            "wound_center"
        ]
    ),
    np.nan,
)

sample_landmark_checks[
    "left_edge_maps_to_minus_one"
] = (
    sample_landmark_checks[
        "left_edge_expected_normalized"
    ].isna()
    | np.isclose(
        sample_landmark_checks[
            "left_edge_expected_normalized"
        ],
        -1.0,
    )
)

sample_landmark_checks[
    "center_maps_to_zero"
] = (
    sample_landmark_checks[
        "center_expected_normalized"
    ].isna()
    | np.isclose(
        sample_landmark_checks[
            "center_expected_normalized"
        ],
        0.0,
    )
)

sample_landmark_checks[
    "right_edge_maps_to_plus_one"
] = (
    sample_landmark_checks[
        "right_edge_expected_normalized"
    ].isna()
    | np.isclose(
        sample_landmark_checks[
            "right_edge_expected_normalized"
        ],
        1.0,
    )
)


# ============================================================
# PER-SAMPLE QC SUMMARY
# ============================================================

sample_qc = (
    merged.groupby(
        "sample_name",
        dropna=False,
    )
    .agg(
        total_rows=(
            "sample_name",
            "size",
        ),

        valid_rotated_rows=(
            "has_valid_rotated_coordinates",
            "sum",
        ),

        raw_distance_rows=(
            "signed_raw_distance_from_wound_center",
            lambda values: int(
                values.notna().sum()
            ),
        ),

        normalized_rows=(
            "normalized_wound_position",
            lambda values: int(
                values.notna().sum()
            ),
        ),

        unavailable_rows=(
            "normalized_wound_position",
            lambda values: int(
                values.isna().sum()
            ),
        ),

        left_tiles=(
            "wound_side",
            lambda values: int(
                values.eq("left").sum()
            ),
        ),

        center_tiles=(
            "wound_side",
            lambda values: int(
                values.eq("center").sum()
            ),
        ),

        right_tiles=(
            "wound_side",
            lambda values: int(
                values.eq("right").sum()
            ),
        ),

        minimum_rotated_w=(
            "rotated_w",
            "min",
        ),

        maximum_rotated_w=(
            "rotated_w",
            "max",
        ),

        minimum_normalized_position=(
            "normalized_wound_position",
            "min",
        ),

        maximum_normalized_position=(
            "normalized_wound_position",
            "max",
        ),

        left_wound_edge=(
            "left_wound_edge",
            "first",
        ),

        wound_center=(
            "wound_center",
            "first",
        ),

        right_wound_edge=(
            "right_wound_edge",
            "first",
        ),

        normalization_status=(
            "normalization_status",
            "first",
        ),

        merge_status=(
            "wound_merge_status",
            "first",
        ),
    )
    .reset_index()
)

sample_qc[
    "percent_normalized"
] = (
    100.0
    * sample_qc["normalized_rows"]
    / sample_qc["total_rows"]
)

sample_qc[
    "spans_wound_center"
] = (
    (
        sample_qc[
            "minimum_rotated_w"
        ]
        < sample_qc[
            "wound_center"
        ]
    )
    & (
        sample_qc[
            "maximum_rotated_w"
        ]
        > sample_qc[
            "wound_center"
        ]
    )
)

sample_qc[
    "spans_left_wound_edge"
] = (
    sample_qc[
        "left_wound_edge"
    ].isna()
    | (
        (
            sample_qc[
                "minimum_rotated_w"
            ]
            <= sample_qc[
                "left_wound_edge"
            ]
        )
        & (
            sample_qc[
                "maximum_rotated_w"
            ]
            >= sample_qc[
                "left_wound_edge"
            ]
        )
    )
)

sample_qc[
    "spans_right_wound_edge"
] = (
    sample_qc[
        "right_wound_edge"
    ].isna()
    | (
        (
            sample_qc[
                "minimum_rotated_w"
            ]
            <= sample_qc[
                "right_wound_edge"
            ]
        )
        & (
            sample_qc[
                "maximum_rotated_w"
            ]
            >= sample_qc[
                "right_wound_edge"
            ]
        )
    )
)


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

OUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

VISUAL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

merged.to_csv(
    OUT_CSV,
    index=False,
)

sample_qc_path = (
    VISUAL_OUTPUT_DIR
    / "wound_normalization_sample_qc.csv"
)

sample_qc.to_csv(
    sample_qc_path,
    index=False,
)

landmark_check_path = (
    VISUAL_OUTPUT_DIR
    / "wound_normalization_landmark_checks.csv"
)

sample_landmark_checks.to_csv(
    landmark_check_path,
    index=False,
)


# ============================================================
# PRINT OVERALL QC
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "OVERALL WOUND MERGE AND NORMALIZATION QC"
)

print(
    "=" * 110
)

total_rows = len(
    merged
)

matched_rows = int(
    merged[
        "wound_merge_status"
    ].eq(
        "matched"
    ).sum()
)

valid_rotated_rows = int(
    merged[
        "has_valid_rotated_coordinates"
    ].sum()
)

raw_distance_rows = int(
    merged[
        "signed_raw_distance_from_wound_center"
    ].notna().sum()
)

normalized_rows = int(
    merged[
        "normalized_wound_position"
    ].notna().sum()
)

unavailable_rows = int(
    merged[
        "normalized_wound_position"
    ].isna().sum()
)

print(
    f"Total rows: {total_rows}"
)

print(
    f"Matched to wound table: {matched_rows}"
)

print(
    "Rows with valid rotated coordinates: "
    f"{valid_rotated_rows}"
)

print(
    "Rows with raw wound-center distance: "
    f"{raw_distance_rows}"
)

print(
    "Rows with normalized values: "
    f"{normalized_rows}"
)

print(
    "Rows without normalized values: "
    f"{unavailable_rows}"
)


# ============================================================
# PRINT PER-SAMPLE QC
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "PER-SAMPLE NORMALIZATION QC"
)

print(
    "=" * 110
)

print(
    sample_qc[
        [
            "sample_name",
            "total_rows",
            "valid_rotated_rows",
            "raw_distance_rows",
            "normalized_rows",
            "unavailable_rows",
            "percent_normalized",
            "left_tiles",
            "right_tiles",
            "normalization_status",
            "spans_wound_center",
            "spans_left_wound_edge",
            "spans_right_wound_edge",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# PRINT NORMALIZATION-UNAVAILABLE REASONS
# ============================================================

unavailable_summary = (
    merged.loc[
        merged[
            "normalized_wound_position"
        ].isna()
    ]
    .groupby(
        [
            "sample_name",
            "normalization_unavailable_reason",
        ],
        dropna=False,
    )
    .size()
    .reset_index(
        name="rows"
    )
)

if not unavailable_summary.empty:
    print(
        "\n"
        + "=" * 110
    )

    print(
        "NORMALIZATION-UNAVAILABLE REASONS"
    )

    print(
        "=" * 110
    )

    print(
        unavailable_summary.to_string(
            index=False
        )
    )


# ============================================================
# VISUAL QC FOR EVERY SAMPLE
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "VISUAL QC FOR EVERY SAMPLE"
)

print(
    "=" * 110
)

visual_check_samples = sorted(
    merged["sample_name"]
    .dropna()
    .unique()
)

print(
    f"Generating visual QC for "
    f"{len(visual_check_samples)} samples."
)

for sample_name in visual_check_samples:
    sample_df = merged.loc[
        merged["sample_name"] == sample_name
    ].copy()

    sample_total_rows = len(
        sample_df
    )

    sample_valid_rotated = int(
        sample_df[
            "has_valid_rotated_coordinates"
        ].sum()
    )

    sample_normalized_rows = int(
        sample_df[
            "normalized_wound_position"
        ].notna().sum()
    )

    sample_unavailable_rows = int(
        sample_df[
            "normalized_wound_position"
        ].isna().sum()
    )

    sample_status_values = (
        sample_df[
            "normalization_status"
        ]
        .dropna()
        .unique()
    )

    if len(sample_status_values) == 1:
        sample_status = str(
            sample_status_values[0]
        )
    else:
        sample_status = "unknown"

    minimum_rotated_w = (
        sample_df[
            "rotated_w"
        ].min()
    )

    maximum_rotated_w = (
        sample_df[
            "rotated_w"
        ].max()
    )

    minimum_normalized = (
        sample_df[
            "normalized_wound_position"
        ].min()
    )

    maximum_normalized = (
        sample_df[
            "normalized_wound_position"
        ].max()
    )

    print(
        f"\nSample: {sample_name}"
    )

    print(
        f"  Rows: {sample_total_rows}"
    )

    print(
        "  Valid rotated coordinates: "
        f"{sample_valid_rotated}"
    )

    print(
        "  Normalized rows: "
        f"{sample_normalized_rows}"
    )

    print(
        "  Unavailable rows: "
        f"{sample_unavailable_rows}"
    )

    print(
        "  Normalization status: "
        f"{sample_status}"
    )

    if pd.notna(minimum_rotated_w) and pd.notna(maximum_rotated_w):
        print(
            "  rotated_w range: "
            f"{minimum_rotated_w:.2f} to "
            f"{maximum_rotated_w:.2f}"
        )
    else:
        print(
            "  rotated_w range: unavailable"
        )

    if pd.notna(minimum_normalized):
        print(
            "  Normalized range: "
            f"{minimum_normalized:.3f} to "
            f"{maximum_normalized:.3f}"
        )
    else:
        print(
            "  Normalized range: unavailable"
        )

    try:
        coordinate_plot = (
            plot_rotated_coordinate_check(
                sample_df=sample_df,
                sample_name=sample_name,
                output_dir=VISUAL_OUTPUT_DIR,
            )
        )

        print(
            f"  Coordinate plot: {coordinate_plot}"
        )

    except ValueError as error:
        print(
            "  [SKIPPED] Coordinate plot: "
            f"{error}"
        )

    try:
        spatial_plot = (
            plot_normalization_check(
                sample_df=sample_df,
                sample_name=sample_name,
                output_dir=VISUAL_OUTPUT_DIR,
            )
        )

        print(
            "  Spatial normalization plot: "
            f"{spatial_plot}"
        )

    except ValueError as error:
        print(
            "  [SKIPPED] Spatial normalization plot: "
            f"{error}"
        )


# ============================================================
# EXPLICIT KM-7 CHECK
# ============================================================

km7_name = (
    "20260320_KM-7_Scan3_bottom.ome"
)

km7 = merged.loc[
    merged["sample_name"] == km7_name
].copy()

if not km7.empty:
    print(
        "\n"
        + "=" * 110
    )

    print(
        "KM-7 EXPLICIT CHECK"
    )

    print(
        "=" * 110
    )

    km7_left_edge = (
        km7[
            "left_wound_edge"
        ].iloc[0]
    )

    km7_center = (
        km7[
            "wound_center"
        ].iloc[0]
    )

    km7_right_edge = (
        km7[
            "right_wound_edge"
        ].iloc[0]
    )

    missing_rotated_w = int(
        km7[
            "rotated_w"
        ].isna().sum()
    )

    missing_normalized = int(
        km7[
            "normalized_wound_position"
        ].isna().sum()
    )

    tiles_left_of_center = int(
        (
            km7[
                "rotated_w"
            ]
            < km7_center
        ).sum()
    )

    tiles_right_of_center = int(
        (
            km7[
                "rotated_w"
            ]
            > km7_center
        ).sum()
    )

    tiles_between_wound_edges = int(
        km7[
            "rotated_w"
        ].between(
            km7_left_edge,
            km7_right_edge,
            inclusive="both",
        ).sum()
    )

    tiles_at_or_below_minus_one = int(
        (
            km7[
                "normalized_wound_position"
            ]
            <= -1
        ).sum()
    )

    tiles_between_minus_one_and_one = int(
        km7[
            "normalized_wound_position"
        ].between(
            -1,
            1,
            inclusive="both",
        ).sum()
    )

    tiles_at_or_above_plus_one = int(
        (
            km7[
                "normalized_wound_position"
            ]
            >= 1
        ).sum()
    )

    print(
        f"Left wound edge: {km7_left_edge}"
    )

    print(
        f"Wound center: {km7_center}"
    )

    print(
        f"Right wound edge: {km7_right_edge}"
    )

    print(
        f"Missing rotated_w: {missing_rotated_w}"
    )

    print(
        "Missing normalized position: "
        f"{missing_normalized}"
    )

    print(
        "Tiles left of center: "
        f"{tiles_left_of_center}"
    )

    print(
        "Tiles right of center: "
        f"{tiles_right_of_center}"
    )

    print(
        "Tiles between wound edges: "
        f"{tiles_between_wound_edges}"
    )

    print(
        "Tiles at or below normalized -1: "
        f"{tiles_at_or_below_minus_one}"
    )

    print(
        "Tiles between normalized -1 and +1: "
        f"{tiles_between_minus_one_and_one}"
    )

    print(
        "Tiles at or above normalized +1: "
        f"{tiles_at_or_above_plus_one}"
    )

else:
    print(
        "\n[WARNING] KM-7 was not found: "
        f"{km7_name}"
    )


# ============================================================
# EXPLICIT MISSING-LEFT-EDGE CHECK
# ============================================================

missing_left_sample_name = (
    "20230721_Day 5 dbdb 4_dbdb Day 5 "
    "6_Scan6_Stitched.ome"
)

missing_left_sample = merged.loc[
    merged["sample_name"]
    == missing_left_sample_name
].copy()

if not missing_left_sample.empty:
    print(
        "\n"
        + "=" * 110
    )

    print(
        "MISSING-LEFT-EDGE SAMPLE CHECK"
    )

    print(
        "=" * 110
    )

    sample_center = (
        missing_left_sample[
            "wound_center"
        ].iloc[0]
    )

    sample_right_edge = (
        missing_left_sample[
            "right_wound_edge"
        ].iloc[0]
    )

    left_side_mask = (
        missing_left_sample[
            "rotated_w"
        ]
        < sample_center
    )

    right_side_mask = (
        missing_left_sample[
            "rotated_w"
        ]
        >= sample_center
    )

    left_side_rows = int(
        left_side_mask.sum()
    )

    right_side_rows = int(
        right_side_mask.sum()
    )

    left_side_normalized_rows = int(
        missing_left_sample.loc[
            left_side_mask,
            "normalized_wound_position",
        ].notna().sum()
    )

    right_side_normalized_rows = int(
        missing_left_sample.loc[
            right_side_mask,
            "normalized_wound_position",
        ].notna().sum()
    )

    right_edge_mapping = (
        (
            sample_right_edge
            - sample_center
        )
        / (
            sample_right_edge
            - sample_center
        )
    )

    print(
        f"Wound center: {sample_center}"
    )

    print(
        f"Right wound edge: {sample_right_edge}"
    )

    print(
        f"Left-side tile rows: {left_side_rows}"
    )

    print(
        f"Right-side tile rows: {right_side_rows}"
    )

    print(
        "Left-side normalized rows "
        "(expected 0): "
        f"{left_side_normalized_rows}"
    )

    print(
        "Right-side normalized rows: "
        f"{right_side_normalized_rows}"
    )

    print(
        "Right edge expected normalized value "
        "(expected 1): "
        f"{right_edge_mapping}"
    )


# ============================================================
# FINAL OUTPUT LOCATIONS
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 110
)

print(
    "Merged wound file:\n"
    f"  {OUT_CSV}"
)

print(
    "Sample QC table:\n"
    f"  {sample_qc_path}"
)

print(
    "Landmark check table:\n"
    f"  {landmark_check_path}"
)

print(
    "Visual QC directory:\n"
    f"  {VISUAL_OUTPUT_DIR}"
)

print(
    "\n[✓] Wound merge, normalization, and QC completed."
)
