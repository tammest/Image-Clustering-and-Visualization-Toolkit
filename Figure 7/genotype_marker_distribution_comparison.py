import os
import re
import numpy as np
import matplotlib.pyplot as plt

# regex to detect genotypes
GENO_RE = re.compile(r'(?<![A-Za-z0-9])(wt|dbdb)(?![A-Za-z0-9])', re.I)

# colors
COL_WT  = "#29352e"   # dark green/gray
COL_DBD = "#e73737"   # red

def plot_wt_vs_dbdb(sample_names, chA, chB, qA, qB, use_quantile=True, out_dir="outputs/genotype_comparison"):
    os.makedirs(out_dir, exist_ok=True)

    # classify sample names
    wt_samples  = [s for s in sample_names if GENO_RE.search(s) and "wt"   in GENO_RE.search(s).group(1).lower()]
    db_samples  = [s for s in sample_names if GENO_RE.search(s) and "dbdb" in GENO_RE.search(s).group(1).lower()]

    # collect pooled values
    vWT_A = _pooled_values(wt_samples, chA)
    vDB_A = _pooled_values(db_samples, chA)
    vWT_B = _pooled_values(wt_samples, chB)
    vDB_B = _pooled_values(db_samples, chB)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

    # --- CD31 ---
    ax = axes[0]
    if vWT_A.size and vDB_A.size:
        lo, hi = np.percentile(np.concatenate([vWT_A, vDB_A]), [1, 99])
        bins = np.linspace(lo, hi, 60)

        # step outlines
        ax.hist(vWT_A, bins=bins, color=COL_WT, label="WT", histtype="step", linewidth=1.5)
        ax.hist(vDB_A, bins=bins, color=COL_DBD, label="db/db", histtype="step", linewidth=1.5)

        # thresholds
        cut_wt = np.quantile(vWT_A, np.clip(qA, 0, 1)) if use_quantile else float(qA)
        cut_db = np.quantile(vDB_A, np.clip(qA, 0, 1)) if use_quantile else float(qA)
        ax.axvline(cut_wt, color=COL_WT, linestyle="--", linewidth=1.2)
        ax.axvline(cut_db, color=COL_DBD, linestyle="--", linewidth=1.2)

    ax.set_title("CD31 pooled by genotype")
    ax.set_xlabel("CD31 feature value")
    ax.set_ylabel("count")
    ax.legend()

    # --- F4/80 ---
    ax = axes[1]
    if vWT_B.size and vDB_B.size:
        lo, hi = np.percentile(np.concatenate([vWT_B, vDB_B]), [1, 99])
        bins = np.linspace(lo, hi, 60)

        # step outlines
        ax.hist(vWT_B, bins=bins, color=COL_WT, label="WT", histtype="step", linewidth=1.5)
        ax.hist(vDB_B, bins=bins, color=COL_DBD, label="db/db", histtype="step", linewidth=1.5)

        # thresholds
        cut_wt = np.quantile(vWT_B, np.clip(qB, 0, 1)) if use_quantile else float(qB)
        cut_db = np.quantile(vDB_B, np.clip(qB, 0, 1)) if use_quantile else float(qB)
        ax.axvline(cut_wt, color=COL_WT, linestyle="--", linewidth=1.2)
        ax.axvline(cut_db, color=COL_DBD, linestyle="--", linewidth=1.2)

    ax.set_title("F4/80 pooled by genotype")
    ax.set_xlabel("F4/80 feature value")
    ax.set_ylabel("count")
    ax.legend()

    fig.tight_layout()
    out_png = os.path.join(out_dir, "geno_comparison_step.png")
    out_pdf = os.path.join(out_dir, "geno_comparison_step.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_png}\n[OK] Saved: {out_pdf}")

if __name__ == "__main__":
    plot_wt_vs_dbdb(
        SAMPLES,
        chA=CH_CD31,
        chB=CH_F480,
        qA=0.95,
        qB=0.15,
        use_quantile=True,
        out_dir="outputs/genotype_comparison"
    )
