"""
Generate Publication-Grade Charts for SupeRR Selector S2 Executive Briefing.
Produces 4 figures in assets/charts/ (300 DPI, Rajasthan Royals navy/pink branding):
1. chart1_batters_ci.png: PRR 90% Bootstrap CI & Strike Rate Acceleration Dumbbell.
2. chart2_bowlers_ci.png: PRR 90% Bootstrap CI & Par Economy Suppression Dumbbell.
3. chart3_regime_and_ablation.png: Live Win-Swing Multipliers (2x2x2) & 8-Way Ablation Overlap.
4. chart4_season_heatmap.png: Multi-Year Season-by-Season Net Value Heatmaps (2022-2026).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Set styling and paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "assets", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Colors
RR_NAVY = "#070E22"
RR_PINK = "#EA1A85"
RR_CYAN = "#00E5FF"
RR_SLATE = "#64748B"
RR_LIGHT_GRAY = "#E2E8F0"
RR_MUTED_BG = "#F8FAFC"
RR_GOLD = "#F59E0B"
RR_GREEN = "#10B981"
RR_RED = "#EF4444"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = RR_LIGHT_GRAY
plt.rcParams["axes.linewidth"] = 0.8


def generate_chart1_batters():
    """Chart 1: Batters PRR 90% CI & Base vs HP Strike Rate Dumbbell."""
    print("Generating chart1_batters_ci.png...")
    bat_path = os.path.join(PROCESSED_DIR, "top_batters_under_pressure.csv")
    df = pd.read_csv(bat_path).head(10).iloc[::-1].reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(df))
    names = df["player_name"].values

    # --- Left Panel: PRR Score & 90% CI ---
    ax1.set_facecolor(RR_MUTED_BG)
    for i, r in df.iterrows():
        low = r["PRR_CI_90_Low"]
        high = r["PRR_CI_90_High"]
        val = r["PRR_Score"]
        p_top10 = r["P_Top_10_Pct"]

        ax1.plot([low, high], [i, i], color=RR_NAVY, lw=2.5, solid_capstyle="round", zorder=3)
        ax1.scatter(val, i, color=RR_PINK, s=65, edgecolors=RR_NAVY, lw=1.2, zorder=4)
        ax1.text(high + 0.8, i, f"P(Top10): {p_top10:.1f}%", va="center", ha="left", fontsize=7.5, color=RR_NAVY, fontweight="600")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8.5, fontweight="600", color=RR_NAVY)
    ax1.set_xlim(65, 115)
    ax1.set_xlabel("PRR Score (90% Cluster Bootstrap CI)", fontsize=8.5, fontweight="600", color=RR_NAVY)
    ax1.set_title("A. Batters PRR Score & 90% Bootstrap CI", fontsize=9.5, fontweight="700", color=RR_NAVY, pad=8)
    ax1.grid(axis="x", color=RR_LIGHT_GRAY, linestyle="--", lw=0.6, zorder=1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Right Panel: Dumbbell (Base SR vs HP SR) ---
    ax2.set_facecolor(RR_MUTED_BG)
    for i, r in df.iterrows():
        base_sr = r["Base_SR"]
        hp_sr = r["HP_SR"]
        delta = hp_sr - base_sr

        ax2.plot([base_sr, hp_sr], [i, i], color=RR_SLATE, lw=2.0, alpha=0.7, zorder=3)
        ax2.scatter(base_sr, i, color=RR_SLATE, s=55, zorder=4, label="Baseline SR" if i == 0 else "")
        ax2.scatter(hp_sr, i, color=RR_PINK, s=65, edgecolors=RR_NAVY, lw=1.2, zorder=5, label="High-Pressure SR" if i == 0 else "")
        ax2.text(hp_sr + 1.2, i, f"+{delta:.1f}", va="center", ha="left", fontsize=7.5, color=RR_PINK, fontweight="700")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlim(130, 195)
    ax2.set_xlabel("Strike Rate (Runs per 100 Balls)", fontsize=8.5, fontweight="600", color=RR_NAVY)
    ax2.set_title("B. Clutch Strike Rate Acceleration (Base vs. HP)", fontsize=9.5, fontweight="700", color=RR_NAVY, pad=8)
    ax2.grid(axis="x", color=RR_LIGHT_GRAY, linestyle="--", lw=0.6, zorder=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", fontsize=8, frameon=True, facecolor="#FFFFFF", edgecolor=RR_LIGHT_GRAY)

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "chart1_batters_ci.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def generate_chart2_bowlers():
    """Chart 2: Bowlers PRR 90% CI & Expected vs HP Economy Dumbbell."""
    print("Generating chart2_bowlers_ci.png...")
    bowl_path = os.path.join(PROCESSED_DIR, "top_bowlers_under_pressure.csv")
    df = pd.read_csv(bowl_path).head(10).iloc[::-1].reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(df))
    names = df["player_name"].values

    # --- Left Panel: PRR Score & 90% CI ---
    ax1.set_facecolor(RR_MUTED_BG)
    for i, r in df.iterrows():
        low = r["PRR_CI_90_Low"]
        high = r["PRR_CI_90_High"]
        val = r["PRR_Score"]
        p_top10 = r["P_Top_10_Pct"]

        ax1.plot([low, high], [i, i], color=RR_NAVY, lw=2.5, solid_capstyle="round", zorder=3)
        ax1.scatter(val, i, color=RR_PINK, s=65, edgecolors=RR_NAVY, lw=1.2, zorder=4)
        ax1.text(high + 0.8, i, f"P(Top10): {p_top10:.1f}%", va="center", ha="left", fontsize=7.5, color=RR_NAVY, fontweight="600")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8.5, fontweight="600", color=RR_NAVY)
    ax1.set_xlim(30, 115)
    ax1.set_xlabel("PRR Score (90% Cluster Bootstrap CI)", fontsize=8.5, fontweight="600", color=RR_NAVY)
    ax1.set_title("A. Bowlers PRR Score & 90% Bootstrap CI", fontsize=9.5, fontweight="700", color=RR_NAVY, pad=8)
    ax1.grid(axis="x", color=RR_LIGHT_GRAY, linestyle="--", lw=0.6, zorder=1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Right Panel: Dumbbell (Expected HP Econ vs Actual HP Econ) ---
    ax2.set_facecolor(RR_MUTED_BG)
    for i, r in df.iterrows():
        exp_econ = r["Expected_Econ_HP"]
        hp_econ = r["HP_Economy"]
        suppression = hp_econ - exp_econ  # negative means lower/better

        ax2.plot([hp_econ, exp_econ], [i, i], color=RR_SLATE, lw=2.0, alpha=0.7, zorder=3)
        ax2.scatter(exp_econ, i, color=RR_SLATE, s=55, zorder=4, label="Expected HP Par Econ" if i == 0 else "")
        ax2.scatter(hp_econ, i, color=RR_PINK, s=65, edgecolors=RR_NAVY, lw=1.2, zorder=5, label="Actual HP Economy" if i == 0 else "")
        ax2.text(hp_econ - 0.12, i, f"{suppression:.2f}", va="center", ha="right", fontsize=7.5, color=RR_PINK, fontweight="700")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlim(6.2, 10.2)
    ax2.set_xlabel("Economy Rate (Runs per Over)", fontsize=8.5, fontweight="600", color=RR_NAVY)
    ax2.set_title("B. High-Pressure Par Suppression (Actual vs. Par)", fontsize=9.5, fontweight="700", color=RR_NAVY, pad=8)
    ax2.grid(axis="x", color=RR_LIGHT_GRAY, linestyle="--", lw=0.6, zorder=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", fontsize=8, frameon=True, facecolor="#FFFFFF", edgecolor=RR_LIGHT_GRAY)

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "chart2_bowlers_ci.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def generate_chart3_regime_and_ablation():
    """Chart 3: Win-Swing Multipliers (2x2x2) & 8-Way Ablation Stability."""
    print("Generating chart3_regime_and_ablation.png...")
    val_path = os.path.join(PROCESSED_DIR, "validation_and_sensitivity.json")
    with open(val_path, "r") as f:
        val_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 2.9), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # --- Left Panel: Live Win-Probability Swing Multipliers ---
    ax1.set_facecolor(RR_MUTED_BG)
    categories = ["1st Innings (Setting Par)", "2nd Innings (Chase Target)"]
    ordinary_vals = [1.47, 0.78]
    hp_vals = [4.50, 6.52]
    multipliers = ["3.06x", "8.36x"]

    x = np.arange(len(categories))
    width = 0.32

    rects1 = ax1.bar(x - width/2, ordinary_vals, width, label="Ordinary Deliveries (L < L*)", color=RR_SLATE, alpha=0.85, zorder=3)
    rects2 = ax1.bar(x + width/2, hp_vals, width, label="High-Pressure Crunch (L >= L*)", color=RR_PINK, edgecolor=RR_NAVY, lw=1.0, zorder=3)

    for i in range(len(categories)):
        # Ordinary value text
        ax1.text(x[i] - width/2, ordinary_vals[i] + 0.15, f"{ordinary_vals[i]:.2f} pts", ha="center", va="bottom", fontsize=8, color=RR_SLATE, fontweight="600")
        # HP value text
        ax1.text(x[i] + width/2, hp_vals[i] + 0.15, f"{hp_vals[i]:.2f} pts", ha="center", va="bottom", fontsize=8, color=RR_NAVY, fontweight="700")
        # Callout badge for multiplier
        ax1.annotate(f"{multipliers[i]} SWING",
                     xy=(x[i] + width/2, hp_vals[i] / 2),
                     xytext=(x[i] + width/2, hp_vals[i] / 2),
                     ha="center", va="center", fontsize=8.5, fontweight="800", color="#FFFFFF",
                     bbox=dict(boxstyle="round,pad=0.28", fc=RR_NAVY, ec="none", alpha=0.92))

    ax1.set_ylabel("Mean Absolute Win Prob Swing (|ΔWP| in pts)", fontsize=8.5, fontweight="600", color=RR_NAVY)
    ax1.set_title("A. Live Win Probability Swing Multipliers", fontsize=9.5, fontweight="700", color=RR_NAVY, pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=8.5, fontweight="600", color=RR_NAVY)
    ax1.set_ylim(0, 7.8)
    ax1.grid(axis="y", color=RR_LIGHT_GRAY, linestyle="--", lw=0.6, zorder=1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=8, frameon=True, facecolor="#FFFFFF", edgecolor=RR_LIGHT_GRAY)

    # --- Right Panel: 8-Way Ablation Stability Overlap ---
    ax2.set_facecolor(RR_MUTED_BG)
    ablations = val_data.get("methodological_ablation_and_benchmarks", [])
    checks = []
    bat_overlap = []
    bowl_overlap = []

    for a in ablations:
        label = a["ablation_check"]
        # Shorten label for clean chart presentation
        label = label.replace("Exclude Associate / Emerging Matches", "Exclude Associate Matches")
        label = label.replace("Switch Off Bipartite Ridge Adjustment (theta=0)", "No Ridge Matchup Adj (θ=0)")
        label = label.replace("Naive Death-Overs Phase Rule (Overs 16-20 only)", "Naive Death Overs (Overs 16-20)")
        label = label.replace("Raw Traditional Stats Baseline (Unadjusted HP SR & Economy)", "Raw Traditional Stats Baseline")
        label = label.replace("Official GMM Crossover (Baseline)", "Official Baseline (L* = 0.9927)")
        checks.append(label)
        bat_overlap.append(int(a["batters"]["top10_overlap"].split("/")[0]))
        bowl_overlap.append(int(a["bowlers"]["top10_overlap"].split("/")[0]))

    # Reverse to have Baseline on top
    checks = checks[::-1]
    bat_overlap = bat_overlap[::-1]
    bowl_overlap = bowl_overlap[::-1]

    y_pos = np.arange(len(checks))
    bar_height = 0.36

    ax2.barh(y_pos + bar_height/2, bat_overlap, bar_height, label="Batters Top 10 Overlap", color=RR_NAVY, zorder=3)
    ax2.barh(y_pos - bar_height/2, bowl_overlap, bar_height, label="Bowlers Top 10 Overlap", color=RR_PINK, zorder=3)

    for i in range(len(checks)):
        ax2.text(bat_overlap[i] + 0.25, i + bar_height/2, f"{bat_overlap[i]}/10", va="center", ha="left", fontsize=7.5, color=RR_NAVY, fontweight="700")
        ax2.text(bowl_overlap[i] + 0.25, i - bar_height/2, f"{bowl_overlap[i]}/10", va="center", ha="left", fontsize=7.5, color=RR_PINK, fontweight="700")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(checks, fontsize=8.0, fontweight="600", color=RR_NAVY)
    ax2.set_xlim(0, 11.5)
    ax2.set_xlabel("Top 10 Roster Overlap Count (out of 10)", fontsize=8.5, fontweight="600", color=RR_NAVY)
    ax2.set_title("B. 8-Way Ablation Roster Invariance", fontsize=9.5, fontweight="700", color=RR_NAVY, pad=8)
    ax2.grid(axis="x", color=RR_LIGHT_GRAY, linestyle="--", lw=0.6, zorder=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", fontsize=8, frameon=True, facecolor="#FFFFFF", edgecolor=RR_LIGHT_GRAY)

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "chart3_regime_and_ablation.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def generate_chart4_season_heatmap():
    """Chart 4: 2022-2026 Season-by-Season High-Pressure Net Value Heatmap."""
    print("Generating chart4_season_heatmap.png...")
    val_path = os.path.join(PROCESSED_DIR, "validation_and_sensitivity.json")
    with open(val_path, "r") as f:
        val_data = json.load(f)

    annual_data = val_data.get("season_by_season_consistency_and_lift", {}).get("annual_net_value_matrix", {})
    batters_list = annual_data.get("batters", [])
    bowlers_list = annual_data.get("bowlers", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.3), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    years = ["2022", "2023", "2024", "2025", "2026"]

    # Helper function to plot heatmap
    def plot_matrix(ax, data_list, title, is_bowler=False):
        ax.set_facecolor(RR_MUTED_BG)
        players = [d["Player"] for d in data_list]
        mat = np.zeros((len(data_list), len(years)))
        mask = np.zeros_like(mat, dtype=bool)

        for i, d in enumerate(data_list):
            for j, y in enumerate(years):
                val = d.get(y)
                if val is None:
                    mask[i, j] = True
                    mat[i, j] = 0.0
                else:
                    mat[i, j] = val

        # Diverging norm centered at 0
        vbound = 70.0
        norm = mcolors.TwoSlopeNorm(vmin=-vbound, vcenter=0.0, vmax=vbound)
        cmap = plt.cm.coolwarm

        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        for i in range(len(data_list)):
            for j in range(len(years)):
                if mask[i, j]:
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=7.5, color=RR_SLATE, fontweight="600")
                else:
                    val = mat[i, j]
                    # Choose white or black text based on intensity
                    txt_color = "#FFFFFF" if abs(val) > 35 else RR_NAVY
                    sign = "+" if val > 0 else ""
                    ax.text(j, i, f"{sign}{val:.1f}", ha="center", va="center", fontsize=7.5, color=txt_color, fontweight="700")

        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels(years, fontsize=8.5, fontweight="600", color=RR_NAVY)
        ax.set_yticks(np.arange(len(players)))
        ax.set_yticklabels([f"#{i+1} {p}" for i, p in enumerate(players)], fontsize=8.0, fontweight="600", color=RR_NAVY)
        ax.set_title(title, fontsize=9.0, fontweight="700", color=RR_NAVY, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plot_matrix(ax1, batters_list, "A. Top 10 Batters High-Pressure Net Value (Runs / 100b)")
    plot_matrix(ax2, bowlers_list, "B. Top 10 Bowlers High-Pressure Net Value (Runs / 100b)", is_bowler=True)

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "chart4_season_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    print("=" * 60)
    print("GENERATING PUBLICATION-GRADE CHARTS FOR EXECUTIVE BRIEFING")
    print("=" * 60)
    generate_chart1_batters()
    generate_chart2_bowlers()
    generate_chart3_regime_and_ablation()
    generate_chart4_season_heatmap()
    print("=" * 60)
    print("ALL 4 PUBLICATION CHARTS GENERATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
