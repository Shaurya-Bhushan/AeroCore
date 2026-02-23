from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _plot_aircraft_views(aircraft_mc: List[Dict[str, Any]], output_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    solved_times = [float(r["time_s"]) for r in aircraft_mc if bool(r["solved"]) and np.isfinite(float(r["time_s"]))]
    ax1.hist(solved_times, bins=15, color="tab:blue", alpha=0.8)
    ax1.set_xlabel("Mission Time [s]")
    ax1.set_ylabel("Count")
    ax1.set_title("Aircraft Monte Carlo Mission Time")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    path1 = output_dir / "aircraft_monte_carlo_hist.png"
    fig1.savefig(path1, dpi=180)
    plt.close(fig1)
    paths["aircraft_mc_hist"] = str(path1)

    sorted_air = sorted(aircraft_mc, key=lambda r: int(r.get("run", 0)))
    fig1s, ax1s = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    run_idx = [int(r.get("run", i)) for i, r in enumerate(sorted_air)]
    t_vals = [float(r.get("time_s", np.nan)) for r in sorted_air]
    e_vals = [float(r.get("energy_used_wh", np.nan)) for r in sorted_air]
    ax1s[0].plot(run_idx, t_vals, marker="o", color="tab:blue")
    ax1s[0].set_ylabel("Mission Time [s]")
    ax1s[0].set_title("Aircraft Time vs Scenario")
    ax1s[0].grid(alpha=0.3)
    ax1s[1].plot(run_idx, e_vals, marker="o", color="tab:orange")
    ax1s[1].set_ylabel("Energy Used [Wh]")
    ax1s[1].set_xlabel("Scenario Index")
    ax1s[1].set_title("Aircraft Energy vs Scenario")
    ax1s[1].grid(alpha=0.3)
    fig1s.tight_layout()
    path1s = output_dir / "aircraft_scenario_sensitivity.png"
    fig1s.savefig(path1s, dpi=180)
    plt.close(fig1s)
    paths["aircraft_scenario_sensitivity"] = str(path1s)

    fig1b, ax1b = plt.subplots(1, 2, figsize=(12, 4))
    beam_t = [float(r["beam_time_s"]) for r in aircraft_mc if np.isfinite(float(r["beam_time_s"]))]
    hyb_t = [float(r["hybrid_time_s"]) for r in aircraft_mc if np.isfinite(float(r["hybrid_time_s"]))]
    if beam_t and hyb_t:
        ax1b[0].boxplot([beam_t, hyb_t], labels=["beam", "hybrid"])
    else:
        ax1b[0].text(0.5, 0.5, "insufficient hybrid samples", ha="center", va="center", transform=ax1b[0].transAxes)
    ax1b[0].set_title("Aircraft Time Variance")
    ax1b[0].set_ylabel("Mission Time [s]")
    ax1b[0].grid(alpha=0.3)

    beam_e = [float(r["beam_energy_used_wh"]) for r in aircraft_mc if np.isfinite(float(r["beam_energy_used_wh"]))]
    hyb_e = [float(r["hybrid_energy_used_wh"]) for r in aircraft_mc if np.isfinite(float(r["hybrid_energy_used_wh"]))]
    if beam_e and hyb_e:
        ax1b[1].boxplot([beam_e, hyb_e], labels=["beam", "hybrid"])
    else:
        ax1b[1].text(0.5, 0.5, "insufficient hybrid samples", ha="center", va="center", transform=ax1b[1].transAxes)
    ax1b[1].set_title("Aircraft Energy Variance")
    ax1b[1].set_ylabel("Energy Used [Wh]")
    ax1b[1].grid(alpha=0.3)
    fig1b.tight_layout()
    path1b = output_dir / "aircraft_hybrid_vs_beam_variance.png"
    fig1b.savefig(path1b, dpi=180)
    plt.close(fig1b)
    paths["aircraft_hybrid_vs_beam_variance"] = str(path1b)

    return paths


def _plot_spacecraft_views(spacecraft_mc: List[Dict[str, Any]], output_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = [float(r["battery_initial_wh"]) for r in spacecraft_mc]
    y = [float(r["delivered_science"]) for r in spacecraft_mc]
    colors = ["tab:green" if bool(r["solved"]) else "tab:red" for r in spacecraft_mc]
    ax2.scatter(x, y, c=colors, alpha=0.75)
    ax2.set_xlabel("Initial Battery [Wh]")
    ax2.set_ylabel("Delivered Science")
    ax2.set_title("Spacecraft Monte Carlo Robustness")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    path2 = output_dir / "spacecraft_monte_carlo_scatter.png"
    fig2.savefig(path2, dpi=180)
    plt.close(fig2)
    paths["spacecraft_mc_scatter"] = str(path2)

    fig2b, ax2b = plt.subplots(figsize=(9, 5))
    beam_science = [float(r["beam_delivered_science"]) for r in spacecraft_mc if np.isfinite(float(r["beam_delivered_science"]))]
    hyb_science = [float(r["hybrid_delivered_science"]) for r in spacecraft_mc if np.isfinite(float(r["hybrid_delivered_science"]))]
    if beam_science and hyb_science:
        ax2b.boxplot([beam_science, hyb_science], labels=["beam", "hybrid"])
    else:
        ax2b.text(0.5, 0.5, "insufficient hybrid samples", ha="center", va="center", transform=ax2b.transAxes)
    ax2b.set_title("Spacecraft Delivered Science Under ±Power/Timing")
    ax2b.set_ylabel("Delivered Science")
    ax2b.grid(alpha=0.3)
    fig2b.tight_layout()
    path2b = output_dir / "spacecraft_hybrid_vs_beam_science.png"
    fig2b.savefig(path2b, dpi=180)
    plt.close(fig2b)
    paths["spacecraft_hybrid_vs_beam_science"] = str(path2b)

    sorted_sp = sorted(spacecraft_mc, key=lambda r: int(r.get("run", 0)))
    fig2c, ax2c = plt.subplots(figsize=(10, 4))
    sp_idx = [int(r.get("run", i)) for i, r in enumerate(sorted_sp)]
    sp_science = [float(r.get("delivered_science", np.nan)) for r in sorted_sp]
    sp_solar = [float(r.get("solar_charge_w", np.nan)) for r in sorted_sp]
    scatter = ax2c.scatter(sp_idx, sp_science, c=sp_solar, cmap="viridis", s=55)
    ax2c.plot(sp_idx, sp_science, color="tab:gray", alpha=0.5, linewidth=1.0)
    ax2c.set_xlabel("Scenario Index")
    ax2c.set_ylabel("Delivered Science")
    ax2c.set_title("Spacecraft Science vs Scenario")
    ax2c.grid(alpha=0.3)
    cb = fig2c.colorbar(scatter, ax=ax2c)
    cb.set_label("Solar Charge [W]")
    fig2c.tight_layout()
    path2c = output_dir / "spacecraft_scenario_science.png"
    fig2c.savefig(path2c, dpi=180)
    plt.close(fig2c)
    paths["spacecraft_scenario_science"] = str(path2c)

    return paths


def _plot_baseline_comparison(baseline_rows: List[Dict[str, Any]], output_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    labels = [f"{r['domain']}:{r['metric']}" for r in baseline_rows]
    x_idx = np.arange(len(labels))
    width = 0.38
    beam_vals = [float(r["beam"]) for r in baseline_rows]
    greedy_vals = [float(r["greedy"]) for r in baseline_rows]
    ax3.bar(x_idx - width / 2.0, beam_vals, width=width, label="beam")
    ax3.bar(x_idx + width / 2.0, greedy_vals, width=width, label="greedy")
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(labels, rotation=20, ha="right")
    ax3.set_title("Baseline Comparison (Beam vs Greedy)")
    ax3.legend()
    ax3.grid(alpha=0.3, axis="y")
    fig3.tight_layout()
    path3 = output_dir / "baseline_comparison.png"
    fig3.savefig(path3, dpi=180)
    plt.close(fig3)
    paths["baseline_comparison"] = str(path3)
    return paths


def _plot_stress_views(
    stress_rows: List[Dict[str, Any]],
    stress_details: Dict[str, List[Dict[str, Any]]],
    aircraft_cfg: Dict[str, Any],
    spacecraft_cfg: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}

    fig4, ax4 = plt.subplots(1, 2, figsize=(12, 4))
    aircraft_stress = [r for r in stress_rows if r.get("domain") == "aircraft"]
    spacecraft_stress = [r for r in stress_rows if r.get("domain") == "spacecraft"]

    if aircraft_stress:
        labels = [str(r["scenario"]) for r in aircraft_stress]
        vals = [float(r.get("energy_remaining_wh", np.nan)) for r in aircraft_stress]
        x_idx = np.arange(len(labels))
        ax4[0].bar(x_idx, vals, color="tab:blue", alpha=0.8)
        ax4[0].axhline(float(aircraft_cfg.get("reserve_energy_wh", 0.0)), color="tab:red", linestyle="--", label="reserve")
        ax4[0].set_xticks(x_idx)
        ax4[0].set_xticklabels(labels, rotation=20, ha="right")
        ax4[0].set_title("Aircraft Stress Energy Margin")
        ax4[0].set_ylabel("Energy Remaining [Wh]")
        ax4[0].legend()
        ax4[0].grid(alpha=0.3, axis="y")

    if spacecraft_stress:
        labels = [str(r["scenario"]) for r in spacecraft_stress]
        vals = [float(r.get("final_battery_wh", np.nan)) for r in spacecraft_stress]
        x_idx = np.arange(len(labels))
        ax4[1].bar(x_idx, vals, color="tab:green", alpha=0.8)
        ax4[1].axhline(float(spacecraft_cfg.get("battery_min_wh", 0.0)), color="tab:red", linestyle="--", label="battery_min")
        ax4[1].set_xticks(x_idx)
        ax4[1].set_xticklabels(labels, rotation=20, ha="right")
        ax4[1].set_title("Spacecraft Stress Battery Margin")
        ax4[1].set_ylabel("Final Battery [Wh]")
        ax4[1].legend()
        ax4[1].grid(alpha=0.3, axis="y")

    fig4.tight_layout()
    path4 = output_dir / "stress_margin_plot.png"
    fig4.savefig(path4, dpi=180)
    plt.close(fig4)
    paths["stress_margin_plot"] = str(path4)

    downlink_rows = stress_details.get("spacecraft_low_solar", [])
    if downlink_rows:
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        xs = [float(r["start_s"]) / 3600.0 for r in downlink_rows]
        ys = [float(r["min_elevation_deg"]) for r in downlink_rows]
        ax5.scatter(xs, ys, color="tab:blue", label="scheduled downlinks")
        ax5.axhline(float(spacecraft_cfg.get("min_downlink_elevation_deg", 10.0)), color="tab:red", linestyle="--", label="min elevation")
        ax5.set_xlabel("Start Time [hours]")
        ax5.set_ylabel("Minimum Elevation in Window [deg]")
        ax5.set_title("Stress Scenario Downlink Schedule vs Elevation Constraint")
        ax5.grid(alpha=0.3)
        ax5.legend()
        fig5.tight_layout()
        path5 = output_dir / "stress_downlink_vs_constraints.png"
        fig5.savefig(path5, dpi=180)
        plt.close(fig5)
        paths["stress_downlink_vs_constraints"] = str(path5)

    trace_rows = stress_details.get("spacecraft_low_solar_trace", [])
    if trace_rows:
        sorted_trace = sorted(trace_rows, key=lambda r: float(r.get("time_s", 0.0)))
        t_h = [float(r.get("time_s", 0.0)) / 3600.0 for r in sorted_trace]
        batt = [float(r.get("battery_wh", np.nan)) for r in sorted_trace]
        data = [float(r.get("data_buffer_mb", np.nan)) for r in sorted_trace]
        sci = [float(r.get("delivered_science", np.nan)) for r in sorted_trace]

        fig6, ax6 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax6[0].plot(t_h, batt, color="tab:green")
        ax6[0].set_ylabel("Battery [Wh]")
        ax6[0].grid(alpha=0.3)
        ax6[1].plot(t_h, data, color="tab:orange")
        ax6[1].set_ylabel("Data Buffer [MB]")
        ax6[1].grid(alpha=0.3)
        ax6[2].plot(t_h, sci, color="tab:blue")
        ax6[2].set_ylabel("Delivered Science")
        ax6[2].set_xlabel("Mission Time [hours]")
        ax6[2].grid(alpha=0.3)
        fig6.suptitle("Spacecraft Stress Scenario Resource Evolution")
        fig6.tight_layout()
        path6 = output_dir / "spacecraft_stress_resource_evolution.png"
        fig6.savefig(path6, dpi=180)
        plt.close(fig6)
        paths["spacecraft_stress_resource_evolution"] = str(path6)

    return paths


def plot_validation_bundle(
    aircraft_mc: List[Dict[str, Any]],
    spacecraft_mc: List[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
    stress_rows: List[Dict[str, Any]],
    stress_details: Dict[str, List[Dict[str, Any]]],
    aircraft_cfg: Dict[str, Any],
    spacecraft_cfg: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    paths.update(_plot_aircraft_views(aircraft_mc, output_dir))
    paths.update(_plot_spacecraft_views(spacecraft_mc, output_dir))
    paths.update(_plot_baseline_comparison(baseline_rows, output_dir))
    paths.update(_plot_stress_views(stress_rows, stress_details, aircraft_cfg, spacecraft_cfg, output_dir))
    return paths
