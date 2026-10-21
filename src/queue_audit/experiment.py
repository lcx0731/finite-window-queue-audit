"""Reproduce two exact boundary examples and seeded, independent queue runs."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import audit_window, occupancy_steps, simulate_fcfs, trace_areas


SCENARIOS = (
    ("constant_input", "Constant input", ((0.0, 160.0, 0.7),)),
    ("sustained_overload", "Sustained overload", ((0.0, 160.0, 1.3),)),
    ("burst_then_relief", "Burst then relief", ((0.0, 80.0, 1.5), (80.0, 160.0, 0.25))),
)


def synthetic_trace(rng, segments):
    """Piecewise-Poisson arrivals; independent exponential services of mean 1.

    Draw the Poisson count on each segment, then sorted uniform event times
    conditional on that count. The queue is initially empty at time zero.
    """
    chunks = []
    previous = 0.0
    for left, right, rate in segments:
        if not all(math.isfinite(x) for x in (left, right, rate)) or not (
            previous <= left < right and rate >= 0
        ):
            raise ValueError("segments must be ordered, non-overlapping, and have nonnegative rates")
        count = rng.poisson(rate * (right - left))
        chunks.append(np.sort(rng.uniform(left, right, count)))
        previous = right
    arrivals = np.concatenate(chunks) if chunks else np.array([], dtype=float)
    services = rng.exponential(1.0, len(arrivals))
    return simulate_fcfs(arrivals, services)


def _estimate(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "mcse": float(np.std(values, ddof=1) / math.sqrt(len(values))),
    }


def _check(audit, trace):
    scale = max(1.0, audit.system_area, audit.completed_sojourn_sum,
                audit.prewindow_completed_area, audit.unfinished_window_area)
    errors = (
        abs(audit.system_area - audit.event_sweep_area),
        abs(audit.system_area - audit.corrected_occupancy * audit.duration),
    )
    areas = trace_areas(trace, audit.left, audit.right)
    errors += (abs(areas["system"] - audit.system_area),
               abs(areas["system"] - areas["waiting"] - areas["busy"]))
    if audit.conservation_residual != 0 or max(errors) > 1e-11 * scale:
        raise ArithmeticError("finite-window accounting checks failed")
    if not (0 <= areas["busy"] <= audit.duration * (1 + 1e-12)):
        raise ArithmeticError("single-server busy time is inconsistent")
    return max(errors)


def run_experiment(replicates=400, seed=20261020):
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("replicates must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    exact_trace = simulate_fcfs([0, 1, 2, 3], [4, 1, 2, 1])
    exact = {}
    max_error = 0.0
    for label, left, right in (("build_up", 0, 4), ("drain", 4, 8), ("fully_observed", 0, 9)):
        audit = audit_window(exact_trace, left, right)
        max_error = max(max_error, _check(audit, exact_trace))
        exact[label] = audit.to_dict()
    scenarios = []
    for scenario_index, (key, label, segments) in enumerate(SCENARIOS):
        rows = []
        no_completions = 0
        for replicate in range(replicates):
            rng = np.random.default_rng(np.random.SeedSequence([seed, scenario_index, replicate]))
            trace = synthetic_trace(rng, segments)
            audit = audit_window(trace, 80, 160)
            max_error = max(max_error, _check(audit, trace))
            no_completions += int(audit.completions == 0)
            areas = trace_areas(trace, 80, 160)
            rows.append({
                "mean_occupancy": audit.mean_occupancy,
                "completion_only_proxy": audit.naive_occupancy,
                "corrected_occupancy": audit.corrected_occupancy,
                "signed_mismatch": audit.naive_occupancy - audit.mean_occupancy,
                "prewindow_correction": audit.prewindow_completed_area / audit.duration,
                "unfinished_correction": audit.unfinished_window_area / audit.duration,
                "utilization": areas["busy"] / audit.duration,
                "carry_in_jobs": audit.carry_in_jobs,
                "carry_out_jobs": audit.carry_out_jobs,
            })
        scenarios.append({
            "key": key, "label": label,
            "arrival_segments": [list(segment) for segment in segments],
            "window": [80, 160],
            "replicates_with_no_completions": no_completions,
            "estimates": {name: _estimate([row[name] for row in rows]) for name in rows[0]},
        })
    return {
        "data": "Seeded synthetic arrivals and services; no external or personal data.",
        "initial_condition": "Empty queue at time zero; no stationarity assumption.",
        "service_distribution": "Independent exponential with mean 1 time unit.",
        "seed": seed, "replicates_per_scenario": replicates,
        "mcse_unit": "Independent simulation replication; signed mismatch is paired within each run.",
        "maximum_absolute_identity_error": max_error,
        "exact_examples": exact, "scenarios": scenarios,
    }


def _plot_exact(output):
    trace = simulate_fcfs([0, 1, 2, 3], [4, 1, 2, 1])
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1]}, layout="constrained")
    for i, (arrival, start, departure) in enumerate(zip(trace.arrivals, trace.starts, trace.departures)):
        ax.broken_barh([(arrival, start - arrival)], (i - 0.28, 0.56),
                       facecolors="#d4a84b", label="Waiting" if i == 0 else None)
        ax.broken_barh([(start, departure - start)], (i - 0.28, 0.56),
                       facecolors="#376b8a", label="In service" if i == 0 else None)
        ax.plot(arrival, i, "o", color="#253541", markersize=4)
    ax.set(yticks=range(4), yticklabels=[f"Job {i + 1}" for i in range(4)],
           title="The same four jobs can understate or overstate occupancy")
    ax.legend(loc="upper right", ncols=2)
    edges, counts = occupancy_steps(trace.arrivals, trace.departures, 0, 9)
    bx.stairs(counts, edges, color="#376b8a", linewidth=2, fill=True, alpha=0.35)
    bx.set(xlabel="Time (arbitrary units)", ylabel="Jobs in system", ylim=(0, 4.9),
           xlim=(0, 9), yticks=range(5))
    for panel in (ax, bx):
        panel.axvspan(4, 8, color="#bf7044", alpha=0.08)
        panel.axvline(4, color="#bf7044", linestyle="--", linewidth=1)
        panel.axvline(8, color="#bf7044", linestyle="--", linewidth=1)
        panel.spines[["top", "right"]].set_visible(False)
    bx.text(0.2, 4.3, "[0, 4): area = 0 − 0 + 10 = 10", fontsize=10)
    bx.text(4.2, 4.3, "[4, 8): area = 13 − 9 + 4 = 8", fontsize=10)
    fig.savefig(output / "boundary-accounting.png", dpi=170)
    plt.close(fig)


def _plot_scenarios(result, output):
    scenarios = result["scenarios"]
    x = np.arange(len(scenarios))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    for offset, key, label, color in (
        (-0.18, "mean_occupancy", "Time-average occupancy", "#376b8a"),
        (0.18, "completion_only_proxy", "Completion-only proxy", "#bf7044"),
    ):
        ax.bar(x + offset, [s["estimates"][key]["mean"] for s in scenarios], 0.34,
               yerr=[2 * s["estimates"][key]["mcse"] for s in scenarios],
               capsize=3, color=color, label=label)
    ax.set(ylabel="Mean over independent runs (jobs)", title="Same measurement window: [80, 160)")
    ax.legend(fontsize=9, loc="upper left")
    for offset, key, sign, label, color in (
        (-0.18, "prewindow_correction", 1, "+ pre-window completed time / T", "#bf7044"),
        (0.18, "unfinished_correction", -1, "− unfinished in-window time / T", "#376b8a"),
    ):
        bx.bar(x + offset, [sign * s["estimates"][key]["mean"] for s in scenarios], 0.34,
               color=color, alpha=0.8, label=label)
    bx.errorbar(x, [s["estimates"]["signed_mismatch"]["mean"] for s in scenarios],
                yerr=[2 * s["estimates"]["signed_mismatch"]["mcse"] for s in scenarios],
                fmt="D", color="#222b34", capsize=4, label="Net mismatch (proxy − actual)")
    bx.axhline(0, color="#6a737a", linewidth=0.8)
    bx.set(ylabel="Signed occupancy contribution (jobs)", title="Boundary terms explain the mismatch")
    bx.legend(fontsize=8, loc="lower left")
    for panel in (ax, bx):
        panel.set_xticks(x, [s["label"].replace(" ", "\n") for s in scenarios])
        panel.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Synthetic queues · {result['replicates_per_scenario']} runs per scenario · "
                 "error bars = ±2 Monte Carlo SE", fontsize=12)
    fig.savefig(output / "replication-comparison.png", dpi=170)
    plt.close(fig)


def write_outputs(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    _plot_exact(output)
    _plot_scenarios(result, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--replicates", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    try:
        result = run_experiment(args.replicates, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    write_outputs(result, args.output)
    print(json.dumps({"output": str(args.output),
                      "maximum_absolute_identity_error": result["maximum_absolute_identity_error"],
                      "replicates_per_scenario": args.replicates}, indent=2))


if __name__ == "__main__":
    main()
