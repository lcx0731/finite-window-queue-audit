import json

import numpy as np
import pytest

from queue_audit.experiment import run_experiment, synthetic_trace, write_outputs


def test_seeded_experiment_repeats_exactly():
    first = run_experiment(replicates=8, seed=42)
    assert first == run_experiment(replicates=8, seed=42)
    assert first["maximum_absolute_identity_error"] < 1e-9
    assert len(first["scenarios"]) == 3
    for scenario in first["scenarios"]:
        metrics = scenario["estimates"]
        assert metrics["mean_occupancy"]["mean"] == pytest.approx(metrics["corrected_occupancy"]["mean"])
        assert metrics["signed_mismatch"]["mean"] == pytest.approx(
            metrics["prewindow_correction"]["mean"] - metrics["unfinished_correction"]["mean"])
        assert metrics["signed_mismatch"]["mcse"] >= 0


def test_output_json_and_plots_are_created(tmp_path):
    result = run_experiment(replicates=3, seed=20261020)
    write_outputs(result, tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text()) == result
    for name in ("boundary-accounting.png", "replication-comparison.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (tmp_path / name).stat().st_size > 10_000


@pytest.mark.parametrize("replicates,seed", [(1, 1), (True, 1), (2.5, 1), (2, -1), (2, False)])
def test_bad_experiment_arguments(replicates, seed):
    with pytest.raises(ValueError):
        run_experiment(replicates, seed)


@pytest.mark.parametrize("segments", [((0, 1, -1),), ((1, 0, 1),),
                                        ((0, 2, 1), (1, 3, 1)), ((0, np.inf, 1),)])
def test_invalid_arrival_segments(segments):
    with pytest.raises(ValueError):
        synthetic_trace(np.random.default_rng(1), segments)


def test_zero_intensity_produces_no_arrivals():
    trace = synthetic_trace(np.random.default_rng(1), ((0, 10, 0),))
    assert len(trace.arrivals) == 0


def test_mcse_uses_independent_run_level_differences():
    # Compute the first scenario independently from the same replication streams.
    from queue_audit import audit_window

    differences = []
    for replicate in range(5):
        rng = np.random.default_rng(np.random.SeedSequence([19, 0, replicate]))
        trace = synthetic_trace(rng, ((0, 160, 0.7),))
        audit = audit_window(trace, 80, 160)
        differences.append(audit.naive_occupancy - audit.mean_occupancy)
    result = run_experiment(5, 19)["scenarios"][0]["estimates"]["signed_mismatch"]
    assert result["mean"] == np.mean(differences)
    assert result["mcse"] == np.std(differences, ddof=1) / np.sqrt(5)
