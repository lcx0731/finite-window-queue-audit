"""Boundary identities checked against integer-grid and max-plus oracles."""

from itertools import combinations_with_replacement, product
import math

import numpy as np
import pytest

from queue_audit import (
    audit_observations, audit_window, interval_area, occupancy_steps,
    simulate_fcfs, trace_areas,
)


def test_fcfs_recurrence_and_idle_period():
    trace = simulate_fcfs([0, 1, 2, 3, 12], [4, 1, 2, 1, 2])
    np.testing.assert_array_equal(trace.starts, [0, 4, 5, 7, 12])
    np.testing.assert_array_equal(trace.departures, [4, 5, 7, 8, 14])


def test_inputs_are_copied_and_arrays_read_only():
    arrivals, services = np.array([1.0, 2.0]), np.array([1.0, 1.0])
    trace = simulate_fcfs(arrivals, services)
    arrivals[:] = 100
    services[:] = 100
    np.testing.assert_array_equal(trace.departures, [2, 3])
    for value in (trace.arrivals, trace.services, trace.starts, trace.departures):
        with pytest.raises(ValueError):
            value[0] = 0


def test_tied_arrivals_zero_services_and_simultaneous_departures():
    trace = simulate_fcfs([0, 0, 0, 2, 2], [0, 2, 0, 0, 1])
    np.testing.assert_array_equal(trace.departures, [0, 2, 2, 2, 3])
    audit = audit_window(trace, 0, 2)
    assert audit.arrivals == 3 and audit.completions == 1
    assert audit.carry_out_jobs == 2 and audit.system_area == 4
    assert audit.conservation_residual == 0
    edges, counts = occupancy_steps(trace.arrivals, trace.departures, 0, 4)
    np.testing.assert_array_equal(edges, [0, 2, 3, 4])
    np.testing.assert_array_equal(counts, [2, 1, 0])


@pytest.mark.parametrize("arrivals,services", [
    ([1, 0], [1, 1]), ([0], []), ([-1], [1]), ([0], [-1]),
    ([math.nan], [1]), ([0], [math.inf]), ([math.inf], [1]),
    ([[0]], [[1]]), ([True], [1]), ([0], [False]),
    ([0j], [1]), ([0], [1j]), (["0"], [1]),
    ([1e308], [1e308]), ([1e16], [0.1]),
])
def test_bad_trace_inputs_rejected(arrivals, services):
    with pytest.raises(ValueError):
        simulate_fcfs(arrivals, services)


@pytest.mark.parametrize("left,right", [
    (-1, 1), (1, 1), (2, 1), (0, math.inf), (math.nan, 1),
    (False, 1), (0, True), (None, 1), (0, "bad"),
])
def test_bad_window_inputs_rejected(left, right):
    with pytest.raises(ValueError):
        audit_observations([], [], left, right)


def test_empty_trace_and_no_completion_mean():
    trace = simulate_fcfs([], [])
    audit = audit_window(trace, 3, 8)
    assert audit.mean_completed_sojourn is None
    assert audit.mean_occupancy == audit.corrected_occupancy == audit.naive_occupancy == 0
    assert audit.conservation_residual == 0
    assert trace_areas(trace, 3, 8) == {"system": 0, "waiting": 0, "busy": 0}


@pytest.mark.parametrize("left,right,area,c,p,u,n0,n1,completed", [
    (0, 4, 10, 0, 0, 10, 0, 4, 0),
    (4, 8, 8, 13, 9, 4, 4, 1, 3),
    (8, 9, 0, 5, 5, 0, 1, 0, 1),
    (0, 9, 18, 18, 0, 0, 0, 0, 4),
])
def test_hand_computed_boundary_corrections(left, right, area, c, p, u, n0, n1, completed):
    trace = simulate_fcfs([0, 1, 2, 3], [4, 1, 2, 1])
    audit = audit_window(trace, left, right)
    assert audit.system_area == audit.event_sweep_area == c - p + u == area
    assert audit.completed_sojourn_sum == c
    assert audit.prewindow_completed_area == p
    assert audit.unfinished_window_area == u
    assert audit.carry_in_jobs == n0 and audit.carry_out_jobs == n1
    assert audit.completions == completed
    assert audit.corrected_occupancy == audit.mean_occupancy
    assert audit.conservation_residual == 0


def test_observation_only_api_does_not_assume_fcfs():
    audit = audit_observations([2, 0, 1], [np.nan, np.nan, 3], 2, 5)
    assert audit.system_area == audit.event_sweep_area == 7
    assert audit.completed_sojourn_sum == 2
    assert audit.prewindow_completed_area == 1
    assert audit.unfinished_window_area == 6
    assert audit.carry_in_jobs == 2 and audit.carry_out_jobs == 2
    assert audit.corrected_occupancy == pytest.approx(7 / 3)
    assert audit.conservation_residual == 0


@pytest.mark.parametrize("arrivals,departures", [
    ([0], []), ([0], [5]), ([5], [np.nan]), ([2], [1]),
    ([0], [np.inf]), ([0], [-1]), ([np.nan], [1]),
    ([0], [[1]]), ([0], [True]), ([0], ["NaN"]),
])
def test_invalid_or_future_observations_rejected(arrivals, departures):
    with pytest.raises(ValueError):
        audit_observations(arrivals, departures, 0, 5)


def test_events_at_right_are_unobserved_events_at_left_are_observed():
    trace = simulate_fcfs([0, 2, 4], [2, 2, 2])
    audit = audit_window(trace, 2, 4)
    assert (audit.carry_in_jobs, audit.arrivals, audit.completions, audit.carry_out_jobs) == (1, 1, 1, 1)
    assert audit.system_area == 2
    assert audit.completed_sojourn_sum == audit.prewindow_completed_area == 2
    assert audit.unfinished_window_area == 2


def test_future_services_and_future_arrivals_cannot_change_observation_audit():
    a = simulate_fcfs([0, 1, 2], [100, 1, 1])
    b = simulate_fcfs([0, 1, 2, 99], [200, 50, 7, 1])
    assert audit_window(a, 3, 10) == audit_window(b, 3, 10)


def test_integer_exhaustive_oracles_for_540_small_traces():
    # Independent O(n^2) max-plus solution, then integer-grid occupancy counts.
    # Neither oracle uses the sequential recursion or clipped-overlap formula.
    for arrivals in combinations_with_replacement(range(4), 3):
        for services in product(range(3), repeat=3):
            trace = simulate_fcfs(arrivals, services)
            expected = [max(arrivals[j] + sum(services[j:i + 1]) for j in range(i + 1))
                        for i in range(3)]
            np.testing.assert_array_equal(trace.departures, expected)
            for left, right in ((0, 2), (1, 4), (4, 8), (0, 9)):
                expected_area = sum(sum(a <= tick + 0.5 < d for a, d in zip(arrivals, expected))
                                    for tick in range(left, right))
                audit = audit_window(trace, left, right)
                assert audit.system_area == audit.event_sweep_area == expected_area
                assert audit.corrected_occupancy == expected_area / (right - left)
                assert audit.conservation_residual == 0
                areas = trace_areas(trace, left, right)
                assert areas["waiting"] + areas["busy"] == expected_area
                assert 0 <= areas["busy"] <= right - left


def test_area_additivity_but_completed_mean_not_additive():
    trace = simulate_fcfs([0, 1, 2, 3], [4, 1, 2, 1])
    parts = [audit_window(trace, a, b) for a, b in ((0, 4), (4, 8), (8, 9))]
    whole = audit_window(trace, 0, 9)
    assert sum(p.system_area for p in parts) == whole.system_area
    assert sum(p.completed_sojourn_sum for p in parts) == whole.completed_sojourn_sum
    assert sum(p.unfinished_window_area - p.prewindow_completed_area for p in parts) == 0


def test_time_scale_and_translation_invariance():
    trace = simulate_fcfs([0, 1, 2, 3], [4, 1, 2, 1])
    base = audit_window(trace, 4, 8)
    scaled = simulate_fcfs(trace.arrivals * 2.5 + 100, trace.services * 2.5)
    changed = audit_window(scaled, 110, 120)
    assert changed.system_area == base.system_area * 2.5
    assert changed.mean_occupancy == base.mean_occupancy
    assert changed.corrected_occupancy == base.corrected_occupancy
    assert changed.mean_completed_sojourn == pytest.approx(base.mean_completed_sojourn * 2.5, rel=1e-14)


def test_general_interval_sweep_with_unsorted_and_zero_length_intervals():
    entries, exits = [3, 1, 0, 2, 4], [5, 1, 7, 6, 4]
    edges, counts = occupancy_steps(entries, exits, 1, 5)
    assert interval_area(entries, exits, 1, 5) == 9
    assert sum(np.diff(edges) * counts) == 9
    assert np.all(counts >= 0)


@pytest.mark.parametrize("entries,exits", [([1], [0]), ([0], []), ([np.nan], [1])])
def test_invalid_intervals_rejected(entries, exits):
    for function in (interval_area, occupancy_steps):
        with pytest.raises(ValueError):
            function(entries, exits, 0, 3)


def test_overflowing_area_is_rejected():
    with pytest.raises(ValueError, match="floating-point range"):
        interval_area([0, 0], [1e308, 1e308], 0, 1e308)


def test_observation_row_permutation_invariance():
    a, d = np.array([0, 1, 3, 6]), np.array([2, 5, np.nan, np.nan])
    expected = audit_observations(a, d, 2, 8)
    for permutation in ((3, 2, 1, 0), (1, 3, 0, 2)):
        assert audit_observations(a[list(permutation)], d[list(permutation)], 2, 8) == expected


def test_old_completed_records_can_be_omitted():
    whole = audit_observations([0, 1, 4], [1, 3, np.nan], 3, 8)
    trimmed = audit_observations([1, 4], [3, np.nan], 3, 8)
    assert whole == trimmed
