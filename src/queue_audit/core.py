"""FCFS simulation and exact-in-real-arithmetic finite-window identities.

Events at the left endpoint are observed; events at the right endpoint are not.
Counts at the boundaries are *left limits*. Time integrals are unaffected by the
value at a single event time. Incomplete departures are represented by NaN, not
by an estimated eventual departure time.
"""

from dataclasses import dataclass, field
import math

import numpy as np


def _vector(values, name, *, allow_nan=False):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a one-dimensional real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=float, copy=True)
    valid = np.isfinite(result) | (np.isnan(result) if allow_nan else False)
    if not np.all(valid) or np.any(result < 0):
        raise ValueError(f"{name} must contain nonnegative finite times" +
                         (" or NaN for incomplete jobs" if allow_nan else ""))
    return result


def _window(left, right):
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        raise ValueError("window endpoints must be finite numeric times")
    try:
        left, right = float(left), float(right)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("window endpoints must be finite numeric times") from exc
    if not (math.isfinite(left) and math.isfinite(right) and 0 <= left < right):
        raise ValueError("window must satisfy 0 <= left < right < infinity")
    return left, right


def _sum(values):
    try:
        answer = math.fsum(float(value) for value in values)
    except OverflowError as exc:
        raise ValueError("time integral exceeds floating-point range") from exc
    if not math.isfinite(answer):
        raise ValueError("time integral exceeds floating-point range")
    return answer


def _intervals(entries, exits):
    entries, exits = _vector(entries, "entries"), _vector(exits, "exits")
    if entries.shape != exits.shape or np.any(exits < entries):
        raise ValueError("interval arrays must match and each exit must follow its entry")
    return entries, exits


@dataclass(frozen=True, eq=False)
class QueueTrace:
    """An initially empty, work-conserving, single-server FCFS queue.

    Tied arrivals retain input order. Zero service times are allowed. Stored
    arrays are copied and read-only. Times are float64; a positive service too
    small to advance its start timestamp is rejected rather than silently lost.
    """

    arrivals: np.ndarray
    services: np.ndarray
    starts: np.ndarray = field(init=False)
    departures: np.ndarray = field(init=False)

    def __post_init__(self):
        arrivals = _vector(self.arrivals, "arrivals")
        services = _vector(self.services, "services")
        if arrivals.shape != services.shape or np.any(np.diff(arrivals) < 0):
            raise ValueError("arrivals must be sorted and match service times")
        starts, departures = np.empty_like(arrivals), np.empty_like(arrivals)
        previous = 0.0
        for i, (arrival, service) in enumerate(zip(arrivals, services)):
            start = max(float(arrival), previous)
            departure = start + float(service)
            if not math.isfinite(departure) or (service > 0 and departure <= start):
                raise ValueError("service completion cannot be represented at this time scale")
            starts[i], departures[i] = start, departure
            previous = departure
        for name, value in (("arrivals", arrivals), ("services", services),
                            ("starts", starts), ("departures", departures)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def simulate_fcfs(arrivals, services):
    """Compute start_i=max(arrival_i, departure_(i-1)), departure_i=start_i+service_i."""
    return QueueTrace(arrivals, services)


def interval_area(entries, exits, left, right):
    """Sum clipped interval lengths; overlapping intervals each contribute."""
    left, right = _window(left, right)
    entries, exits = _intervals(entries, exits)
    return _sum(np.maximum(0.0, np.minimum(exits, right) - np.maximum(entries, left)))


def occupancy_steps(entries, exits, left, right):
    """Independently sweep entry/exit events, grouping simultaneous changes.

    Return edges and counts, with count[j] on [edges[j], edges[j+1]). Zero-length
    intervals contribute no occupancy, including when they occur at a boundary.
    This function does not obtain its result from the clipped-overlap formula.
    """
    left, right = _window(left, right)
    entries, exits = _intervals(entries, exits)
    count = int(np.count_nonzero((entries <= left) & (exits > left)))
    events = {}
    for entry, exit_ in zip(entries, exits):
        if left < entry < right:
            events[float(entry)] = events.get(float(entry), 0) + 1
        if left < exit_ < right:
            events[float(exit_)] = events.get(float(exit_), 0) - 1
    edges, counts = [left], []
    for time, delta in sorted(events.items()):
        counts.append(count)
        edges.append(time)
        count += delta
    counts.append(count)
    edges.append(right)
    return np.asarray(edges), np.asarray(counts, dtype=np.int64)


@dataclass(frozen=True)
class WindowAudit:
    left: float
    right: float
    arrivals: int
    completions: int
    carry_in_jobs: int
    carry_out_jobs: int
    system_area: float
    event_sweep_area: float
    completed_sojourn_sum: float
    prewindow_completed_area: float
    unfinished_window_area: float

    @property
    def duration(self):
        return self.right - self.left

    @property
    def mean_occupancy(self):
        return self.system_area / self.duration

    @property
    def departure_rate(self):
        return self.completions / self.duration

    @property
    def mean_completed_sojourn(self):
        return self.completed_sojourn_sum / self.completions if self.completions else None

    @property
    def naive_occupancy(self):
        """C/T; equals departure rate times completed mean when completions > 0.

        If there are no completions C/T is zero, but a completed mean is undefined.
        """
        return self.completed_sojourn_sum / self.duration

    @property
    def corrected_occupancy(self):
        return _sum([self.completed_sojourn_sum, -self.prewindow_completed_area,
                     self.unfinished_window_area]) / self.duration

    @property
    def conservation_residual(self):
        return self.carry_in_jobs + self.arrivals - self.completions - self.carry_out_jobs

    def to_dict(self):
        from dataclasses import asdict

        result = asdict(self)
        for name in ("mean_occupancy", "departure_rate", "mean_completed_sojourn",
                     "naive_occupancy", "corrected_occupancy", "conservation_residual"):
            result[name] = getattr(self, name)
        return result


def audit_observations(arrivals, observed_departures, left, right):
    """Audit complete event records available strictly before ``right``.

    Include every job arriving before right, or at least every job whose lifetime
    intersects the window. Include historical arrivals for carry-in jobs. Each
    departure must either be an observed timestamp < right, or NaN if incomplete.
    Rows need not be sorted and no FCFS discipline is assumed here. Missing rows
    or falsely censored completions cannot be detected from these arrays alone.
    """
    left, right = _window(left, right)
    arrivals = _vector(arrivals, "arrivals")
    departures = _vector(observed_departures, "observed_departures", allow_nan=True)
    if arrivals.shape != departures.shape:
        raise ValueError("arrival and departure records must have equal length")
    if np.any(arrivals >= right):
        raise ValueError("arrival records must be strictly before the cutoff")
    observed = ~np.isnan(departures)
    if np.any(departures[observed] >= right) or np.any(departures[observed] < arrivals[observed]):
        raise ValueError("observed departures must follow arrival and be strictly before cutoff")
    clipped_departures = np.where(observed, departures, right)
    completed = observed & (departures >= left)
    unfinished = ~observed
    edges, counts = occupancy_steps(arrivals, clipped_departures, left, right)
    # Python multiplication gives inf (checked by _sum), not a NumPy warning.
    sweep_area = _sum(float(dt) * int(n) for dt, n in zip(np.diff(edges), counts))
    return WindowAudit(
        left=left, right=right,
        arrivals=int(np.count_nonzero(arrivals >= left)),
        completions=int(np.count_nonzero(completed)),
        carry_in_jobs=int(np.count_nonzero((arrivals < left) & (clipped_departures >= left))),
        carry_out_jobs=int(np.count_nonzero(unfinished)),
        system_area=interval_area(arrivals, clipped_departures, left, right),
        event_sweep_area=sweep_area,
        completed_sojourn_sum=_sum(departures[completed] - arrivals[completed]),
        prewindow_completed_area=_sum(np.maximum(0.0, left - arrivals[completed])),
        unfinished_window_area=_sum(right - np.maximum(left, arrivals[unfinished])),
    )


def audit_window(trace, left, right):
    """Hide future departures before calling the observation-only audit."""
    left, right = _window(left, right)
    mask = trace.arrivals < right
    observed = np.where(trace.departures[mask] < right, trace.departures[mask], np.nan)
    return audit_observations(trace.arrivals[mask], observed, left, right)


def trace_areas(trace, left, right):
    """Time integrals of system occupancy, waiting jobs, and server busy state."""
    return {
        "system": interval_area(trace.arrivals, trace.departures, left, right),
        "waiting": interval_area(trace.arrivals, trace.starts, left, right),
        "busy": interval_area(trace.starts, trace.departures, left, right),
    }
