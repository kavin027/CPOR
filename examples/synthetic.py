"""Small causal stream exercising protection, weight fitting and settlement.
This synthetic integration check is not a benchmark accuracy experiment.
"""
from pathlib import Path
import sys, json
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpor import CPOR, load_core

def main():
    S = np.array([[1., 1.], [1., 0.], [0., 1.]])
    v = np.ones(3)
    core = load_core()
    actor = CPOR(S, v)
    previous = None
    fits = 0
    for t in range(10):
        # These predictors depend only on the time index and past feedback.
        ub = np.array([10. + t * .2, 20. + t * .1])
        b = S @ ub
        experts = b[:, None] * np.linspace(.96, 1.04, 9)[None, :]
        target = experts.mean(axis=1)
        row = actor.issue(t, b, ub, target, experts, previous,
                          calibration=t < 4, reset=t == 4)
        assert np.allclose(row["pred"], S @ row["witness"], atol=1e-8)
        assert core.original.certified(b, row["pred"], actor.A,
                                       core.original.RR, core.original.K, True)
        assert not row["solver_failed"], row["fit"]
        fits += int(t >= 4)
        # Reveal this outcome only after issuing the forecast.
        previous = S @ (ub + np.array([1.0, -.5]))
    actor.close(previous)
    actor._actor.pending = None
    assert all(a >= 0 for a in actor.A)
    report = dict(releases=10, scored_weight_fits=fits, solver_failures=0,
                  exact_envelope_checks=True, nonnegative_final_allowances=True,
                  coherence_checked_at_float_tolerance=True,
                  scope="Synthetic integration check, not empirical table reproduction.")
    print(json.dumps(report, indent=2))
    (Path(__file__).parent / "SYNTHETIC_CHECK.json").write_text(json.dumps(report, indent=2))

if __name__ == "__main__":
    with threadpool_limits(1):
        main()
