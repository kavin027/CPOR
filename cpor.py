"""CPOR: Cumulatively Protected Online Reconciliation.

Public entry point for the unchanged protected root-risk procedure.
Historical storage keys and frozen implementation files remain readable.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import runpy
import sys

METHOD_NAME = "CPOR"
FULL_NAME = "Cumulatively Protected Online Reconciliation"
LEGACY_STORAGE_KEY = "C0"
ALIASES = {"CPOR", "C0", "RootFull", "RR-RECAP"}
_CORE = None


def storage_key(name=METHOD_NAME):
    """Translate a public method name to its unchanged archived identifier."""
    if name not in ALIASES:
        raise ValueError(f"Unknown CPOR alias: {name}")
    return LEGACY_STORAGE_KEY


def load_core(project_root=None):
    global _CORE
    root = Path(project_root or Path(__file__).resolve().parent).resolve()
    source = root / "output/paired_root_actual_credit_20261013/numerical_v2/run.py"
    if _CORE is not None and Path(_CORE.__file__).resolve() == source:
        return _CORE
    if not source.is_file():
        raise FileNotFoundError(f"Frozen CPOR implementation not found: {source}")
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("_cpor_frozen_core", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _CORE = module
    return module


class CPOR:
    """Public CPOR interface; all numerical operations delegate unchanged.

    S is the nonnegative aggregation matrix; v contains fixed node variances.
    issue() retains the frozen driver's release/feedback signature.
    """
    method_name = METHOD_NAME
    full_name = FULL_NAME

    def __init__(self, S, v, *, project_root=None):
        self._actor = load_core(project_root).Actor(S, v, storage_key())

    def __getattr__(self, name):
        return getattr(self._actor, name)

    def issue(self, *args, **kwargs):
        return self._actor.issue(*args, **kwargs)


def replay(args):
    """Replay one cached configuration, writing CPOR-named artifacts."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    root = args.project_root.resolve()
    dest = args.out.resolve()
    if not dest.is_relative_to(root):
        raise ValueError("Replay output must remain inside the supplied project root.")
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Use a fresh or empty replay output directory.")
    core = load_core(root)
    cell = core.CELLS[args.cell]
    paths = core.original.paths(cell)
    x = dict(np.load(paths["endpoint"], allow_pickle=True))
    bank = np.load(paths["bank"])["bank"]
    saved = dict(np.load(paths["target"]))
    targets, experts = core.make_contexts(x, bank, saved, core.original)
    with threadpool_limits(1):
        actor = CPOR(x["S"], x["v"], project_root=root)
        for t in range(36):
            actor.issue(t, x["b"][t], x["ub"][t], targets[t], experts[t],
                        None if t == 0 else x["truth"][t-1], calibration=True)
        rows, logs = [], []
        for t in range(36, 108):
            row = actor.issue(t, x["b"][t], x["ub"][t], targets[t], experts[t],
                              x["truth"][t-1], reset=t == 36)
            logs.append(row.pop("fit"))
            rows.append(row)
        output = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
        checks = core.original.audit(x, output, 1)
    original = Path(core.__file__).parent / "results" / Path(*cell) / "42/C0.npz"
    old = np.load(original)["pred"]
    report = dict(method=METHOD_NAME, full_name=FULL_NAME, archived_key=LEGACY_STORAGE_KEY,
                  cell=cell, checks=checks,
                  max_abs_replay_difference=float(np.max(np.abs(old-output["pred"]))),
                  mean_node_rmse=float(np.sqrt(np.mean(
                      (output["pred"]-x["truth"][36:])**2, axis=0)).mean()),
                  solver_failures=int(output["solver_failed"].sum()), logs=logs)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "CPOR.npz", **output)
    (dest / "CPOR_REPLAY_RESULT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "logs"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=f"{METHOD_NAME}: {FULL_NAME}")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-results", help="Check supplied aggregate results; no fitting.")
    check.add_argument("--evidence", type=Path, required=True)
    audit = sub.add_parser("audit-ledger", help="Audit saved forecasts; requires archived inputs.")
    audit.add_argument("--evidence", type=Path, required=True)
    audit.add_argument("--project-root", type=Path, required=True)
    play = sub.add_parser("replay", help="Replay a single cached configuration with CPOR outputs.")
    play.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    play.add_argument("--cell", type=int, default=0)
    play.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        replay(args)
    else:
        script = args.evidence / ("check_results.py" if args.command == "check-results" else "audit_ledger.py")
        sys.argv = [str(script)] + ([] if args.command == "check-results" else
                                   ["--project-root", str(args.project_root)])
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
