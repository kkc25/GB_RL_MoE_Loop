"""Read-only CPU router artifact supervision diagnostics, not deployment evaluation."""
from __future__ import annotations
import argparse
import json
from mixtral_lab.router_artifact_diagnostics import diagnose_router_artifacts


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('tune-report','checkpoint','table','output-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--near-tie-tolerance', type=float, default=1e-6)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = diagnose_router_artifacts(args.tune_report,args.checkpoint,args.table,args.output_dir,
        batch_size=args.batch_size,near_tie_tolerance=args.near_tie_tolerance)
    print(json.dumps({'status':result['status'],'label':result['label'],
        'checkpoint_update':result['checkpoint_update'],'output_dir':args.output_dir},allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
