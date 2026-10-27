"""Read-only checks protecting archived run evidence from fresh-run writes."""

from pathlib import Path


PREFLIGHT_FILES = frozenset({"resolved_config.json", "preflight.json", "environment.json"})


def guard_run_directory(path, *, resume=False, additional_allowed=()):
    """Allow a new/preflight-only directory, or an explicitly resumed run.

    Checkpoints are not committed, so their absence does not establish that a
    directory is unused. Never append a fresh run to archived logs/summaries.
    This check must precede any preparation writes; exact resume validation is
    still the responsibility of the checkpoint loader.
    """
    path = Path(path)
    if not path.exists():
        return
    if not path.is_dir():
        raise FileExistsError(f"run output is not a directory: {path}")
    if resume:
        return
    allowed = PREFLIGHT_FILES | frozenset(additional_allowed)
    occupied = sorted(item.name for item in path.iterdir() if item.name not in allowed or not item.is_file())
    if occupied:
        raise FileExistsError(
            f"{path} contains existing run evidence or unrelated files ({', '.join(occupied[:5])}); "
            "set output_dir to a fresh directory, or use --resume with the matching checkpoint"
        )
