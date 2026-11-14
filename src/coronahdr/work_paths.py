"""Default work storage without depending on a particular checkout or drive."""
from pathlib import Path
import tempfile


def project_artifacts():
    return Path(__file__).resolve().parents[2]/'artifacts'


def default_work_directory(artifacts=None):
    artifacts = Path(artifacts) if artifacts is not None else project_artifacts()
    if artifacts.is_symlink() or artifacts.is_dir():
        return artifacts.absolute()
    directory = Path(tempfile.gettempdir())/'CoronaHDR-work'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def work_exclusion_root(artifacts=None):
    artifacts = Path(artifacts) if artifacts is not None else project_artifacts()
    if artifacts.is_symlink():
        return artifacts.resolve().parent
    return default_work_directory(artifacts).resolve()
