from pathlib import Path

from coronahdr.work_paths import default_work_directory, work_exclusion_root


def test_fresh_checkout_uses_writable_temporary_directory(tmp_path, monkeypatch):
    project = tmp_path/'project'
    project.mkdir()
    monkeypatch.setattr('coronahdr.work_paths.tempfile.gettempdir', lambda: str(tmp_path))
    work = default_work_directory(project/'artifacts')
    assert work == tmp_path/'CoronaHDR-work' and work.is_dir()
    assert work_exclusion_root(project/'artifacts') == work.resolve()


def test_existing_external_artifacts_link_remains_the_default(tmp_path):
    project = tmp_path/'project'
    project.mkdir()
    work = tmp_path/'external'
    (work/'artifacts').mkdir(parents=True)
    link = project/'artifacts'
    link.symlink_to(work/'artifacts', target_is_directory=True)
    assert default_work_directory(link) == link.absolute()
    assert work_exclusion_root(link) == work.resolve()


def test_disconnected_external_link_is_not_silently_replaced(tmp_path):
    link = tmp_path/'artifacts'
    link.symlink_to(tmp_path/'disconnected'/'artifacts', target_is_directory=True)
    assert default_work_directory(link) == link.absolute()
    assert link.is_symlink() and not link.exists()
