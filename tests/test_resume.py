"""Discover and continue folder jobs without replacing their old data."""
import fcntl
import json
from pathlib import Path
import pytest
from coronahdr.desktop_jobs import create_folder_job, unfinished_jobs, resume_folder_job, publish_job


def folder_job(tmp_path, monkeypatch):
    import coronahdr.desktop_jobs as module
    monkeypatch.setattr(module, '__file__', str(tmp_path/'project/src/coronahdr/desktop_jobs.py'))
    lights = tmp_path/'Lights'
    lights.mkdir()
    (lights/'light.NEF').write_bytes(b'fixture')
    return create_folder_job({'light': lights}, temporary=tmp_path, destination=tmp_path/'output')


def test_discovery_and_resume_preserve_previous_attempt(tmp_path, monkeypatch):
    path, job = folder_job(tmp_path, monkeypatch)
    partial = Path(job['result'])
    partial.mkdir()
    (partial/'unfinished.tiff').write_bytes(b'keep')
    acquisition = path.parent/'acquisition/prepared/preparation'
    acquisition.mkdir(parents=True)
    (acquisition/'report.json').write_text('{}')
    broken = path.parent.parent/'broken'
    broken.mkdir()
    (broken/'job.json').write_text('{')
    assert [p for p, _ in unfinished_jobs(tmp_path)] == [path]
    resumed, updated = resume_folder_job(path)
    assert updated['previous_result'] == job['result']
    assert updated['result'] != job['result'] and updated['work'] != job['work']
    assert updated['resume_acquisition'] == str(path.parent/'acquisition')
    assert (partial/'unfinished.tiff').read_bytes() == b'keep'
    assert json.loads(path.read_text())['status'] == 'resumed'
    assert [p for p, _ in unfinished_jobs(tmp_path)] == [resumed]
    second, again = resume_folder_job(resumed)
    assert again['resume_acquisition'] == updated['resume_acquisition']
    with pytest.raises(ValueError, match='unfinished'):
        resume_folder_job(path)


def test_active_jobs_are_not_resumed(tmp_path, monkeypatch):
    path, job = folder_job(tmp_path, monkeypatch)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert unfinished_jobs(tmp_path) == []
        with pytest.raises(ValueError, match='still processing'):
            resume_folder_job(path)
    assert len(unfinished_jobs(tmp_path)) == 1


def test_both_mode_resume_keeps_shared_acquisition_and_uses_new_result(tmp_path, monkeypatch):
    from coronahdr.folder_import import run_both_folders
    import coronahdr.folder_import as module
    path, job = folder_job(tmp_path, monkeypatch)
    job.update(hdr_mode='both', output_base_name='Eclipse', detect_hot_pixels=True)
    path.write_text(json.dumps(job))
    source = path.parent/'original-preparation'
    source.mkdir()
    review = path.parent/'geometry.json'
    review.write_text('{}')
    calls = []

    def child_run(child_path):
        child = json.loads(child_path.read_text())
        calls.append((child['hdr_mode'], child.get('profile_source'), child.get('_shared_geometry')))
        if child['hdr_mode'] == 'aesthetic':
            raise RuntimeError('Interrupted aesthetic stage')
        output = Path(child['result'])
        output.mkdir()
        (output/'report.json').write_text(json.dumps(dict(accepted_indices=[0], warnings=[])))
        child.update(profile_source=str(source), profile_review=str(review))
        child_path.write_text(json.dumps(child))

    monkeypatch.setattr(module, 'run_folders', child_run)
    with pytest.raises(RuntimeError, match='Interrupted'):
        run_both_folders(path, job)
    saved = json.loads(path.read_text())
    assert saved['profile_source'] == str(source)
    assert calls == [('scientific', None, False), ('aesthetic', str(source), True)]
    resumed, updated = resume_folder_job(path)
    assert updated['profile_source'] == str(source)
    assert updated['result'] != job['result']
    assert updated['output_base_name'] == 'Eclipse'
    assert (Path(job['result'])/'scientific/report.json').is_file()


def test_completed_job_can_be_revised_without_replacing_its_output(tmp_path, monkeypatch):
    path, job = folder_job(tmp_path, monkeypatch)
    job['status'] = 'completed'
    path.write_text(json.dumps(job))
    assert unfinished_jobs(tmp_path) == []
    assert len(unfinished_jobs(tmp_path, include_completed=True)) == 1
    with pytest.raises(ValueError):
        resume_folder_job(path)
    resumed, revised = resume_folder_job(path, destination=tmp_path/'revised')
    assert revised['destination'] == str(tmp_path/'revised')
    assert revised['previous_result'] == job['result']
    assert json.loads(path.read_text())['status'] == 'completed'


def test_resume_completed_hdr_only_verifies_and_publishes(tmp_path, monkeypatch):
    from coronahdr.folder_import import run_folders
    path, job = folder_job(tmp_path, monkeypatch)
    source = path.parent/'source'
    source.mkdir()
    recipe = dict(frames=[], calibration={})
    (source/'report.json').write_text(json.dumps(dict(recipe=recipe)))
    job['profile_source'] = str(source)
    path.write_text(json.dumps(job))
    result = Path(job['result'])
    result.mkdir()
    (result/'report.json').write_text(json.dumps(dict(folder_processing={})))
    (result/'master.fits').write_bytes(b'finished fits')
    (result/'master.tiff').write_bytes(b'finished tiff')
    verified = []
    monkeypatch.setattr('coronahdr.folder_import.match_profile', lambda *args: ([0], {}))
    monkeypatch.setattr('coronahdr.streaming.load_cached_run', lambda source, recipe: verified.append(source))
    def unexpected(*args, **kwargs):
        raise AssertionError('Completed HDR must not be rebuilt')
    monkeypatch.setattr('coronahdr.aesthetic_pipeline.build_aesthetic_hdr', unexpected)
    resumed, updated = resume_folder_job(path)
    run_folders(resumed)
    assert verified == [source]
    assert json.loads(resumed.read_text())['result'] == str(result)
    publish_job(resumed)
    assert (Path(job['destination'])/'CoronaHDR_aesthetic.tiff').read_bytes() == b'finished tiff'
    assert json.loads(resumed.read_text())['status'] == 'completed'


def test_resume_completed_pair_with_new_name_reuses_staged_results(tmp_path, monkeypatch):
    from coronahdr.folder_import import run_folders
    from coronahdr.desktop_jobs import OutputNameConflict
    path, job = folder_job(tmp_path, monkeypatch)
    source = path.parent/'source'
    source.mkdir()
    (source/'report.json').write_text(json.dumps(dict(recipe=dict(frames=[], calibration={}))))
    result = Path(job['result'])
    result.mkdir()
    (result/'report.json').write_text(json.dumps(dict(folder_processing=dict(hdr_mode='both'))))
    for mode in ('scientific', 'aesthetic'):
        folder = result/mode
        folder.mkdir()
        (folder/'report.json').write_text('{}')
        for ext in ('fits', 'tiff'):
            (folder/f'master.{ext}').write_bytes(f'{mode}-{ext}'.encode())
    destination = Path(job['destination'])
    destination.mkdir(exist_ok=True)
    existing = destination/'CoronaHDR_scientific.fits'
    existing.write_bytes(b'keep')
    job.update(hdr_mode='both', profile_source=str(source), status='failed')
    path.write_text(json.dumps(job))
    with pytest.raises(OutputNameConflict):
        resume_folder_job(path)
    resumed, _ = resume_folder_job(path, base_name='New')
    checked = []
    monkeypatch.setattr('coronahdr.streaming.load_cached_run', lambda source, recipe: checked.append(source))
    run_folders(resumed)
    assert checked == [source]
    assert Path(json.loads(resumed.read_text())['result']) == result
    publish_job(resumed)
    assert (destination/'New_scientific.fits').read_bytes() == b'scientific-fits'
    assert (destination/'New_aesthetic.tiff').read_bytes() == b'aesthetic-tiff'
    assert existing.read_bytes() == b'keep'
