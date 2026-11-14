"""Verified folder profiles and work-directory exclusion."""
import hashlib
import json
from pathlib import Path
import pytest
from coronahdr.folder_import import discover, match_profile, related_folders


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def case(tmp_path):
    lights, bias, source, work = [tmp_path/name for name in ('Lights', 'Bias', 'source', 'CoronaHDR_Work')]
    for folder in (lights, bias, source, work):
        folder.mkdir()
    light = lights/'a.NEF'
    light.write_bytes(b'light')
    calibration = bias/'bias.NEF'
    calibration.write_bytes(b'bias')
    report = dict(inputs=[dict(metadata=dict(source_sha256=digest(light)))],
        calibration=dict(bias=[dict(sources=[dict(metadata=dict(source_sha256=digest(calibration)))])]))
    (source/'report.json').write_text(json.dumps(report))
    return dict(light=str(lights), bias=str(bias)), source, work


def test_matching_uses_content_and_roles(case):
    folders, source, work = case
    selected, audit = match_profile(folders, source, work)
    assert selected == [0]
    assert len(audit['bias']) == 1
    path = Path(folders['light'])/'a.NEF'
    path.rename(path.with_name('renamed.nef'))
    assert match_profile(folders, source, work)[0] == [0]


def test_changed_missing_and_duplicate_files_are_rejected(case):
    folders, source, work = case
    with pytest.raises(ValueError, match='selected bias'):
        match_profile(dict(light=folders['light']), source, work)
    lights = Path(folders['light'])
    duplicate = lights/'duplicate.nef'
    duplicate.write_bytes(b'light')
    with pytest.raises(ValueError, match='Duplicate'):
        match_profile(folders, source, work)
    duplicate.write_bytes(b'unknown')
    with pytest.raises(ValueError, match='No validated automatic profile'):
        match_profile(folders, source, work)


@pytest.mark.parametrize('roles', [('dark',), ('dark_flat',), ('dark', 'dark_flat')])
@pytest.mark.parametrize('keep_bias', [False, True])
def test_validated_bias_is_accepted_in_dark_slots_without_duplicate_subtraction(case, roles, keep_bias):
    folders, source, work = case
    selected = dict(folders) if keep_bias else dict(light=folders['light'])
    selected.update({role: folders['bias'] for role in roles})
    indices, audit = match_profile(selected, source, work)
    assert indices == [0]
    for role in roles:
        assert audit[role][0]['effective_role'] == 'bias'
        assert audit[role][0]['selected_role'] == role


def test_discovery_excludes_work_and_links(case, tmp_path):
    folders, source, work = case
    (work/'generated.tiff').write_bytes(b'generated')
    (tmp_path/'linked').symlink_to(work, target_is_directory=True)
    jobs = tmp_path/'.coronahdr-jobs'/'old-job'
    jobs.mkdir(parents=True)
    (jobs/'generated.tiff').write_bytes(b'generated')
    assert work/'generated.tiff' not in discover(tmp_path, work)
    assert not any('linked' in str(path) for path in discover(tmp_path, work))
    assert not any('.coronahdr-jobs' in path.parts for path in discover(tmp_path, work))
    with pytest.raises(ValueError, match='outside the working directory'):
        discover(work, work)


def test_related_folders_find_nested_candidates_and_exclude_work_and_links(tmp_path):
    lights = tmp_path/'Lights'
    lights.mkdir()
    for name in ('Bias', 'Calibrations/Flats', 'Calibrations/Dark-Flats', 'old/Bias', 'CoronaHDR_Work/Flats'):
        (tmp_path/name).mkdir(parents=True, exist_ok=True)
    (tmp_path/'linked').symlink_to(tmp_path/'Calibrations', target_is_directory=True)
    found = related_folders(lights, tmp_path/'CoronaHDR_Work')
    assert found['bias'] == [tmp_path/'Bias', tmp_path/'old/Bias']
    assert found['flat'] == [tmp_path/'Calibrations/Flats']
    assert found['dark_flat'] == [tmp_path/'Calibrations/Dark-Flats']
    assert found['dark'] == []


@pytest.mark.parametrize('mode', ['aesthetic', 'scientific'])
@pytest.mark.parametrize('use_alias', [False, True])
@pytest.mark.parametrize('alignment', ['conservative', 'ael_visual'])
def test_folder_mode_dispatch_and_diagnostic_warnings(case, tmp_path, monkeypatch, mode, use_alias, alignment):
    if mode == 'scientific' and alignment == 'ael_visual':
        pytest.skip('AEL visual alignment is available only for aesthetic mode')
    import numpy as np
    from astropy.io import fits
    from coronahdr.folder_import import run_folders
    import coronahdr.folder_import as module
    folders, source, work = case
    original = json.loads((source/'report.json').read_text())
    original.update(recipe=dict(reference=0, method='weighted', frames=[]), reference=0)
    original['inputs'] = [dict(path=f'frame{i}.NEF', metadata=dict(source_sha256=str(i))) for i in range(3)]
    (source/'report.json').write_text(json.dumps(original))
    review = tmp_path/'review.json'
    review.write_text(json.dumps(dict(matrices_yx={'0': np.eye(3).tolist(), '1': np.eye(3).tolist()})))
    audit = {'light': [], 'dark': [dict(effective_role='bias', selected_role='dark')] if use_alias else []}
    original['rejected_frames'] = [dict(index=2, reason='Reference-level radiometric disagreement',
                                        diagnostic=dict(raw_relative_error=.23, raw_regional_relative_errors=[.25]))]
    (source/'report.json').write_text(json.dumps(original))
    selections = []
    def select(rows):
        selections.extend(rows)
        return [row['index'] for row in rows if row['status'] == 'accepted']
    monkeypatch.setattr('coronahdr.decisions.select_qualified_frames', select)
    monkeypatch.setattr(module, 'match_profile', lambda *args: ([0, 1, 2], audit))
    if use_alias:
        folders = dict(light=folders['light'], dark=folders['bias'])
    result_path = work/'result'
    calls = []

    def export(output, accepted):
        output = Path(output)
        output.mkdir()
        (output/'master.fits').write_bytes(b'fits')
        (output/'master.tiff').write_bytes(b'tiff')
        fits.writeto(output/'support.fits', np.array([[1, 0]], dtype='uint8'))
        fits.writeto(output/'unresolved_saturation.fits', np.array([[0, 1]], dtype='uint8'))
        return dict(accepted_indices=accepted, warnings=['Known detector limitation'])

    def aesthetic(source, review, output, indices, work_directory):
        assert work_directory == work/'work'
        calls.append(('aesthetic', indices))
        return export(output, indices)

    def scientific(recipe_path, output):
        recipe = json.loads(Path(recipe_path).read_text())
        assert recipe['method'] == 'weighted'
        assert recipe['processing']['include_indices'] == [0, 1]
        assert recipe['processing']['rejection_policy'] == 'exclude'
        assert recipe['processing']['decision_stage'] == 'Scientific HDR'
        assert 'dark' not in recipe.get('calibration', {})
        calls.append(('scientific', [0]))
        return export(output, [0])

    monkeypatch.setattr('coronahdr.aesthetic_pipeline.build_aesthetic_hdr', aesthetic)
    monkeypatch.setattr('coronahdr.pipeline.run_manifest', scientific)
    def defects(source, review, indices, path, mode, *, work_directory):
        assert Path(work_directory) == work
        assert path == work/'defect-profile'
        return Path(source), dict(count=1, unconfirmed_motion_count=0, applied=True, mode=mode)

    monkeypatch.setattr('coronahdr.defects.build_profile', defects)
    def intermediates(source, result, output, indices, stages, warn):
        assert indices == [0, 1, 2] and stages == ['calibrated']
        warn('Intermediate disk space notice')
        calls.append(('intermediates', stages))
        return dict(records=[dict(stage='calibrated')])

    monkeypatch.setattr('coronahdr.intermediates.export_intermediates', intermediates)
    job = dict(folders=folders, profile_source=str(source), profile_review=str(review), work=str(work),
               result=str(result_path), hdr_mode=mode, bias_fallback=not use_alias, save_intermediates=['calibrated'],
               detect_hot_pixels=use_alias, output_base_name='Eclipse', alignment=alignment,
               ael_visual_review=str(review) if alignment == 'ael_visual' else None)
    path = work/'job.json'
    path.write_text(json.dumps(job))
    run_folders(path)
    updated = json.loads(path.read_text())
    report = json.loads((result_path/'report.json').read_text())
    assert calls[0][0] == mode
    assert any('thermal dark signal' in warning for warning in updated['warnings'])
    assert any('saturated' in warning for warning in updated['warnings'])
    assert any('without valid observations' in warning for warning in updated['warnings'])
    if alignment == 'conservative':
        assert selections[2]['reason'] == 'Reference-level radiometric disagreement'
        assert selections[2]['measurements']['median_error'] == .23
        assert not any('unreliable geometries' in warning for warning in updated['warnings'])
    if alignment == 'ael_visual':
        assert any('uses 2 of 3 selected frames' in warning and 'lunar limb' in warning for warning in updated['warnings'][-4:])
        assert any('uses 2 of 3 selected frames' in warning for warning in report['warnings'])
    assert report['folder_processing']['bias_fallback'] is True
    assert (result_path/f'Eclipse_{mode}.fits').read_bytes() == b'fits'
    assert report['folder_processing']['named_masters'] == [f'Eclipse_{mode}.fits', f'Eclipse_{mode}.tiff']
    assert updated['bias_folder_aliases'] == (['dark'] if use_alias else [])
    if use_alias:
        assert report['defect_detection']['mode'] == mode
        assert report['defect_detection']['applied']
    assert report['intermediates']['records'] == [dict(stage='calibrated')]
    assert 'Intermediate disk space notice' in report['warnings']
    assert 'Intermediate disk space notice' in updated['warnings']
    assert updated['rejected_indices'] == ([2] if mode == 'aesthetic' else [1, 2])
    assert json.loads((source/'report.json').read_text()) == original


def test_conservative_selection_controls_defect_detection_and_stack(case, tmp_path, monkeypatch):
    import numpy as np
    from coronahdr.folder_import import run_folders
    folders, source, work = case
    report = json.loads((source/'report.json').read_text())
    report.update(reference=0, recipe=dict(reference=0), inputs=[
        dict(path=f'frame{i}.NEF', metadata=dict(source_sha256=str(i))) for i in range(3)])
    (source/'report.json').write_text(json.dumps(report))
    review = tmp_path/'review.json'
    review.write_text(json.dumps(dict(matrices_yx={str(i): np.eye(3).tolist() for i in range(3)})))
    monkeypatch.setattr('coronahdr.folder_import.match_profile', lambda *args: ([0, 1, 2], {}))
    monkeypatch.setattr('coronahdr.decisions.select_qualified_frames', lambda rows: [0, 2])
    def defects(source, review, indices, path, mode, *, work_directory):
        assert indices == [0, 2]
        return Path(source), dict(count=0, unconfirmed_motion_count=0)
    monkeypatch.setattr('coronahdr.defects.build_profile', defects)
    def stack(source, review, output, *, indices, work_directory):
        assert indices == [0, 2]
        raise RuntimeError('selected frames reached stack')
    monkeypatch.setattr('coronahdr.aesthetic_pipeline.build_aesthetic_hdr', stack)
    path = work/'job.json'
    path.write_text(json.dumps(dict(folders=folders, profile_source=str(source), profile_review=str(review),
        work=str(work), result=str(work/'result'), hdr_mode='aesthetic', detect_hot_pixels=True)))
    with pytest.raises(RuntimeError, match='selected frames reached stack'):
        run_folders(path)
    assert json.loads(path.read_text())['defect_selection'] == [0, 2]


def test_saved_registration_offers_wider_retry_from_cached_preparation(case, monkeypatch):
    from coronahdr.folder_import import run_folders
    folders, source, work = case
    report = json.loads((source/'report.json').read_text())
    report.update(recipe=dict(reference=0, frames=[], calibration={}, max_shift=256, processing={}),
                  output=dict(shape_yx=[3863, 5716]),
                  rejected_frames=[dict(index=1, reason='Registration exceeds max_shift')])
    (source/'report.json').write_text(json.dumps(report))
    job = dict(folders=folders, profile_source=str(source), profile_review='saved-review.json',
               work=str(work), result=str(work/'result'), hdr_mode='aesthetic')
    path = work/'job.json'
    path.write_text(json.dumps(job))
    monkeypatch.setattr('coronahdr.folder_import.match_profile', lambda *args: ([0, 1], {}))
    prompts = []
    def choose(message, choices, default):
        prompts.append(message)
        return choices[0]
    monkeypatch.setattr('coronahdr.decisions.ask', choose)
    def run(manifest, output):
        recipe = json.loads(manifest.read_text())
        assert recipe['max_shift'] == 5716
        assert recipe['processing']['cache_from'] == str(source)
        assert recipe['processing']['include_indices'] == [0, 1]
        assert not output.exists()
        raise RuntimeError('Cached registration retry reached')
    monkeypatch.setattr('coronahdr.pipeline.run_manifest', run)
    with pytest.raises(RuntimeError, match='Cached registration retry reached'):
        run_folders(path)
    assert len(prompts) == 1 and 'Saved registration excluded' in prompts[0]


def test_visual_folder_job_builds_separate_ael_geometry(case, monkeypatch):
    from coronahdr.folder_import import run_folders
    folders, source, work = case
    report = json.loads((source/'report.json').read_text())
    report.update(recipe=dict(reference=0, method='weighted', frames=[]), reference=0)
    (source/'report.json').write_text(json.dumps(report))
    job = dict(folders=folders, profile_source=str(source), profile_review='original-review.json',
               work=str(work), result=str(work/'result'), hdr_mode='aesthetic', alignment='ael_visual')
    path = work/'job.json'
    path.write_text(json.dumps(job))
    monkeypatch.setattr('coronahdr.folder_import.match_profile', lambda *args: ([0], {}))
    def build(source_path, review_path, *, indices):
        assert source_path == str(source)
        assert indices == [0]
        assert review_path.parent == work
        assert review_path.name.startswith('ael-visual-geometry-')
        raise RuntimeError('AEL visual geometry reached')
    monkeypatch.setattr('coronahdr.ael_visual.write_review', build)
    with pytest.raises(RuntimeError, match='AEL visual geometry reached'):
        run_folders(path)


def test_both_modes_keep_scientific_geometry_when_visual_alignment_selected(case, monkeypatch):
    import coronahdr.folder_import as module
    folders, source, work = case
    result = work/'result'
    path = work/'job.json'
    job = dict(folders=folders, profile_source=str(source), profile_review='conservative.json',
               work=str(work), result=str(result), hdr_mode='both', alignment='ael_visual')
    path.write_text(json.dumps(job))
    seen = []
    def run_child(child_path):
        child = json.loads(child_path.read_text())
        seen.append((child['hdr_mode'], child['alignment'], child['profile_review']))
        if child['hdr_mode'] == 'aesthetic':
            child['profile_review'] = child['ael_visual_review'] = 'ael-visual.json'
        child_path.write_text(json.dumps(child))
        output = Path(child['result']); output.mkdir()
        (output/'report.json').write_text(json.dumps(dict(accepted_indices=[0],warnings=[])))
        for name in ('master.fits','master.tiff'):
            (output/name).write_bytes(b'fixture')
    monkeypatch.setattr(module, 'run_folders', run_child)
    module.run_both_folders(path, job)
    assert seen == [('scientific','conservative','conservative.json'),
                    ('aesthetic','ael_visual','conservative.json')]
    saved = json.loads(path.read_text())
    assert saved['profile_review'] == 'conservative.json'
    assert saved['ael_visual_review'] == 'ael-visual.json'


@pytest.mark.parametrize('detect', [False, True])
@pytest.mark.parametrize('standalone', [False, True])
def test_both_modes_share_geometry_but_keep_distinct_defect_policies(case, tmp_path, monkeypatch, detect, standalone):
    import numpy as np
    from astropy.io import fits
    from coronahdr.folder_import import run_folders
    from coronahdr.desktop_jobs import publish_job, is_published
    folders, source, work = case
    original = json.loads((source/'report.json').read_text())
    original.update(recipe=dict(reference=0, method='weighted', frames=[]), reference=0)
    original['inputs'] = [dict(path=f'frame{i}.NEF', metadata=dict(source_sha256=str(i))) for i in range(2)]
    (source/'report.json').write_text(json.dumps(original))
    review = tmp_path/'review.json'
    review.write_text(json.dumps(dict(matrices_yx={'0': np.eye(3).tolist(), '1': np.eye(3).tolist()})))
    monkeypatch.setattr('coronahdr.folder_import.match_profile', lambda *args: ([0, 1], {}))
    calls = []

    def defects(profile, geometry, indices, destination, mode, *, work_directory):
        assert Path(profile) == source
        destination.mkdir()
        (destination/'report.json').write_text(json.dumps(original))
        calls.append(('detect', mode))
        return destination, dict(count=1, unconfirmed_motion_count=0, applied=True, mode=mode)

    def result(output, accepted):
        output = Path(output)
        output.mkdir()
        for name in ('master.fits', 'master.tiff'):
            (output/name).write_bytes(output.name.encode())
        fits.writeto(output/'support.fits', np.ones((1, 1), dtype='uint8'))
        fits.writeto(output/'unresolved_saturation.fits', np.zeros((1, 1), dtype='uint8'))
        return dict(accepted_indices=accepted, warnings=[])

    def scientific(recipe_path, output):
        recipe = json.loads(Path(recipe_path).read_text())
        assert recipe['processing']['cache_from'] == (str(work/'scientific/defect-profile') if detect else str(source))
        calls.append(('stack', 'scientific'))
        return result(output, [0])

    def aesthetic(profile, geometry, output, indices, work_directory):
        assert str(profile) == (str(work/'aesthetic/defect-profile') if detect else str(source))
        calls.append(('stack', 'aesthetic'))
        return result(output, indices)

    monkeypatch.setattr('coronahdr.defects.build_profile', defects)
    monkeypatch.setattr('coronahdr.pipeline.run_manifest', scientific)
    monkeypatch.setattr('coronahdr.aesthetic_pipeline.build_aesthetic_hdr', aesthetic)
    job = dict(kind='folders', folders=folders, profile_source=str(source), profile_review=str(review),
               work=str(work), destination=str(tmp_path/'published'), result=str(work/'result'),
               hdr_mode='both', detect_hot_pixels=detect, output_base_name='Eclipse')
    if standalone:
        job['publication'] = 'standalone_final_files_v1'
    path = work/'job.json'
    path.write_text(json.dumps(job))
    run_folders(path)
    assert calls == ([('detect', 'scientific'), ('stack', 'scientific'),
                      ('detect', 'aesthetic'), ('stack', 'aesthetic')] if detect else
                     [('stack', 'scientific'), ('stack', 'aesthetic')])
    if not standalone:
        assert (work/'result/Eclipse_scientific.tiff').read_bytes() == b'scientific'
        assert (work/'result/Eclipse_aesthetic.fits').read_bytes() == b'aesthetic'
    else:
        assert not list((work/'result').glob('Eclipse_*'))
    report = json.loads((work/'result/report.json').read_text())
    assert report['results']['scientific']['accepted_indices'] == [0]
    assert report['results']['aesthetic']['accepted_indices'] == [0, 1]
    if detect:
        assert report['results']['scientific']['defect_detection']['mode'] == 'scientific'
        assert report['results']['aesthetic']['defect_detection']['mode'] == 'aesthetic'
    publish_job(path)
    assert is_published(json.loads(path.read_text()))
    assert (tmp_path/'published/Eclipse_aesthetic.tiff').read_bytes() == b'aesthetic'
    if standalone:
        assert {p.name for p in (tmp_path/'published').iterdir()} == {
            f'Eclipse_{mode}.{ext}' for mode in ('scientific', 'aesthetic') for ext in ('fits', 'tiff')}


def test_both_mode_does_not_publish_if_one_result_is_missing(tmp_path):
    from coronahdr.desktop_jobs import publish_job
    result = tmp_path/'result'
    result.mkdir()
    (result/'report.json').write_text('{}')
    path = tmp_path/'job.json'
    path.write_text(json.dumps(dict(result=str(result), destination=str(tmp_path/'published'), hdr_mode='both')))
    with pytest.raises(ValueError, match='Incomplete job'):
        publish_job(path)
    assert not (tmp_path/'published').exists()


def test_progress_stage_is_only_emitted_for_interactive_jobs(monkeypatch, capsys):
    from coronahdr.folder_import import progress_stage
    monkeypatch.delenv('CORONAHDR_INTERACTIVE', raising=False)
    progress_stage('prepare')
    assert not capsys.readouterr().out
    monkeypatch.setenv('CORONAHDR_INTERACTIVE', '1')
    progress_stage('aesthetic')
    assert capsys.readouterr().out == 'CORONAHDR_STAGE aesthetic\n'
