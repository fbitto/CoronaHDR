"""AEL-style visual reviews stay separate from conservative scientific jobs."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coronahdr.ael_visual import write_review
from coronahdr.rigid import reviewed_rigid
from coronahdr.streaming import CachedFrame


@pytest.mark.parametrize('include_candidate', [False, True])
def test_visual_review_keeps_valid_geometry_and_records_rejected_candidates(tmp_path, monkeypatch, include_candidate):
    import coronahdr.ael_visual as module
    inputs = [dict(path=f'frame{i}', metadata=dict(source_sha256=str(i), lunar_disk=[32.,32.,8.])) for i in range(3)]
    report = dict(inputs=inputs, reference=0, recipe=dict(processing=dict(registration_bin=4,
        registration_roi_xywh=[0,0,64,64])))
    (tmp_path/'report.json').write_text(json.dumps(report))
    frames = [CachedFrame(np.full((64,64,3), i, dtype=float), dict(state='linear_rgb'), Path(f'frame{i}')) for i in range(3)]
    monkeypatch.setattr(module, 'load_cached_run', lambda *args: (frames, inputs, {}))
    monkeypatch.setattr(module, 'registration_auxiliary', lambda frame, *args: SimpleNamespace(data=frame.data))
    monkeypatch.setattr(module, 'release_pages', lambda frame: None)
    monkeypatch.setattr(module, 'valid_mask', lambda frame: np.ones(frame.data.shape, bool))
    monkeypatch.setattr(module, 'check_rigid_agreement', lambda *args: dict(passed=True, disagreement_native_px=.1))
    prompts = []
    def decide(rows):
        prompts.extend(rows)
        return [2] if include_candidate else []
    monkeypatch.setattr(module, 'review_frames', decide)
    def register(reference, moving, *args, **kwargs):
        index = int(moving[0,0])
        return np.array([[1.,0.,index],[0.,1.,0.]]), dict(status='candidate' if index==2 else 'registered',
            heldout_correlation=.5 if index==2 else .99, angle_deg=0., ecc=.99, qualification='test')
    monkeypatch.setattr(module, 'register', register)
    path = tmp_path/'geometry.json'
    write_review(tmp_path,path)
    review = json.loads(path.read_text())
    assert review['alignment'] == 'ael_visual'
    assert set(review['matrices_yx']) == ({'0','1','2'} if include_candidate else {'0','1'})
    assert len(prompts) == 3 and prompts[2]['filename'] == 'frame2' and '0.500' in prompts[2]['reason']
    assert prompts[1]['measurements']['heldout_correlation'] == .99
    assert prompts[1]['measurements']['independent_check']['disagreement_native_px'] == .1
    assert prompts[0]['reference'] is True
    if include_candidate:
        assert review['metrics']['2']['user_decision'] == 'Include experimental frame'
    else:
        assert review['rejected_frames'][0]['index'] == 2
        assert review['rejected_frames'][0]['quality']['heldout_correlation'] == .5
        assert 'candidate_matrix_yx' in review['rejected_frames'][0]
    poses, _, _, _, _ = reviewed_rigid(path, inputs, 0, [0,1])
    np.testing.assert_allclose(poses[1][:2,2], [0.,-1.])
    with pytest.raises(FileExistsError):
        write_review(tmp_path,path)


def test_visual_alignment_is_restricted_to_aesthetic_mode(tmp_path):
    from coronahdr.desktop_jobs import create_folder_job
    lights = tmp_path/'Lights'; lights.mkdir()
    with pytest.raises(ValueError,match='only for aesthetic'):
        create_folder_job({'light':str(lights)},mode='scientific',alignment='ael_visual',temporary=tmp_path/'work')


def test_frame_review_defaults_to_excluding_candidates_and_checks_reply(monkeypatch, capsys):
    import builtins
    from coronahdr.decisions import review_frames
    rows = [dict(index=0, status='accepted'), dict(index=1, status='candidate')]
    monkeypatch.delenv('CORONAHDR_INTERACTIVE', raising=False)
    assert review_frames(rows) == []
    monkeypatch.setenv('CORONAHDR_INTERACTIVE', '1')
    monkeypatch.setattr(builtins, 'input', lambda: '[0]')
    with pytest.raises(ValueError, match='invalid frame review'):
        review_frames(rows)
    assert 'CORONAHDR_FRAME_REVIEW' in capsys.readouterr().out


def test_conservative_selection_cannot_restore_rejected_or_remove_reference(monkeypatch):
    import builtins
    from coronahdr.decisions import select_qualified_frames
    rows = [dict(index=0, status='accepted', reference=True), dict(index=1, status='accepted'),
            dict(index=2, status='rejected')]
    monkeypatch.delenv('CORONAHDR_INTERACTIVE', raising=False)
    assert select_qualified_frames(rows) == [0, 1]
    monkeypatch.setenv('CORONAHDR_INTERACTIVE', '1')
    for reply in ('[1]', '[0, 2]', 'null'):
        monkeypatch.setattr(builtins, 'input', lambda: reply)
        with pytest.raises(ValueError, match='invalid frame selection'):
            select_qualified_frames(rows)


def test_independent_rigid_agreement_uses_native_pixel_threshold(monkeypatch):
    import coronahdr.ael_visual as module
    reference = SimpleNamespace(data=np.ones((64, 64)))
    roi = [0, 0, 256, 256]
    identity = np.eye(3)
    monkeypatch.setattr(module, 'measure_rigid', lambda *args: (identity, {'split_control_error_px': .2}))
    assert module.check_rigid_agreement(reference, reference, identity, 4, roi)['passed'] is True
    shifted = identity.copy()
    shifted[0, 2] = 1.01
    result = module.check_rigid_agreement(reference, reference, shifted, 4, roi)
    assert result['passed'] is False and result['disagreement_native_px'] > 1.


def test_failed_independent_check_becomes_candidate(tmp_path, monkeypatch):
    import coronahdr.ael_visual as module
    inputs = [dict(path=f'frame{i}', metadata=dict(source_sha256=str(i), lunar_disk=[32.,32.,8.])) for i in range(3)]
    report = dict(inputs=inputs, reference=0, recipe=dict(processing=dict(registration_bin=4, registration_roi_xywh=[0,0,64,64])))
    (tmp_path/'report.json').write_text(json.dumps(report))
    frames = [CachedFrame(np.full((64,64,3), i, dtype=float), dict(state='linear_rgb'), Path(f'frame{i}')) for i in range(3)]
    monkeypatch.setattr(module, 'load_cached_run', lambda *args: (frames, inputs, {}))
    monkeypatch.setattr(module, 'registration_auxiliary', lambda frame, *args: SimpleNamespace(data=frame.data))
    monkeypatch.setattr(module, 'release_pages', lambda frame: None)
    monkeypatch.setattr(module, 'valid_mask', lambda frame: np.ones(frame.data.shape, bool))
    monkeypatch.setattr(module, 'register', lambda *args, **kwargs: (np.eye(2,3), dict(status='registered', heldout_correlation=.9, angle_deg=0.)))
    monkeypatch.setattr(module, 'check_rigid_agreement', lambda ref, moving, *args: dict(passed=bool(moving.data[0,0] != 2), disagreement_native_px=2.))
    rows = []
    monkeypatch.setattr(module, 'review_frames', lambda value: rows.extend(value) or [])
    path = tmp_path/'review.json'
    module.write_review(tmp_path, path)
    assert rows[2]['status'] == 'candidate' and '2.000 native pixels' in rows[2]['reason']
    review = json.loads(path.read_text())
    assert set(review['matrices_yx']) == {'0','1'}
    assert review['rejected_frames'][0]['quality']['independent_check']['passed'] is False
