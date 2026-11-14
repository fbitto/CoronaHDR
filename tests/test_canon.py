from io import BytesIO
import json
import struct

import numpy as np
import pytest
import tifffile

from coronahdr.raw import camera_exif, cr3_exif
from coronahdr.canon import canon_recipe, lunar_disk


def box(kind, payload, extended=False):
    return (struct.pack('>I4sQ', 1, kind, len(payload)+16) if extended
            else struct.pack('>I4s', len(payload)+8, kind))+payload


@pytest.fixture
def cr3_bytes():
    def tiff(tags):
        stream = BytesIO()
        tifffile.imwrite(stream, np.zeros((1, 1), np.uint8), metadata=None, extratags=tags)
        return stream.getvalue()
    identity = tiff([(271, 's', 0, 'Canon', False), (272, 's', 0, 'Canon EOS fixture', False)])
    exposure = tiff([(33434, '2I', 1, (1, 250), False), (33437, '2I', 1, (8, 1), False),
        (34855, 'H', 1, 800, False), (42033, 's', 0, '12345', False),
        (37386, '2I', 1, (250, 1), False)])
    canon = bytes.fromhex('85c0b687820f11e08111f4ce462b6a48')
    return box(b'ftyp', b'crx \0\0\0\1crx isom')+box(b'moov', box(b'uuid', canon+
        box(b'CMT1', identity)+box(b'CMT2', exposure), extended=True))+box(b'mdat', b'CMT1 not metadata')


def test_cr3_native_metadata_no_exiftool(tmp_path, cr3_bytes):
    path = tmp_path/'camera.CR3'
    path.write_bytes(cr3_bytes)
    metadata = camera_exif(path)
    assert metadata['exposure_s'] == 1/250
    assert metadata['aperture'] == 8 and metadata['iso'] == 800
    assert metadata['focal_length_mm'] == 250
    assert metadata['sensor_id'] == 'Canon EOS fixture serial 12345'
    assert metadata['gain'] is None and 'raw_exif_status' not in metadata


@pytest.mark.parametrize('payload', [b'', b'\x00\x00\x00\x04moov',
    b'\x00\x00\x00\x01moov\x00', b'\x00\x00\x00\xffmoov', box(b'mdat', b'CMT1fakeCMT2fake')])
def test_bad_cr3_metadata_is_not_invented(tmp_path, payload):
    path = tmp_path/'bad.cr3'
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        cr3_exif(path)
    metadata = camera_exif(path)
    assert 'raw_exif_status' in metadata and 'exposure_s' not in metadata
    assert metadata['gain'] is None


def eclipse():
    y, x = np.indices((768, 1024))
    radius = np.hypot(x-377.3, y-341.7)
    return np.where(radius < 65, 2048., 2048.+3000*np.exp(-(radius-65)/35))


def test_measured_disk_uses_native_centres_without_padding():
    disk, residual = lunar_disk(eclipse())
    np.testing.assert_allclose(disk[:2], [377.3, 341.7], atol=1.)
    assert 62 < disk[2] < 67 and residual < 2
    with pytest.raises(ValueError, match='No closed lunar limb'):
        lunar_disk(np.ones((512, 512)))


def test_canon_recipe_excludes_generated_tiffs_and_rejects_mixed_iso(tmp_path, monkeypatch):
    import coronahdr.canon as module
    lights, work = tmp_path/'lights', tmp_path/'CoronaHDR_Work'
    lights.mkdir(); work.mkdir()
    for name in ('a.CR3', 'b.cr3', 'old.tiff'):
        (lights/name).write_bytes(b'fixture')
    metadata = dict(sensor_id='Canon fixture serial 1', iso=800, aperture=8,
                    focal_length_mm=250, exposure_s=.1)
    monkeypatch.setattr(module, 'camera_exif', lambda path: dict(metadata))
    monkeypatch.setattr(module, 'decode_raw', lambda path: (eclipse(), {}))
    recipe = canon_recipe({'light': lights}, work)
    assert recipe['max_shift'] == 1024
    assert len(recipe['frames']) == 2 and not recipe['calibration']
    assert recipe['processing']['canvas'] == 'union'
    assert all('crop_xywh' not in item for item in recipe['frames'])
    assert all(item['overrides']['lunar_disk'] for item in recipe['frames'])
    for item in recipe['frames']:
        overrides = item['overrides']
        assert overrides['limb_margin_px'] == 8
        assert 70 < overrides['lunar_disk'][2]+overrides['limb_margin_px'] < 75
    monkeypatch.setattr(module, 'camera_exif', lambda path: dict(metadata, iso=400 if path.name=='b.cr3' else 800))
    with pytest.raises(ValueError, match='Incompatible Canon light'):
        canon_recipe({'light': lights}, work)


def test_canon_preparation_exports_hash_bound_translation_geometry(tmp_path, monkeypatch):
    import coronahdr.canon as module
    recipe = dict(processing=dict(registration_roi_xywh=[0, 0, 512, 512]))
    monkeypatch.setattr(module, 'canon_recipe', lambda *args: recipe)
    calls = []
    def run(manifest, source):
        calls.append(json.loads(manifest.read_text()))
        return dict(reference=0, inputs=[dict(metadata=dict(source_sha256='original'))],
                    transformations=[dict(index=0, shift_dy_dx=[2., -3.], path=[0])])
    monkeypatch.setattr('coronahdr.pipeline.run_manifest', run)
    source, review = module.prepare_canon({'light': 'lights'}, tmp_path, tmp_path/'run')
    geometry = json.loads(review.read_text())
    assert source == tmp_path/'run/prepared' and calls == [recipe]
    assert calls[0]['processing']['decision_stage'] == 'Preparation and registration'
    assert geometry['input_source_sha256'] == ['original']
    np.testing.assert_array_equal(np.array(geometry['matrices_yx']['0'])[:2, 2], [2., -3.])
    with pytest.raises(FileExistsError):
        module.prepare_canon({}, tmp_path, tmp_path/'run')


def test_canon_folder_job_does_not_require_nikon_profile(tmp_path, monkeypatch):
    import coronahdr.desktop_jobs as module
    project, external, lights = tmp_path/'project', tmp_path/'external', tmp_path/'lights'
    project.mkdir(); external.mkdir(); lights.mkdir()
    (external/'artifacts').mkdir()
    (project/'artifacts').symlink_to(external/'artifacts', target_is_directory=True)
    (lights/'light.CR3').write_bytes(b'fixture')
    monkeypatch.setattr(module, '__file__', str(project/'src/coronahdr/desktop_jobs.py'))
    path, job = module.create_folder_job({'light': str(lights)}, mode='scientific')
    assert path.is_file() and job['camera_family'] == 'Canon'
    assert job['profile_source'] is None and job['profile_review'] is None
    assert job['work'] == str(path.parent)
    assert path.is_relative_to(external)


def test_fresh_nikon_recipe_does_not_borrow_existing_geometry(tmp_path, monkeypatch):
    import coronahdr.canon as module
    lights, work = tmp_path/'lights', tmp_path/'temporary'
    lights.mkdir()
    work.mkdir()
    (lights/'a.NEF').write_bytes(b'fixture')
    metadata = dict(sensor_id='Nikon fixture serial 1', iso=800, aperture=8,
                    focal_length_mm=250, exposure_s=.1)
    monkeypatch.setattr(module, 'camera_exif', lambda path: dict(metadata))
    monkeypatch.setattr(module, 'decode_raw', lambda path: (eclipse(), {}))
    recipe = module.canon_recipe({'light': lights}, work, camera_family='Nikon')
    assert recipe['camera_family'] == 'Nikon'
    assert recipe['frames'][0]['overrides']['lunar_disk']
    assert 'cache_from' not in recipe['processing']
    assert 'rigid_review' not in recipe['processing']


def test_fresh_nikon_folder_job_without_archived_run(tmp_path, monkeypatch):
    import coronahdr.desktop_jobs as module
    lights, temporary = tmp_path/'Lights', tmp_path/'Temporary'
    lights.mkdir()
    temporary.mkdir()
    (lights/'light.NEF').write_bytes(b'fixture')
    monkeypatch.setattr(module, '__file__', str(tmp_path/'empty-project/src/coronahdr/desktop_jobs.py'))
    path, job = module.create_folder_job({'light': str(lights)}, temporary=temporary,
                                         destination=tmp_path/'output')
    assert job['camera_family'] == 'Nikon'
    assert job['profile_source'] is None and job['profile_review'] is None
    assert path.is_relative_to(temporary)
    import coronahdr.folder_import as folders
    def prepare(selected, work, output, camera_family):
        assert camera_family == 'Nikon'
        assert output.is_relative_to(temporary)
        assert Path(work).is_relative_to(temporary)
        source, review = output/'prepared', output/'geometry.json'
        source.mkdir(parents=True)
        review.write_text('{}')
        return source, review
    def stop_after_preparation(selected, source, work):
        assert Path(source).is_relative_to(temporary)
        updated = json.loads(path.read_text())
        assert Path(updated['profile_review']).is_relative_to(temporary)
        raise RuntimeError('Preparation dispatch verified')
    from pathlib import Path
    monkeypatch.setattr('coronahdr.canon.prepare_canon', prepare)
    monkeypatch.setattr(folders, 'match_profile', stop_after_preparation)
    with pytest.raises(RuntimeError, match='Preparation dispatch verified'):
        folders.run_folders(path)


def test_resume_completed_acquisition_keeps_registration(tmp_path, monkeypatch):
    import coronahdr.canon as module
    previous = tmp_path/'previous'
    source = previous/'prepared'
    source.mkdir(parents=True)
    recipe = dict(processing=dict(registration_roi_xywh=[0, 0, 512, 512]))
    (previous/'recipe.json').write_text(json.dumps(recipe))
    report = dict(recipe=recipe, reference=0, inputs=[dict(metadata=dict(source_sha256='original'))],
                  transformations=[dict(index=0, shift_dy_dx=[300., -400.], path=[0])])
    (source/'report.json').write_text(json.dumps(report))
    verified = []
    monkeypatch.setattr('coronahdr.streaming.load_cached_run', lambda source, recipe: verified.append(source))
    def unexpected(*args):
        raise AssertionError('Completed acquisition must not run again')
    monkeypatch.setattr(module, 'canon_recipe', unexpected)
    monkeypatch.setattr('coronahdr.pipeline.run_manifest', unexpected)
    resumed, review = module.prepare_canon({}, tmp_path, tmp_path/'continued', resume_from=previous)
    assert resumed == source and verified == [source]
    geometry = json.loads(review.read_text())
    np.testing.assert_array_equal(np.array(geometry['matrices_yx']['0'])[:2, 2], [300., -400.])


@pytest.mark.parametrize('truncated_report', [False, True])
def test_resume_acquisition_dispatches_preparation_checkpoint(tmp_path, monkeypatch, truncated_report):
    import coronahdr.canon as module
    checkpoint = tmp_path/'previous/prepared/preparation'
    checkpoint.mkdir(parents=True)
    (checkpoint/'report.json').write_text('{}')
    if truncated_report:
        (checkpoint.parent/'report.json').write_text('{')
    recipe = dict(max_shift=256, processing=dict(registration_roi_xywh=[0, 0, 512, 512]))
    (tmp_path/'previous/recipe.json').write_text(json.dumps(recipe))
    def run(manifest, source):
        saved = json.loads(manifest.read_text())
        assert saved['max_shift'] == 1024
        assert saved['processing']['cache_from'] == str(checkpoint)
        return dict(reference=0, inputs=[dict(metadata=dict(source_sha256='original'))],
                    transformations=[dict(index=0, shift_dy_dx=[0., 0.])])
    monkeypatch.setattr('coronahdr.pipeline.run_manifest', run)
    module.prepare_canon({}, tmp_path, tmp_path/'continued', resume_from=tmp_path/'previous')
