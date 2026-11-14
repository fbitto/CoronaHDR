import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from astropy.io import fits

from coronahdr.display_export import analyze, export
from coronahdr.desktop_previews import render_preview


@pytest.mark.parametrize('extension', ['.png', '.tiff', '.fits'])
@pytest.mark.parametrize('repair', [False, True])
def test_display_export_formats_and_validity(tmp_path, extension, repair):
    data = np.ones((40, 50, 3), np.float32)
    data[..., 0] *= 2
    data[3, 3] = np.nan
    data[10:20, 10:20] = np.nan
    source = tmp_path/'source.tiff'
    tifffile.imwrite(source, data, photometric='rgb')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = analyze(source)
    assert result['missing'] == 101 and result['eligible'] == 1
    levels = dict(exposure=0., black=0., white=1., midtones=.5, saturation=1.)
    destination = tmp_path/('display'+extension)
    receipt = export(source, destination, levels, None, repair, result['identity'])
    mask = cv2.imread(receipt['validity'], cv2.IMREAD_UNCHANGED)
    assert mask[3, 3] == (127 if repair else 0) and mask[15, 15] == 0 and mask[0, 0] == 255
    if extension == '.png':
        image = cv2.cvtColor(cv2.imread(str(destination), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    elif extension == '.tiff':
        image = tifffile.imread(destination)
    else:
        image = np.moveaxis(fits.getdata(destination), 0, -1)
    assert image.dtype == np.uint16 and image.shape == data.shape
    assert np.all(image[15, 15] == 0) and bool(image[3, 3].any()) == repair
    assert image[0, 0, 0] > image[0, 0, 1]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        export(source, destination, levels, None, repair, result['identity'])


def test_changed_source_cannot_be_exported(tmp_path):
    source = tmp_path/'source.tiff'
    tifffile.imwrite(source, np.ones((10, 10, 3), np.float32), photometric='rgb')
    receipt = analyze(source)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        export(source, tmp_path/'out.png', dict(exposure=0., black=0., white=1., midtones=.5, saturation=1.),
               None, False, receipt['identity'])
    assert not (tmp_path/'out.png').exists()


def test_export_matches_preview_levels(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.uniform(0.01, 10, (30, 40, 3)).astype(np.float32)
    source = tmp_path/'source.tiff'
    tifffile.imwrite(source, data, photometric='rgb')
    levels = dict(exposure=-1.2, black=.05, white=.85, midtones=.4, saturation=1.2)
    gains = [1.1, 1., 1.3]
    preview, _, _, _ = render_preview(source, 0, gains, levels)
    receipt = analyze(source)
    destination = tmp_path/'out.tiff'
    export(source, destination, levels, gains, False, receipt['identity'])
    image = tifffile.imread(destination)
    for y, x in ((0, 0), (9, 11), (20, 30)):
        pixel = preview.pixelColor(x, y)
        expected = np.rint(image[y, x].astype(float)*255/65535).astype(int)
        assert np.max(np.abs(expected - [pixel.red(), pixel.green(), pixel.blue()])) <= 1


def test_failed_mask_publication_rolls_back_image(tmp_path, monkeypatch):
    import coronahdr.display_export as module
    source = tmp_path/'source.tiff'
    tifffile.imwrite(source, np.ones((10, 10, 3), np.float32), photometric='rgb')
    receipt = analyze(source)
    real_link = module.os.link
    def fail_mask(src, dst):
        if str(dst).endswith('_validity.png'):
            raise OSError('mask write failed')
        return real_link(src, dst)
    monkeypatch.setattr(module.os, 'link', fail_mask)
    destination = tmp_path/'out.png'
    with pytest.raises(OSError, match='mask write failed'):
        export(source, destination, dict(exposure=0., black=0., white=1., midtones=.5, saturation=1.),
               None, False, receipt['identity'])
    assert not destination.exists() and not (tmp_path/'out_validity.png').exists()


def test_export_without_hardlink_support(tmp_path, monkeypatch):
    import errno
    import coronahdr.display_export as module
    source = tmp_path/'source.tiff'
    tifffile.imwrite(source, np.ones((10, 10, 3), np.float32), photometric='rgb')
    receipt = analyze(source)
    def unsupported(*args):
        raise OSError(errno.EOPNOTSUPP, 'hard links unavailable')
    monkeypatch.setattr(module.os, 'link', unsupported)
    result = export(source, tmp_path/'out.png', dict(exposure=0., black=0., white=1., midtones=.5, saturation=1.),
                    None, False, receipt['identity'])
    assert Path(result['image']).is_file() and Path(result['validity']).is_file()
