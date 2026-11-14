"""Deterministic analytic corona, sampled independently at displaced coordinates."""
import json
from pathlib import Path
import numpy as np
from astropy.io import fits
from .io import write_image
from .pipeline import run_manifest
from .sensor import SensorFrame, phase_map, level_map


def corona(y, x, size):
    dx, dy = x-size/2, y-size/2
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    background = 25 + 800*np.exp(-radius/65)
    structures = (1 + .24*np.cos(3*angle+.018*radius) + .17*np.sin(7*angle-.04*radius)
                  + .09*np.cos(13*angle+.08*radius))
    rays = 500*np.exp(-((dy-.28*dx-12)/9)**2)*np.exp(-radius/100)
    knot = 6000*np.exp(-((dx-48)**2+(dy+30)**2)/60)
    return background*structures + rays + knot


def make_scene(directory, noise=.25, size=192, moon_motion=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    y, x = np.indices((size, size), dtype=float)
    truth = corona(y, x, size).astype(np.float32)
    write_image(directory/'truth.fits', truth, {'state': 'linear_mono'})
    rng = np.random.default_rng(4217)
    offsets = [(0., 0.), (1.35, -2.42), (-2.28, 1.71), (.65, 2.37)]
    exposures = [.25, .5, 1., 2.]
    entries = []
    for i, ((dy, dx), exposure) in enumerate(zip(offsets, exposures)):
        image = corona(y-dy, x-dx, size)*exposure
        image += rng.normal(0, noise, image.shape)
        moon = [size/2+dx+(i*2.3 if moon_motion else 0), size/2+dy-(i*1.4 if moon_motion else 0), 27.]
        hidden = (x-moon[0])**2+(y-moon[1])**2 <= moon[2]**2
        image[hidden] = rng.normal(0, noise, hidden.sum())
        image = np.minimum(image, 1000).astype(np.float32)
        metadata = dict(state='linear_mono', exposure_s=exposure, white_level=1000.,
                        low_signal=1., black_level=64., black_subtracted=True,
                        response_group='synthetic fixed camera response', iso=100,
                        gain=1., aperture=8., lunar_disk=moon, limb_margin_px=5.,
                        temperature_c=12., active_area_xywh=[0, 0, size, size],
                        provenance='Analytic synthetic scene; additive Gaussian noise; hard clipping',
                        noise_sigma=noise)
        filename = f'light_{i:02d}.{"tiff" if i % 2 else "fits"}'
        write_image(directory/filename, image, metadata)
        entries.append({'path': filename})
    recipe = dict(method='weighted', reference=0, max_shift=12, frames=entries)
    (directory/'manifest.json').write_text(json.dumps(recipe, indent=2))
    (directory/'truth.json').write_text(json.dumps(dict(source_offsets_dy_dx=offsets,
        expected_shifts_dy_dx=[[-dy, -dx] for dy, dx in offsets], noise_sigma=noise,
        conditions='192x192 analytic asymmetric corona, fixed response, four exposures, independent moving lunar disk, no clouds; error tested outside occultation/saturation and image edges'), indent=2))
    return directory/'manifest.json'


def make_cfa_scene(directory):
    """Synthetic detector files, not a claim of real-camera RAW validation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    size = 192
    y, x = np.indices((size, size), dtype=float)
    rng = np.random.default_rng(9812)
    base = dict(state='linear_cfa', role='light', sensor_id='analytic Bayer detector',
        exposure_s=1., iso=100., gain=1., aperture=8., cfa_pattern='RGGB',
        active_area_xywh=[5, 3, size, size], black_level=[100., 110., 120., 130.],
        white_level=[1100., 1110., 1120., 1130.], black_subtracted=False,
        temperature_c=12., response_group='fixed synthetic Bayer response',
        lunar_disk=None, low_signal=1., limb_margin_px=5., provenance='synthetic detector counts')
    detector = SensorFrame(np.zeros((size, size)), base, directory/'virtual')
    labels = phase_map(detector.data.shape, base)
    pedestal = level_map(detector, 'black_level') + .025*x + .01*y
    white = level_map(detector, 'white_level')
    dark_current = .6 + .003*y
    response = .75 + .25*((x-95.5)**2+(y-95.5)**2)/(2*95.5**2)
    response /= np.median(response)
    color_factors = dict(R=.8, G1=1.2, G2=1.2, B=1.)
    color = np.zeros((size, size))
    for phase, factor in color_factors.items():
        color[labels == phase] = factor
    truth = corona(y, x, size)[..., None]*np.array([.8, 1.2, 1.])
    write_image(directory/'truth.fits', truth, {'state': 'linear_rgb'})
    calibration = {key: [] for key in ('bias', 'dark', 'flat', 'dark_flat')}

    def save(name, data, **changes):
        metadata = dict(base, **changes)
        write_image(directory/name, np.asarray(data, np.float32), metadata)
        return {'path': name}

    for i in range(3):
        calibration['bias'].append(save(f'bias_{i}.fits', pedestal, role='bias', exposure_s=0.))
    flat_exposure = .75
    for i in range(3):
        calibration['flat'].append(save(f'flat_{i}.fits', 500*color*response+pedestal+dark_current*flat_exposure,
                                       role='flat', exposure_s=flat_exposure))
        calibration['dark_flat'].append(save(f'dark_flat_{i}.fits', pedestal+dark_current*flat_exposure,
            role='dark_flat', exposure_s=flat_exposure, dark_contains_bias=True))
    offsets = [(0., 0.), (1.35, -2.42), (-2.28, 1.71), (.65, 2.37)]
    entries = []
    for i, ((dy, dx), exposure) in enumerate(zip(offsets, [.25, .5, 1., 2.])):
        for repeat in range(2):
            calibration['dark'].append(save(f'dark_{i}_{repeat}.fits', pedestal+dark_current*exposure,
                role='dark', exposure_s=exposure, dark_contains_bias=True))
        signal = corona(y-dy, x-dx, size)*color*response*exposure
        moon = [96+dx+i*2.3, 96+dy-i*1.4, 27.]
        signal[(x-moon[0])**2+(y-moon[1])**2 <= moon[2]**2] = 0
        source = signal+pedestal+dark_current*exposure+rng.normal(0, .25, signal.shape)
        entries.append(save(f'light_{i}.tiff', np.minimum(source, white), exposure_s=exposure, lunar_disk=moon))
    recipe = dict(method='weighted', reference=0, max_shift=12, frames=entries, calibration=calibration)
    (directory/'manifest.json').write_text(json.dumps(recipe, indent=2))
    (directory/'truth.json').write_text(json.dumps(dict(expected_shifts_dy_dx=[[-dy, -dx] for dy, dx in offsets],
        conditions='192x192 synthetic Bayer detector at odd sensor origin; distinct R/G1/G2/B pedestal; spatial bias/dark/flat; four exposures; independent lunar motion; additive noise sigma 0.25; no real RAW files'), indent=2))
    return directory/'manifest.json'


def demo(directory, cfa=False):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f'Refusing to overwrite {directory}')
    directory.mkdir(parents=True)
    manifest = make_cfa_scene(directory/'inputs') if cfa else make_scene(directory/'inputs')
    report = run_manifest(manifest, directory/'output')
    truth_info = json.loads((directory/'inputs/truth.json').read_text())
    estimated = np.array([item['shift_dy_dx'] for item in report['transformations']])
    expected = np.array(truth_info['expected_shifts_dy_dx'])
    error = np.linalg.norm(estimated-expected, axis=1)
    master = fits.getdata(directory/'output/master.fits')
    truth = fits.getdata(directory/'inputs/truth.fits')
    if master.ndim == 3:
        master, truth = np.moveaxis(master, 0, -1), np.moveaxis(truth, 0, -1)
    support = fits.getdata(directory/'output/support.fits') > 0
    # Independent validation region, not the registration fit sample selection.
    y, x = np.indices(master.shape[:2])
    region = support & (x > 15) & (x < 177) & (y > 15) & (y < 177) & (np.hypot(x-96, y-96) > 43)
    relative_rmse = float(np.sqrt(np.mean((master[region]-truth[region])**2))/np.sqrt(np.mean(truth[region]**2)))
    validation = dict(max_shift_error_px=float(error.max()), shift_errors_px=error.tolist(),
                      relative_radiance_rmse=relative_rmse, independent_validation_pixels=int(region.sum()),
                      unresolved_saturation_pixels=int((fits.getdata(directory/'output/unresolved_saturation.fits') > 0).sum()),
                      conditions=truth_info['conditions'])
    (directory/'validation.json').write_text(json.dumps(validation, indent=2))
    if error.max() >= .1 or relative_rmse >= .02:
        raise ValueError(f'Synthetic acceptance failed: {validation}')
    return validation
