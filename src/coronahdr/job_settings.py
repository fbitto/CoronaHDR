"""Portable, user-editable folder-job settings."""
import json
from pathlib import Path


def read_settings(path):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read settings: {error}') from error
    if not isinstance(data, dict) or data.get('format') != 'coronahdr-folder-settings' or data.get('version') != 1:
        raise ValueError('Unsupported CoronaHDR settings file or version')
    folders = data.get('folders')
    if not isinstance(folders, dict) or set(folders) - {'light', 'bias', 'dark', 'flat', 'dark_flat'} or not all(isinstance(v, str) for v in folders.values()):
        raise ValueError('Invalid input folders in settings')
    defaults = dict(mode='aesthetic', alignment='conservative', output_format='both', base_name='',
                    temporary='', destination='', lunar_margin=0, bias_fallback=False,
                    detect_hot_pixels=False, save_intermediates=[], prominence=False)
    options = data.get('settings')
    if not isinstance(options, dict) or set(options) - set(defaults):
        raise ValueError('Invalid settings fields')
    options = defaults | options
    if (options['mode'] not in ('scientific', 'aesthetic', 'both') or
        options['alignment'] not in ('conservative', 'ael_visual') or
        options['output_format'] not in ('fits', 'tiff', 'both') or
        any(type(options[k]) is not bool for k in ('bias_fallback', 'detect_hot_pixels', 'prominence')) or
        (options['lunar_margin'] is not None and (type(options['lunar_margin']) is not int or not -4 <= options['lunar_margin'] <= 4)) or
        any(not isinstance(options[k], str) for k in ('base_name', 'temporary', 'destination')) or
        not isinstance(options['save_intermediates'], list) or
        any(x not in ('calibrated', 'debayered', 'registered') for x in options['save_intermediates'])):
        raise ValueError('Invalid values in settings file')
    from .desktop_jobs import output_base_name
    output_base_name(options['base_name'])
    return folders, options
