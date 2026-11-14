"""Linear HDR and detector inspection command line interface."""
import argparse
import json
import sys
from pathlib import Path
from .io import read_frame
from .pipeline import run_manifest
from .synthetic import demo
from .sensor import SensorFrame, level_map
from .calibration import sample_valid


def main():
    parser = argparse.ArgumentParser(description='CoronaHDR: linear eclipse HDR, without rendering or filters')
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='Validate a linear frame and show metadata')
    inspect.add_argument('path', type=Path)
    inspect.add_argument('--overrides', type=Path, help='JSON object of explicit metadata declarations/corrections')
    run = commands.add_parser('run', help='Process a JSON recipe into a new output directory')
    run.add_argument('manifest', type=Path)
    run.add_argument('--output', required=True, type=Path)
    aesthetic = commands.add_parser('aesthetic', help='Empirical visual-quality HDR from a prepared disk run and reviewed geometry')
    aesthetic.add_argument('source', type=Path, help='Completed run with report.json and a retained prepared cache')
    aesthetic.add_argument('--geometry-review', required=True, type=Path)
    aesthetic.add_argument('--output', required=True, type=Path)
    aesthetic.add_argument('--indices', nargs='+', type=int, help='Explicit indices; default: all reliably registered frames')
    aesthetic.add_argument('--feather', type=float, default=192, help='Weight boundary taper in native pixels')
    aesthetic.add_argument('--weight-sigma', type=float, default=8, help='Gaussian sigma for weights only, in native pixels')
    aesthetic.add_argument('--lunar-margin', type=int, choices=range(-4, 5), help='Margin around the unpadded fitted Moon, in native pixels')
    aesthetic.add_argument('--work-directory', type=Path, help='New temporary working directory on any drive')
    desktop = commands.add_parser('desktop', help='Open the desktop interface (default when no command is given)')
    desktop.add_argument('--source', type=Path)
    desktop.add_argument('--geometry-review', type=Path)
    demonstration = commands.add_parser('demo', help='Create, process and validate a synthetic scene')
    demonstration.add_argument('--output', required=True, type=Path)
    demonstration.add_argument('--cfa', action='store_true', help='Include synthetic Bayer bias/dark/flat calibration and linear demosaicing')
    args = parser.parse_args(sys.argv[1:] or ['desktop'])
    try:
        if sys.version_info[:3] != (3, 14, 8):
            raise ValueError(f'CoronaHDR requires the installed Python 3.14.8; running {sys.version} at {sys.executable}')
        if args.command == 'inspect':
            overrides = json.loads(args.overrides.read_text()) if args.overrides else None
            frame = read_frame(args.path, overrides)
            masks = (dict(saturation_samples=int((frame.data >= level_map(frame, 'white_level')).sum()),
                          invalid_samples=int((~sample_valid(frame)).sum())) if isinstance(frame, SensorFrame)
                     else {key: int(value.sum()) for key, value in frame.masks().items()})
            result = dict(path=str(frame.path), shape=list(frame.data.shape), working_dtype=str(frame.data.dtype),
                          metadata=frame.metadata, masks=masks)
        elif args.command == 'run':
            report = run_manifest(args.manifest, args.output)
            result = dict(output=str(args.output), reference=report['reference'], frames=len(report['inputs']))
            if 'accepted_indices' in report:
                result.update(requested=len(report.get('requested_indices', report['inputs'])),
                              accepted=len(report['accepted_indices']), rejected=len(report['rejected_frames']),
                              canvas_shape_yx=report['output']['shape_yx'])
        elif args.command == 'aesthetic':
            from .aesthetic_pipeline import build_aesthetic_hdr
            report = build_aesthetic_hdr(args.source, args.geometry_review, args.output,
                                         indices=args.indices, feather=args.feather, weight_sigma=args.weight_sigma,
                                         **({'lunar_margin': args.lunar_margin} if args.lunar_margin is not None else {}),
                                         **({'work_directory': args.work_directory} if args.work_directory is not None else {}))
            result = dict(output=str(args.output), reference=report['reference'],
                          accepted=len(report['accepted_indices']), geometry_rejected=len(report['rejected_frames']),
                          qualification=report['output']['qualification'])
        elif args.command == 'desktop':
            from .desktop import launch
            return launch(args.source, args.geometry_review)
        else:
            result = demo(args.output, cfa=args.cfa)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, StopIteration) as error:
        parser.exit(2, f'error: {error}\n')
