"""Conservative resource limits and lightweight stage timing."""
from contextlib import contextmanager
import os
import resource
import subprocess
import sys
import time


def available_memory():
    try:
        page = os.sysconf('SC_PAGE_SIZE')
        if sys.platform == 'darwin':
            output = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=2, check=True).stdout
            pages = {line.split(':', 1)[0].strip(): int(line.split(':', 1)[1].strip().rstrip('.'))
                     for line in output.splitlines() if ':' in line and line.split(':', 1)[0].strip()
                     in ('Pages free', 'Pages inactive', 'Pages speculative')}
            if 'Pages free' not in pages or 'Pages inactive' not in pages:
                return None
            return sum(pages.values())*page
        return os.sysconf('SC_AVPHYS_PAGES')*page
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def detector_workers(shape, count):
    """Bound full-frame candidate scans; unknown memory falls back to serial."""
    try:
        memory = os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
        available = available_memory()
        if available is None:
            return 1
        pixels = int(shape[0])*int(shape[1])*(int(shape[2]) if len(shape) == 3 else 1)
        per_frame = max(pixels*96, 1)
        budget = min(int(memory*.25), int(available*.5))
        return max(1, min(2, count, max(1, (os.cpu_count() or 1)-1), budget//per_frame))
    except (OSError, ValueError, KeyError, TypeError):
        return 1


@contextmanager
def stage(name):
    start = time.perf_counter()
    cpu = time.process_time()
    before = resource.getrusage(resource.RUSAGE_SELF)
    try:
        yield
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss*1024
        print(f'PERFORMANCE {name}: wall={time.perf_counter()-start:.3f}s, '
              f'cpu={time.process_time()-cpu:.3f}s, process_peak_rss={rss} bytes, '
              f'input_blocks={usage.ru_inblock-before.ru_inblock}, output_blocks={usage.ru_oublock-before.ru_oublock}', flush=True)
