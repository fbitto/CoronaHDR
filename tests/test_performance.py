import os

from coronahdr.performance import detector_workers


def test_detector_workers_obey_memory_cpu_and_work_limits(monkeypatch):
    import coronahdr.performance as performance
    monkeypatch.setattr(os, 'cpu_count', lambda: 10)
    monkeypatch.setattr(os, 'sysconf', lambda key: {'SC_PHYS_PAGES': 24*1024**3//4096, 'SC_PAGE_SIZE': 4096}[key])
    monkeypatch.setattr(performance, 'available_memory', lambda: 16*1024**3)
    assert detector_workers((3728, 5600), 26) == 2
    assert detector_workers((5000, 7300), 63) == 1
    assert detector_workers((96, 96), 1) == 1
    monkeypatch.setattr(performance, 'available_memory', lambda: 1024**3)
    assert detector_workers((3728, 5600), 26) == 1
    monkeypatch.setattr(os, 'cpu_count', lambda: 1)
    assert detector_workers((96, 96), 20) == 1
    def unavailable(key):
        raise ValueError(key)
    monkeypatch.setattr(os, 'sysconf', unavailable)
    assert detector_workers((96, 96), 20) == 1


def test_stage_reports_even_when_work_fails(capsys):
    import pytest
    from coronahdr.performance import stage
    with pytest.raises(ValueError):
        with stage('test'):
            raise ValueError('failure')
    output = capsys.readouterr().out
    assert 'PERFORMANCE test:' in output and 'process_peak_rss=' in output
