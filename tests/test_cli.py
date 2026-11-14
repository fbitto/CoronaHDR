"""Default desktop dispatch and explicit command-line parsing."""
import sys

import pytest

from coronahdr.cli import main


@pytest.mark.parametrize('arguments', [[], ['desktop'], ['desktop', '--source', 'prepared', '--geometry-review', 'review.json']])
def test_desktop_dispatch(monkeypatch, arguments):
    from pathlib import Path
    import coronahdr.desktop as desktop
    calls = []
    monkeypatch.setattr(sys, 'argv', ['coronahdr', *arguments])
    monkeypatch.setattr(desktop, 'launch', lambda source, review: calls.append((source, review)) or 7)
    assert main() == 7
    assert calls == ([(Path('prepared'), Path('review.json'))] if '--source' in arguments else [(None, None)])


@pytest.mark.parametrize('arguments, code', [(['--help'], 0), (['inspect'], 2), (['unknown'], 2)])
def test_explicit_arguments_are_not_replaced_with_desktop(monkeypatch, arguments, code):
    monkeypatch.setattr(sys, 'argv', ['coronahdr', *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == code
