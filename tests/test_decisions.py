import pytest
from coronahdr.decisions import ask


def test_noninteractive_decisions_keep_existing_behavior(monkeypatch, capsys):
    monkeypatch.delenv('CORONAHDR_INTERACTIVE', raising=False)
    assert ask('Choose', ['crop', 'keep'], 'keep') == 'keep'
    assert capsys.readouterr().out == ''


def test_interactive_decision_and_cancel(monkeypatch, capsys):
    monkeypatch.setenv('CORONAHDR_INTERACTIVE', '1')
    monkeypatch.setattr('builtins.input', lambda: 'crop')
    assert ask('Choose', ['crop', 'keep'], 'keep') == 'crop'
    assert capsys.readouterr().out.startswith('CORONAHDR_DECISION ')
    monkeypatch.setattr('builtins.input', lambda: '')
    with pytest.raises(ValueError, match='stopped by user'):
        ask('Choose', ['crop', 'keep'], 'keep')
    def eof():
        raise EOFError
    monkeypatch.setattr('builtins.input', eof)
    with pytest.raises(ValueError, match='no desktop decision'):
        ask('Choose', ['crop', 'keep'], 'keep')
