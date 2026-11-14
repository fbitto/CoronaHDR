"""Synchronous worker decisions, presented by the desktop over its process pipes."""
import json
import os


def ask(message, choices, default):
    if os.environ.get('CORONAHDR_INTERACTIVE') != '1':
        return default
    print('CORONAHDR_DECISION ' + json.dumps(dict(message=message, choices=choices)), flush=True)
    try:
        answer = input()
    except EOFError:
        raise ValueError('Processing stopped: no desktop decision received') from None
    if answer not in choices:
        raise ValueError('Processing stopped by user; preparation files retained')
    return answer


def review_frames(rows):
    """Return explicitly selected candidate indices; noninteractive jobs exclude them."""
    if os.environ.get('CORONAHDR_INTERACTIVE') != '1':
        return []
    print('CORONAHDR_FRAME_REVIEW ' + json.dumps(rows), flush=True)
    try:
        answer = json.loads(input())
    except (EOFError, json.JSONDecodeError):
        raise ValueError('Processing stopped: no valid frame review received') from None
    candidates = {row['index'] for row in rows if row['status'] == 'candidate'}
    if not isinstance(answer, list) or any(type(i) is not int or i not in candidates for i in answer) or len(set(answer)) != len(answer):
        raise ValueError('Processing stopped: invalid frame review selection')
    return answer


def select_qualified_frames(rows):
    """Select among qualified frames without restoring rejected frames."""
    qualified = {row['index'] for row in rows if row['status'] == 'accepted'}
    required = {row['index'] for row in rows if row.get('reference')}
    if os.environ.get('CORONAHDR_INTERACTIVE') != '1':
        return [row['index'] for row in rows if row['index'] in qualified]
    print('CORONAHDR_SELECTION_REVIEW ' + json.dumps(rows), flush=True)
    try:
        answer = json.loads(input())
    except (EOFError, json.JSONDecodeError):
        raise ValueError('Processing stopped: no valid frame selection received') from None
    if (not isinstance(answer, list) or any(type(i) is not int or i not in qualified for i in answer)
            or len(set(answer)) != len(answer) or not required.issubset(answer)):
        raise ValueError('Processing stopped: invalid frame selection')
    return answer
