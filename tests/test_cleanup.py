import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path

from coronahdr.cleanup import clean_jobs, scan_jobs


def make_job(base, ident, **values):
    root = base/'.coronahdr-jobs'/f'{ident:032x}'
    root.mkdir(parents=True)
    output = base/f'output-{ident}'
    output.mkdir()
    job = dict(destination=str(output), result=str(root/'result'), status='failed',
               publication='standalone_final_files_v1', folders={})
    job.update(values)
    (root/'job.json').write_text(json.dumps(job))
    (root/'process.log').write_text('important log')
    (root/'cache').write_text('temporary')
    return root, output


def test_cleanup_multiple_roots_preserves_logs_and_unknown_files(tmp_path):
    one, two = tmp_path/'one', tmp_path/'two'
    one.mkdir(); two.mkdir()
    a, out = make_job(one, 1)
    b, _ = make_job(two, 2)
    (one/'original.CR3').write_text('original')
    (out/f'CoronaHDR_{a.name}_process.log').write_text('collision')
    scan = scan_jobs([one, two, tmp_path/'disconnected'])
    assert len(scan['jobs']) == 2 and len(scan['unavailable']) == 1
    result = clean_jobs([one, two], [a, b])
    assert all(item['status'] == 'removed' for item in result.values())
    assert (one/'original.CR3').read_text() == 'original'
    assert (out/f'CoronaHDR_{a.name}_process.log').read_text() == 'collision'
    assert (out/f'CoronaHDR_{a.name}_process_1.log').read_text() == 'important log'


def test_missing_selection_is_reported_and_other_selected_folder_is_deleted(tmp_path):
    missing, _ = make_job(tmp_path, 1)
    selected, output = make_job(tmp_path, 2)
    unselected, _ = make_job(tmp_path, 3)
    shutil.rmtree(missing)
    result = clean_jobs([tmp_path], [missing, selected])
    assert result[str(missing)]['status'] == 'skipped'
    assert 'no longer exists' in result[str(missing)]['reason']
    assert result[str(selected)]['status'] == 'removed' and not selected.exists()
    assert unselected.exists()
    assert (output/f'CoronaHDR_{selected.name}_process.log').read_text() == 'important log'


def test_job_without_logs_can_be_removed_when_output_parent_is_missing(tmp_path):
    root, _ = make_job(tmp_path, 1, destination=str(tmp_path/'missing'/'output'), status='ready')
    (root/'process.log').unlink()
    result = clean_jobs([tmp_path], [root])
    assert result[str(root)]['status'] == 'removed'
    assert not root.exists()


def test_size_format_shows_small_jobs():
    from coronahdr.desktop_cleanup import format_size
    assert format_size(17) == '17 B'
    assert format_size(2048) == '2.00 KiB'
    assert format_size(1024**2) == '1.00 MiB'


def test_missing_log_destination_still_protects_logs(tmp_path):
    root, _ = make_job(tmp_path, 1, destination=str(tmp_path/'missing'/'output'))
    result = clean_jobs([tmp_path], [root])
    assert result[str(root)]['status'] == 'skipped'
    assert (root/'process.log').read_text() == 'important log'


def test_locks_dependencies_and_legacy_output_are_protected(tmp_path):
    a, _ = make_job(tmp_path, 1)
    b, _ = make_job(tmp_path, 2)
    (b/'recipe.json').write_text(json.dumps({'cache_from': str(a/'cache')}))
    assert clean_jobs([tmp_path], [a])[str(a)]['status'] == 'skipped'
    with (b/'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert clean_jobs([tmp_path], [b])[str(b)]['status'] == 'skipped'
    job = json.loads((b/'job.json').read_text())
    job.pop('publication')
    (b/'job.json').write_text(json.dumps(job))
    (b/'result').mkdir()
    assert clean_jobs([tmp_path], [b])[str(b)]['status'] == 'skipped'
    assert a.exists() and b.exists()


def test_symlinks_never_followed_and_unpublished_master_protected(tmp_path):
    root, _ = make_job(tmp_path, 1)
    outside = tmp_path/'Lights'
    outside.mkdir()
    (outside/'raw').write_text('source')
    (root/'link').symlink_to(outside, target_is_directory=True)
    (root/'master.fits').write_text('unpublished')
    assert clean_jobs([tmp_path], [root])[str(root)]['status'] == 'skipped'
    (root/'master.fits').unlink()
    assert clean_jobs([tmp_path], [root])[str(root)]['status'] == 'removed'
    assert (outside/'raw').read_text() == 'source'


def test_unreadable_known_record_blocks_cleanup(tmp_path):
    root, _ = make_job(tmp_path, 1)
    other, _ = make_job(tmp_path, 2)
    (other/'job.json').write_text('{')
    assert clean_jobs([tmp_path], [root])[str(root)]['status'] == 'skipped'
    assert root.exists()


def test_final_images_must_be_independent(tmp_path):
    root, output = make_job(tmp_path, 1)
    result = root/'result'
    result.mkdir()
    source = result/'master.fits'
    source.write_bytes(b'final image')
    target = output/'test_aesthetic.fits'
    os.link(source, target)
    job = json.loads((root/'job.json').read_text())
    job.update(status='completed', hdr_mode='aesthetic', output_base_name='test',
               output_format='fits', published_files={target.name: hashlib.sha256(target.read_bytes()).hexdigest()})
    (root/'job.json').write_text(json.dumps(job))
    assert clean_jobs([tmp_path], [root])[str(root)]['status'] == 'skipped'
    target.unlink()
    target.write_bytes(source.read_bytes())
    assert clean_jobs([tmp_path], [root])[str(root)]['status'] == 'removed'
    assert target.read_bytes() == b'final image'


def test_cleanup_dialog_lists_locations_and_selects_multiple_jobs(tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from coronahdr import desktop_cleanup

    application = QApplication.instance() or QApplication([])
    a, _ = make_job(tmp_path, 1)
    b, _ = make_job(tmp_path, 2)
    monkeypatch.setattr(desktop_cleanup, 'remember_location', lambda path: None)
    monkeypatch.setattr(desktop_cleanup, 'locations', lambda: [str(tmp_path)])
    monkeypatch.setattr(desktop_cleanup.CleanupDialog, 'scan', lambda self: None)
    dialog = desktop_cleanup.CleanupDialog(None, str(tmp_path))
    dialog.scanned(scan_jobs([tmp_path]), None)
    assert dialog.locations.count() == 1 and dialog.table.rowCount() == 2
    for row in range(2):
        dialog.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
    assert dialog.delete_button.isEnabled()
    assert {row['path'] for row in dialog.jobs} == {str(a), str(b)}
    dialog.close()
    assert application is not None


def test_cleanup_sort_and_select_all_keep_job_identity(tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from coronahdr import desktop_cleanup

    application = QApplication.instance() or QApplication([])
    small, _ = make_job(tmp_path, 1)
    large, _ = make_job(tmp_path, 2)
    locked, _ = make_job(tmp_path, 3)
    (large/'cache').write_bytes(b'x' * 10000)
    monkeypatch.setattr(desktop_cleanup, 'remember_location', lambda path: None)
    monkeypatch.setattr(desktop_cleanup, 'locations', lambda: [str(tmp_path)])
    monkeypatch.setattr(desktop_cleanup.CleanupDialog, 'scan', lambda self: None)
    with (locked/'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dialog = desktop_cleanup.CleanupDialog(None, str(tmp_path))
        dialog.scanned(scan_jobs([tmp_path]), None)
        assert dialog.table.horizontalHeaderItem(1).text() == 'Folder size (estimate)'
        dialog.table.sortItems(1, Qt.SortOrder.DescendingOrder)
        assert dialog.table.item(0, 0).data(Qt.ItemDataRole.UserRole)['path'] == str(large)
        dialog.select_all.click()
        assert {row['path'] for row in dialog.selected_jobs()} == {str(small), str(large)}
        dialog.table.sortItems(1, Qt.SortOrder.AscendingOrder)
        dialog.table.item(next(i for i in range(3) if dialog.table.item(i, 0).data(Qt.ItemDataRole.UserRole)['path'] == str(large)), 0).setCheckState(Qt.CheckState.Unchecked)
        captured = []
        monkeypatch.setattr(desktop_cleanup, 'message_dialog', lambda *args, **kwargs: True)
        monkeypatch.setattr(dialog, 'run_task', lambda selected=None: captured.extend(selected))
        dialog.confirm_delete()
        assert captured == [str(small)]
        dialog.close()
    assert application is not None


def test_long_cleanup_message_is_scrollable_and_fits_screen():
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog, QPlainTextEdit
    from coronahdr.desktop_cleanup import message_dialog

    application = QApplication.instance() or QApplication([])
    text = '/very-long-path/' * 300
    observed = []
    def inspect():
        dialog = application.activeModalWidget()
        edit = dialog.findChild(QPlainTextEdit)
        observed.append((dialog.size(), edit.toPlainText()))
        dialog.reject()
    QTimer.singleShot(0, inspect)
    assert not message_dialog(None, 'Warning', text, confirm=True)
    size, shown = observed[0]
    available = application.primaryScreen().availableGeometry()
    assert size.width() <= available.width() and size.height() <= available.height()
    assert shown == text


def test_confirmed_dialog_deletes_only_selected_existing_job(tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from coronahdr import desktop_cleanup

    application = QApplication.instance() or QApplication([])
    selected, _ = make_job(tmp_path, 1)
    kept, _ = make_job(tmp_path, 2)
    monkeypatch.setattr(desktop_cleanup, 'remember_location', lambda path: None)
    monkeypatch.setattr(desktop_cleanup, 'locations', lambda: [str(tmp_path)])
    monkeypatch.setattr(desktop_cleanup.CleanupDialog, 'scan', lambda self: None)
    monkeypatch.setattr(desktop_cleanup, 'message_dialog', lambda *args, **kwargs: True)
    dialog = desktop_cleanup.CleanupDialog(None, str(tmp_path))
    dialog.scanned(scan_jobs([tmp_path]), None)
    for i in range(dialog.table.rowCount()):
        item = dialog.table.item(i, 0)
        if item.data(Qt.ItemDataRole.UserRole)['path'] == str(selected):
            item.setCheckState(Qt.CheckState.Checked)
    monkeypatch.setattr(dialog, 'run_task', lambda paths: clean_jobs([tmp_path], paths))
    dialog.confirm_delete()
    assert not selected.exists() and kept.exists()
    dialog.close()
    assert application is not None


def test_folder_context_menu_targets_and_missing_location(tmp_path, monkeypatch):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QApplication
    from coronahdr import desktop_cleanup

    application = QApplication.instance() or QApplication([])
    root, output = make_job(tmp_path, 1)
    monkeypatch.setattr(desktop_cleanup, 'remember_location', lambda path: None)
    monkeypatch.setattr(desktop_cleanup, 'locations', lambda: [str(tmp_path)])
    monkeypatch.setattr(desktop_cleanup.CleanupDialog, 'scan', lambda self: None)
    dialog = desktop_cleanup.CleanupDialog(None, str(tmp_path))
    dialog.scanned(scan_jobs([tmp_path]), None)
    dialog.show()
    application.processEvents()
    menus = []
    monkeypatch.setattr(dialog, 'folder_menu', lambda widget, position, path, job_path=None, available=True: menus.append((path, job_path, available)))
    dialog.location_menu(dialog.locations.visualItemRect(dialog.locations.item(0)).center())
    dialog.job_menu(dialog.table.visualItemRect(dialog.table.item(0, 5)).center())
    dialog.job_menu(dialog.table.visualItemRect(dialog.table.item(0, 6)).center())
    assert menus == [(str(tmp_path), None, True), (str(tmp_path), str(root), True), (str(output), str(root), True)]
    opened, errors = [], []
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toLocalFile()) or True)
    monkeypatch.setattr(desktop_cleanup, 'message_dialog', lambda *args, **kwargs: errors.append(args[2]))
    dialog.open_folder(str(output))
    dialog.open_folder(str(tmp_path/'disconnected'))
    assert opened == [str(output)] and len(errors) == 1
    dialog.close()


def test_missing_output_is_labeled_and_cannot_be_opened(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from coronahdr import desktop_cleanup

    application = QApplication.instance() or QApplication([])
    root, _ = make_job(tmp_path, 1, destination=str(tmp_path/'missing'))
    monkeypatch.setattr(desktop_cleanup, 'remember_location', lambda path: None)
    monkeypatch.setattr(desktop_cleanup, 'locations', lambda: [str(tmp_path)])
    monkeypatch.setattr(desktop_cleanup.CleanupDialog, 'scan', lambda self: None)
    dialog = desktop_cleanup.CleanupDialog(None, str(tmp_path))
    dialog.scanned(scan_jobs([tmp_path]), None)
    dialog.show()
    application.processEvents()
    assert '(missing or unavailable)' in dialog.table.item(0, 6).text()
    captured = []
    monkeypatch.setattr(dialog, 'folder_menu', lambda widget, position, path, job_path=None, available=True: captured.append((path, job_path, available)))
    dialog.job_menu(dialog.table.visualItemRect(dialog.table.item(0, 6)).center())
    assert captured == [(str(tmp_path/'missing'), str(root), False)]
    dialog.close()
