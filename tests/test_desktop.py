"""Desktop job safety and isolated, offscreen widget verification."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QProcess, Qt, QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from coronahdr.desktop import CoronaWindow, inspection_preview
from coronahdr.desktop_jobs import create_job, publish_job
from coronahdr.desktop_jobs import create_recipe_job, output_path
from coronahdr.desktop_import import ImportDialog


@pytest.fixture(scope='session')
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    # Dispose native widgets before Python tears down their callbacks and wrappers.
    for widget in application.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def source(tmp_path):
    root = tmp_path/'source'
    root.mkdir()
    report = dict(recipe=dict(reference=0, method='weighted'), inputs=[
        dict(path=f'frame{i}.tiff', metadata=dict(source_sha256=str(i), exposure_s=1, state='linear_rgb'))
        for i in range(3)])
    (root/'report.json').write_text(json.dumps(report))
    tifffile.imwrite(root/'master.tiff', np.ones((32, 48, 3), dtype='float32'), photometric='rgb')
    review = tmp_path/'review.json'
    review.write_text(json.dumps(dict(input_source_sha256=['0', '1', '2'],
        matrices_yx={str(i): np.eye(3).tolist() for i in (0, 1)})))
    return root, review


def test_job_publication_is_exclusive_and_requires_complete_products(source, tmp_path):
    root, review = source
    output = tmp_path/'output'
    path, job = create_job(root, review, output, [0, 1], 'aesthetic', lunar_margin=-2)
    assert job['arguments'][job['arguments'].index('--lunar-margin')+1] == '-2'
    result = Path(job['result'])
    result.mkdir()
    with pytest.raises(ValueError, match='Incomplete'):
        publish_job(path)
    assert not output.exists()
    (result/'report.json').write_text('{}')
    for name in ('master.fits', 'master.tiff'):
        (result/name).write_bytes(b'fixture')
    output.mkdir()
    (output/'master.tiff').write_bytes(b'keep existing')
    with pytest.raises(FileExistsError):
        publish_job(path)
    assert [p.name for p in output.iterdir()] == ['master.tiff']
    assert (output/'master.tiff').read_bytes() == b'keep existing'
    (output/'master.tiff').unlink()
    output.rmdir()
    publish_job(path)
    assert output.is_symlink() and output.resolve() == result
    assert json.loads(path.read_text())['status'] == 'completed'


def test_weighted_recipe_and_geometry_guards(source, tmp_path):
    root, review = source
    before = (root/'report.json').read_bytes()
    path, job = create_job(root, review, tmp_path/'output', [0, 1], 'weighted', lunar_margin=-2)
    recipe = json.loads((path.parent/'recipe.json').read_text())
    assert recipe['processing']['include_indices'] == [0, 1]
    assert recipe['processing']['rejection_policy'] == 'exclude'
    assert recipe['processing']['lunar_margin_px'] == -2
    assert (root/'report.json').read_bytes() == before
    with pytest.raises(ValueError):
        create_job(root, review, tmp_path/'invalid', [0, 2], 'aesthetic')


def test_preview_preserves_pixels_and_marks_missing(app, tmp_path):
    path = tmp_path/'image.tiff'
    data = np.ones((12, 16, 3), dtype='float32')*100
    data[0, 0] = np.nan
    data[0, 1] = -3
    tifffile.imwrite(path, data, photometric='rgb')
    before = path.read_bytes()
    image, high = inspection_preview(path)
    assert high == 100
    assert image.pixelColor(0, 0).red() == 48
    assert image.pixelColor(1, 0).red() == 0
    assert path.read_bytes() == before


def test_copy_job_log_includes_lines_not_shown_in_log_widget(app, tmp_path):
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    complete = ''.join(f'line {i}\n' for i in range(2100))+'last line without newline'
    (tmp_path/'process.log').write_text(complete, encoding='utf-8')
    for line in complete.splitlines():
        window.log.appendPlainText(line)
    assert 'line 0\n' not in window.log.toPlainText()
    window.copy_log_button.setEnabled(True)
    window.copy_log_button.click()
    assert QApplication.clipboard().text() == complete
    assert window.copy_log_button.text() == 'Copied'
    window.close()


def test_preview_is_loaded_off_gui_thread(app, monkeypatch, tmp_path):
    from PySide6.QtCore import QThread
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    import coronahdr.desktop_previews as module
    window = CoronaWindow()
    threads = []
    def preview(path, stretch, gains, prominence=None):
        threads.append(QThread.currentThread())
        return QImage(2, 2, QImage.Format.Format_RGB888), (2, 2), ((0, 0), 0), [1., 1., 1.]
    monkeypatch.setattr(module, 'render_preview', preview)
    window.show_preview(tmp_path/'master.tiff')
    for _ in range(100):
        app.processEvents()
        if not window.hdr_preview.tasks:
            break
        QTest.qWait(10)
    assert threads and threads[0] != window.thread()
    assert not window.hdr_preview.tasks and 'corona' in window.hdr_preview.images
    window.close()


def test_window_stays_open_during_export(app):
    window = CoronaWindow()
    window.show()
    window.hdr_preview.export_tasks.add(object())
    assert not window.close()
    assert 'wait' in window.status.text().lower()
    window.hdr_preview.export_tasks.clear()
    window.close()


def test_recovery_output_verification_runs_off_gui_thread(app, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from PySide6.QtCore import QThread
    from PySide6.QtTest import QTest
    import coronahdr.desktop as module
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    window.job_path.write_text(json.dumps(dict(status='failed', destination=str(tmp_path), result=str(tmp_path),
        publication='standalone_final_files_v1', published_files={'master.fits': 'digest'})))
    window.process = SimpleNamespace(readAllStandardOutput=lambda: b'', exitCode=lambda: 1,
                                     state=lambda: QProcess.ProcessState.NotRunning)
    threads = []
    def verify(job):
        threads.append(QThread.currentThread())
        return False
    monkeypatch.setattr(module, 'is_published', verify)
    window.finished(1, QProcess.ExitStatus.CrashExit)
    for _ in range(100):
        app.processEvents()
        if not window.publication_tasks:
            break
        QTest.qWait(10)
    assert threads and threads[0] != window.thread()
    assert not window.publication_tasks and 'failed' in window.status.text().lower()
    window.process = None
    window.close()


def test_copy_job_log_missing_file_leaves_clipboard_unchanged(app, tmp_path):
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    QApplication.clipboard().setText('unchanged')
    window.copy_job_log()
    assert QApplication.clipboard().text() == 'unchanged'
    assert 'Cannot copy job log' in window.status.text()
    window.close()


def test_copy_log_path_uses_saved_log_after_completion(app, tmp_path):
    temporary = tmp_path/'processing'
    output = tmp_path/'HDR_output'
    temporary.mkdir()
    output.mkdir()
    (temporary/'process.log').write_text('working')
    saved = output/'Eclipse_aesthetic.log'
    saved.write_text('completed')
    window = CoronaWindow()
    window.job_path = temporary/'job.json'
    window.job = dict(status='running', publication='standalone_final_files_v1',
                      destination=str(output), output_base_name='Eclipse', hdr_mode='aesthetic')
    window.copy_job_log_path()
    assert QApplication.clipboard().text() == str(temporary/'process.log')
    window.job['status'] = 'completed'
    window.copy_job_log_path()
    assert QApplication.clipboard().text() == str(saved)
    saved.unlink()
    window.copy_job_log_path()
    assert QApplication.clipboard().text() == str(temporary/'process.log')
    window.close()


def test_copy_log_path_missing_file_preserves_clipboard(app, tmp_path):
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    QApplication.clipboard().setText('unchanged')
    window.copy_job_log_path()
    assert QApplication.clipboard().text() == 'unchanged'
    assert 'Cannot copy job log path' in window.status.text()
    window.close()


def test_view_job_log_shows_complete_log_and_updates(app, tmp_path):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QPlainTextEdit
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    (tmp_path/'process.log').write_text('earlier lines\n', encoding='utf-8')
    window.view_log_button.setEnabled(True)
    window.view_log_button.click()
    app.processEvents()
    dialog = window.log_dialog
    viewer = dialog.findChild(QPlainTextEdit)
    assert dialog.isVisible() and viewer.isReadOnly()
    from PySide6.QtGui import QFontInfo
    assert QFontInfo(window.log.font()).fixedPitch()
    assert QFontInfo(viewer.font()).fixedPitch()
    assert viewer.toPlainText() == 'earlier lines\n'
    window.process = SimpleNamespace(readAllStandardOutput=lambda: 'new line ☀\n'.encode())
    window.read_process()
    assert viewer.toPlainText() == 'earlier lines\nnew line ☀\n'
    assert window.view_log_button.isEnabled()
    dialog.close()
    window.process = None
    window.close()


def test_view_job_log_reports_missing_file(app, tmp_path):
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    window.view_job_log()
    assert 'Cannot view job log' in window.status.text()
    assert window.log_dialog is None
    window.close()


def test_worker_output_enables_copy_and_includes_unterminated_line(app, tmp_path):
    class Worker:
        def readAllStandardOutput(self):
            return b'first line\nlast partial line'
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    window.process = Worker()
    window.read_process()
    assert window.copy_log_button.isEnabled()
    window.copy_log_button.click()
    assert QApplication.clipboard().text() == 'first line\nlast partial line'
    window.process = None
    window.close()


def test_worker_log_preserves_split_utf8_and_final_partial_line(app, tmp_path):
    class Worker:
        chunks = iter([b'corona \xe2', b'\x98', b'\x80 end', b''])
        def readAllStandardOutput(self):
            return next(self.chunks)
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    window.process = Worker()
    for _ in range(3):
        window.read_process()
    window.read_process(final=True)
    assert (tmp_path/'process.log').read_text(encoding='utf-8') == 'corona ☀ end'
    assert window.log.toPlainText() == 'corona ☀ end'
    window.process = None
    window.close()


@pytest.mark.parametrize('mode', ['scientific', 'aesthetic', 'both'])
def test_completed_folder_job_saves_full_log_next_to_independent_images(app, tmp_path, monkeypatch, mode):
    from coronahdr.desktop_jobs import publish_job
    temporary, destination = tmp_path/'temporary', tmp_path/'HDR_output'
    temporary.mkdir()
    result = temporary/'result'
    result.mkdir()
    (result/'report.json').write_text('{}')
    modes = ('scientific', 'aesthetic') if mode == 'both' else (mode,)
    for item in modes:
        folder = result/item if mode == 'both' else result
        folder.mkdir(exist_ok=True)
        (folder/'report.json').write_text('{}')
        for ext in ('fits', 'tiff'):
            (folder/f'master.{ext}').write_bytes(b'image')
    path = temporary/'job.json'
    job = dict(result=str(result), destination=str(destination), hdr_mode=mode,
               output_base_name='Eclipse', publication='standalone_final_files_v1')
    path.write_text(json.dumps(job))
    (temporary/'process.log').write_text('earlier line\n', encoding='utf-8')
    publish_job(path)
    class Worker:
        def readAllStandardOutput(self):
            return 'last line: corona ☀'.encode()
        def state(self):
            return QProcess.ProcessState.NotRunning
    window = CoronaWindow()
    window.job_path = path
    window.process = Worker()
    def no_gui_hash(job):
        raise AssertionError('Completed worker output must not be rehashed on the GUI thread')
    monkeypatch.setattr('coronahdr.desktop.is_published', no_gui_hash)
    monkeypatch.setattr(window, 'show_preview', lambda path, **kwargs: None)
    window.finished(0, QProcess.ExitStatus.NormalExit)
    output_log = destination/f'Eclipse_{mode}.log'
    assert output_log.read_text(encoding='utf-8') == 'earlier line\nlast line: corona ☀'
    assert not output_log.is_symlink() and not output_log.samefile(temporary/'process.log')
    assert {p.name for p in destination.iterdir()} == {
        *(f'Eclipse_{item}.{ext}' for item in modes for ext in ('fits', 'tiff')),
        f'Eclipse_{mode}.log'}
    assert 'job log are in the output folder' in window.status.text()
    window.process = None
    window.close()


def test_completed_job_log_does_not_replace_existing_file(tmp_path):
    from coronahdr.desktop_jobs import save_completed_log
    temporary, destination = tmp_path/'temporary', tmp_path/'output'
    temporary.mkdir()
    destination.mkdir()
    (temporary/'process.log').write_text('new log')
    existing = destination/'Eclipse_aesthetic.log'
    existing.write_text('keep')
    with pytest.raises(FileExistsError):
        save_completed_log(temporary/'job.json', dict(destination=str(destination), output_base_name='Eclipse'))
    assert existing.read_text() == 'keep'


def test_completed_job_log_copy_failure_removes_partial_file(tmp_path, monkeypatch):
    import coronahdr.desktop_jobs as jobs
    temporary, destination = tmp_path/'temporary', tmp_path/'output'
    temporary.mkdir()
    destination.mkdir()
    (temporary/'process.log').write_text('full log')
    def fail(src, dst, **kwargs):
        dst.write(b'partial')
        raise OSError('copy failed')
    monkeypatch.setattr(jobs.shutil, 'copyfileobj', fail)
    with pytest.raises(OSError, match='copy failed'):
        jobs.save_completed_log(temporary/'job.json', dict(destination=str(destination)))
    assert not list(destination.iterdir())
    assert (temporary/'process.log').read_text() == 'full log'


def test_incomplete_recorded_log_is_not_published(app, tmp_path, monkeypatch):
    import coronahdr.desktop as desktop
    import coronahdr.desktop_jobs as jobs
    destination = tmp_path/'output'
    destination.mkdir()
    path = tmp_path/'job.json'
    path.write_text(json.dumps(dict(status='completed', destination=str(destination), result=str(tmp_path),
                                    hdr_mode='aesthetic', publication='standalone_final_files_v1')))
    class Worker:
        def state(self):
            return QProcess.ProcessState.NotRunning
    window = CoronaWindow()
    window.job_path, window.process = path, Worker()
    window.log_write_failed = True
    monkeypatch.setattr(window, 'read_process', lambda **kwargs: None)
    monkeypatch.setattr(window, 'show_preview', lambda path, **kwargs: None)
    monkeypatch.setattr(desktop, 'is_published', lambda job: True)
    def unexpected(*args):
        raise AssertionError('A partial log must not be published')
    monkeypatch.setattr(jobs, 'save_completed_log', unexpected)
    window.finished(0, QProcess.ExitStatus.NormalExit)
    assert 'job log could not be saved' in window.status.text()
    assert not list(destination.iterdir())
    window.process = None
    window.close()


def test_folder_job_gives_log_unused_vertical_space(app, tmp_path, monkeypatch):
    import coronahdr.desktop as desktop
    class NotStartedProcess(QProcess):
        def start(self, *args):
            pass
    monkeypatch.setattr(desktop, 'QProcess', NotStartedProcess)
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    window.job = dict(destination=str(tmp_path/'result'))
    window.show()
    window.launch_job()
    app.processEvents()
    assert window.controls_scroll.maximumHeight() == 165
    assert not window.log_header.isHidden()
    assert window.log.isHidden()
    window.log_toggle.click()
    app.processEvents()
    assert not window.log.isHidden() and window.log.height() > 160
    assert not window.copy_log_button.isEnabled()
    window.close()


def test_window_selection_and_process_cancellation(app, source, tmp_path):
    root, review = source
    window = CoronaWindow(root, review)
    assert window.selected_indices() == [0, 1]
    assert window.start_button.isEnabled()
    assert window.table.item(2, 0).checkState() == Qt.CheckState.Unchecked
    output = tmp_path/'cancelled-output'
    window.job_path, window.job = create_job(root, review, output, [0, 1], 'aesthetic')
    window.process = QProcess(window)
    window.process.start(sys.executable, ['-c', 'import time; time.sleep(30)'])
    assert window.process.waitForStarted(5000)
    window.cancel_job()
    assert window.process.waitForFinished(5000)
    window.finished(window.process.exitCode(), window.process.exitStatus())
    assert not output.exists()
    assert json.loads(window.job_path.read_text())['status'] == 'cancelled'
    window.close()


def test_cancellation_stops_worker_with_parallel_threads(app, source, tmp_path):
    root, review = source
    window = CoronaWindow(root, review)
    window.job_path, window.job = create_job(root, review, tmp_path/'cancelled-parallel', [0, 1], 'aesthetic')
    window.process = QProcess(window)
    script = ('import concurrent.futures, time; '
              'pool=concurrent.futures.ThreadPoolExecutor(max_workers=2); '
              'pool.submit(time.sleep,30); pool.submit(time.sleep,30); '
              'print("ready",flush=True); time.sleep(30)')
    window.process.start(sys.executable, ['-u', '-c', script])
    assert window.process.waitForStarted(5000)
    assert window.process.waitForReadyRead(5000)
    window.cancel_job()
    assert window.process.waitForFinished(5000)
    window.finished(window.process.exitCode(), window.process.exitStatus())
    assert not (tmp_path/'cancelled-parallel').exists()
    assert json.loads(window.job_path.read_text())['status'] == 'cancelled'
    window.close()


def test_import_freezes_paths_and_preserves_explicit_metadata(tmp_path):
    image = tmp_path/'light.tiff'
    image.write_bytes(b'fixture')
    recipe = dict(frames=[dict(path='light.tiff', overrides=dict(state='linear_rgb'))],
                  reference=None, calibration=dict(bias=[dict(path='light.tiff', overrides=dict(role='bias'))]))
    before = json.dumps(recipe)
    temporary = tmp_path/'temporary'
    temporary.mkdir()
    path, job = create_recipe_job(recipe, tmp_path, tmp_path/'output', temporary=temporary)
    frozen = json.loads((path.parent/'recipe.json').read_text())
    assert 'reference' not in frozen
    assert frozen['frames'][0]['path'] == str(image)
    assert frozen['frames'][0]['overrides'] == {'state': 'linear_rgb'}
    assert frozen['calibration']['bias'][0]['overrides'] == {'role': 'bias'}
    assert path.is_relative_to(temporary)
    assert Path(frozen['processing']['work_directory']).is_relative_to(temporary)
    assert not Path(job['result']).is_relative_to(temporary)
    assert frozen['processing']['mode'] == 'disk'
    assert json.dumps(recipe) == before


@pytest.mark.parametrize('change', [dict(reference=True), dict(reference=1), dict(frames=[]),
    dict(frames=[dict(path='missing.fits')]), dict(calibration=dict(unknown=[])), dict(method='aesthetic')])
def test_import_invalid_recipe_does_not_create_job(tmp_path, change):
    image = tmp_path/'light.fits'
    image.write_bytes(b'fixture')
    recipe = dict(frames=[dict(path=str(image))], reference=0)
    recipe.update(change)
    with pytest.raises(ValueError):
        create_recipe_job(recipe, tmp_path, tmp_path/'output')
    assert not (tmp_path/'.coronahdr-jobs').exists()


def test_import_dialog_assigns_roles_without_inventing_state(app, tmp_path):
    image = tmp_path/'light.nef'
    image.write_bytes(b'fixture')
    dialog = ImportDialog()
    dialog.add_paths('light', [image, image])
    dialog.add_paths('bias', [image])
    recipe = json.loads(dialog.editor.toPlainText())
    assert recipe['frames'] == [dict(path=str(image))]
    assert recipe['calibration']['bias'] == [dict(path=str(image), overrides=dict(role='bias'))]
    dialog.destination.setText(str(tmp_path/'output'))
    dialog.temporary.setText(str(tmp_path))
    dialog.submit()
    assert dialog.job['status'] == 'ready'
    dialog.close()


def test_output_accepts_any_drive_but_never_replaces_existing_paths(tmp_path):
    assert output_path(Path.cwd()/'internal-output-must-not-exist').parent == Path.cwd()
    assert output_path(tmp_path) == tmp_path.resolve()
    (tmp_path/'link').symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError, match='never replaced'):
        output_path(tmp_path/'link')
    with pytest.raises(ValueError, match='existing parent'):
        output_path(tmp_path/'missing'/'output')


@pytest.mark.parametrize('method', ['weighted', 'aesthetic'])
def test_separate_temporary_and_output_locations(source, tmp_path, method):
    root, review = source
    temporary, final = tmp_path/'temporary-drive', tmp_path/'output-drive'
    temporary.mkdir()
    final.mkdir()
    path, job = create_job(root, review, final/'product', [0, 1], method, temporary=temporary)
    assert path.is_relative_to(temporary)
    result = Path(job['result'])
    assert result.is_relative_to(final) and not result.is_relative_to(temporary)
    if method == 'weighted':
        recipe = json.loads((path.parent/'recipe.json').read_text())
        assert Path(recipe['processing']['work_directory']).is_relative_to(temporary)
    else:
        index = job['arguments'].index('--work-directory')
        assert Path(job['arguments'][index+1]).is_relative_to(temporary)
    result.mkdir()
    (result/'report.json').write_text('{}')
    for name in ('master.fits', 'master.tiff'):
        (result/name).write_bytes(b'finished product')
    publish_job(path)
    assert (final/'product/master.tiff').read_bytes() == b'finished product'
    assert (final/'product').resolve().is_relative_to(final)


def test_invalid_temporary_location_creates_no_job(source, tmp_path):
    root, review = source
    with pytest.raises(ValueError, match='temporary files folder'):
        create_job(root, review, tmp_path/'output', [0, 1], 'weighted', temporary=tmp_path/'missing')
    assert not (tmp_path/'.coronahdr-jobs').exists()


def test_folder_job_uses_selected_storage_without_astro_profile(tmp_path, monkeypatch):
    import coronahdr.desktop_jobs as jobs

    project = tmp_path/'project'
    project.mkdir()
    monkeypatch.setattr(jobs, '__file__', str(project/'src/coronahdr/desktop_jobs.py'))

    lights, temporary, output = [tmp_path/name for name in ('Lights', 'Temporary', 'Output')]
    for directory in (lights, temporary, output):
        directory.mkdir()
    (lights/'image.CR3').write_bytes(b'Original placeholder; no decoding in job creation')
    path, job = jobs.create_folder_job({'light': str(lights)}, temporary=temporary,
                                 destination=output/'product', mode='scientific')
    assert path.is_relative_to(temporary)
    assert job['work'] == str(path.parent)
    assert Path(job['result']).is_relative_to(temporary)
    assert job['publication'] == 'standalone_final_files_v1'
    assert job['destination'] == str(output/'product')
    assert (output/'product'/'CoronaHDR_settings.json').is_file()


def test_folder_job_defaults_to_sibling_output_and_temporary_result(tmp_path, monkeypatch):
    import coronahdr.desktop_jobs as jobs
    lights, temporary = tmp_path/'Lights', tmp_path/'temporary'
    lights.mkdir()
    temporary.mkdir()
    (lights/'image.CR3').write_bytes(b'fixture')
    monkeypatch.setattr('coronahdr.folder_import.discover', lambda *args: [lights/'image.CR3'])
    path, job = jobs.create_folder_job({'light': str(lights)}, temporary=temporary)
    assert Path(job['destination']) == tmp_path/'HDR_output'
    assert Path(job['result']).is_relative_to(temporary)
    assert path.is_relative_to(temporary)
    assert (tmp_path/'HDR_output'/'CoronaHDR_settings.json').is_file()


def test_new_folder_job_has_no_separate_prominence_option(app, tmp_path, monkeypatch):
    from coronahdr.desktop_folders import FolderDialog
    from coronahdr.desktop_jobs import create_folder_job
    lights = tmp_path/'Lights'
    lights.mkdir()
    (lights/'one.CR3').write_bytes(b'fixture')
    monkeypatch.setattr('coronahdr.folder_import.discover', lambda *args: [lights/'one.CR3'])
    dialog = FolderDialog()
    assert not hasattr(dialog, 'prominence')
    path, job = create_folder_job({'light': lights}, temporary=tmp_path, destination=tmp_path/'output')
    assert 'prominence' not in job
    assert 'prominence' not in json.loads(Path(job['settings_path']).read_text())['settings']
    assert path.is_file()
    dialog.close()


def test_settings_load_into_editable_folder_dialog(app, tmp_path, monkeypatch):
    from coronahdr.desktop_jobs import create_folder_job
    from coronahdr.desktop_folders import FolderDialog
    from PySide6.QtWidgets import QFileDialog
    lights = tmp_path/'Lights'
    lights.mkdir()
    (lights/'one.CR3').write_bytes(b'fixture')
    monkeypatch.setattr('coronahdr.folder_import.discover', lambda *args: [lights/'one.CR3'])
    _, job = create_folder_job({'light': lights}, temporary=tmp_path, destination=tmp_path/'output',
                               base_name='Eclipse', mode='both', detect_hot_pixels=True, lunar_margin=-2)
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (job['settings_path'], ''))
    dialog = FolderDialog()
    emitted = []
    dialog.settings_loaded.connect(lambda path, settings: emitted.append((path, settings)))
    dialog.load_settings()
    assert dialog.folders['light'].text() == str(lights)
    assert dialog.base_name.text() == 'Eclipse' and dialog.hdr_mode.currentData() == 'both'
    assert dialog.lunar_margin.value() == -2 and dialog.detect_hot_pixels.isChecked()
    assert emitted[0][0] == job['settings_path'] and emitted[0][1]['mode'] == 'both'
    dialog.base_name.setText('New name')
    assert dialog.base_name.text() == 'New name' and dialog.job is None
    dialog.close()


def test_saved_settings_load_matching_images_and_published_log(app, tmp_path, monkeypatch):
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPlainTextEdit
    output = tmp_path/'output'
    output.mkdir()
    settings = output/'Eclipse_settings.json'
    settings.write_text('{}')
    for name in ('Eclipse_aesthetic.fits', 'Eclipse_prominence.fits'):
        (output/name).write_bytes(b'fixture')
    log = output/'Eclipse_both.log'
    log.write_text('Completed job\nImportant detail')
    options = dict(base_name='Eclipse', mode='both', output_format='fits', prominence=True)
    window = CoronaWindow()
    paths = []
    monkeypatch.setattr(window.hdr_preview, 'set_path', lambda role, path: paths.append((role, path)))
    window.show_saved_settings(str(settings), options)
    for _ in range(100):
        app.processEvents()
        if not window.saved_log_tasks:
            break
        QTest.qWait(10)
    assert paths == [('corona', output/'Eclipse_aesthetic.fits')]
    assert 'Important detail' in window.log.toPlainText()
    assert window.process is None and window.job_path is None
    window.copy_job_log_path()
    assert QApplication.clipboard().text() == str(log)
    window.copy_job_log()
    assert QApplication.clipboard().text() == log.read_text()
    window.view_job_log()
    assert 'Important detail' in window.log_dialog.findChild(QPlainTextEdit).toPlainText()
    window.log_dialog.close()
    window.close()


def test_saved_settings_reports_missing_outputs_without_starting_job(app, tmp_path, monkeypatch):
    from PySide6.QtTest import QTest
    window = CoronaWindow()
    paths = []
    monkeypatch.setattr(window.hdr_preview, 'set_path', lambda role, path: paths.append((role, path)))
    window.show_saved_settings(str(tmp_path/'Eclipse_settings.json'),
                               dict(base_name='Eclipse', mode='scientific', output_format='both', prominence=True))
    for _ in range(100):
        app.processEvents()
        if not window.saved_log_tasks:
            break
        QTest.qWait(10)
    assert paths == [('corona', None)]
    assert 'not found' in window.hdr_preview.labels['corona'].text().lower()
    assert 'unavailable' in window.status.text().lower()
    assert not window.copy_log_button.isEnabled() and window.process is None
    window.close()


def test_old_saved_log_worker_cannot_replace_newer_log(app, tmp_path, monkeypatch):
    import threading
    from PySide6.QtTest import QTest
    first, second = tmp_path/'first', tmp_path/'second'
    first.mkdir(); second.mkdir()
    (first/'Eclipse_aesthetic.log').write_text('old log')
    (second/'Eclipse_aesthetic.log').write_text('new log')
    started, release = threading.Event(), threading.Event()
    original = Path.read_text
    def delayed(path, *args, **kwargs):
        if path == first/'Eclipse_aesthetic.log':
            started.set()
            release.wait(5)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', delayed)
    window = CoronaWindow()
    monkeypatch.setattr(window.hdr_preview, 'set_path', lambda *args: None)
    options = dict(base_name='Eclipse', mode='aesthetic', output_format='fits', prominence=False)
    try:
        window.show_saved_settings(str(first/'Eclipse_settings.json'), options)
        assert started.wait(5)
        window.show_saved_settings(str(second/'Eclipse_settings.json'), options)
        for _ in range(200):
            app.processEvents()
            if window.log.toPlainText() == 'new log':
                break
            QTest.qWait(10)
        assert window.log.toPlainText() == 'new log'
    finally:
        release.set()
        for _ in range(200):
            app.processEvents()
            if not window.saved_log_tasks:
                break
            QTest.qWait(10)
    assert window.log.toPlainText() == 'new log'
    window.close()


def test_settings_filename_collision_does_not_overwrite(tmp_path, monkeypatch):
    from coronahdr.desktop_jobs import create_folder_job, OutputNameConflict
    lights = tmp_path/'Lights'
    lights.mkdir()
    (lights/'one.CR3').write_bytes(b'fixture')
    destination = tmp_path/'output'
    destination.mkdir()
    target = destination/'Eclipse_settings.json'
    target.write_text('keep')
    monkeypatch.setattr('coronahdr.folder_import.discover', lambda *args: [lights/'one.CR3'])
    with pytest.raises(OutputNameConflict):
        create_folder_job({'light': lights}, temporary=tmp_path, destination=destination, base_name='Eclipse')
    assert target.read_text() == 'keep'


def test_bad_settings_leave_form_unchanged(app, tmp_path, monkeypatch):
    from coronahdr.desktop_folders import FolderDialog
    from PySide6.QtWidgets import QFileDialog
    target = tmp_path/'settings.json'
    target.write_text('{ invalid')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(target), ''))
    dialog = FolderDialog()
    dialog.base_name.setText('Keep this')
    dialog.load_settings()
    assert dialog.base_name.text() == 'Keep this' and 'Cannot read settings' in dialog.status.text()
    dialog.close()


def test_processing_prompt_wraps_and_stays_within_screen(app):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QLabel
    from coronahdr.desktop_paths import ask_choice
    observed = []
    message = 'Aesthetic HDR: the saved geometry supports a shared detector field. ' * 8
    def inspect():
        dialog = app.activeModalWidget()
        label = next(item for item in dialog.findChildren(QLabel) if item.text() == message)
        observed.append((dialog.width(), dialog.screen().availableGeometry().width(), label.wordWrap()))
        dialog.reject()
    QTimer.singleShot(0, inspect)
    _, accepted = ask_choice(None, 'Review processing decision', message, ['Crop', 'Keep'])
    assert not accepted and observed[0][0] <= min(680, observed[0][1]) and observed[0][2]


def test_folder_job_rejects_temporary_files_inside_final_output(tmp_path, monkeypatch):
    import coronahdr.desktop_jobs as jobs
    lights, output = tmp_path/'Lights', tmp_path/'HDR_output'
    lights.mkdir()
    output.mkdir()
    (lights/'image.CR3').write_bytes(b'fixture')
    monkeypatch.setattr('coronahdr.folder_import.discover', lambda *args: [lights/'image.CR3'])
    with pytest.raises(ValueError, match='temporary files folder outside'):
        jobs.create_folder_job({'light': lights}, temporary=output, destination=output)
    assert not list(output.iterdir())


@pytest.mark.parametrize('mode', ['scientific', 'aesthetic', 'both'])
def test_folder_publication_contains_only_independent_final_files(tmp_path, monkeypatch, mode):
    import coronahdr.desktop_jobs as jobs
    lights, temporary, destination = tmp_path/'Lights', tmp_path/'temp', tmp_path/'HDR_output'
    lights.mkdir()
    temporary.mkdir()
    destination.mkdir()
    (destination/'unrelated.txt').write_text('keep')
    (lights/'image.CR3').write_bytes(b'fixture')
    monkeypatch.setattr('coronahdr.folder_import.discover', lambda *args: [lights/'image.CR3'])
    path, job = jobs.create_folder_job({'light': lights}, mode=mode, base_name='Eclipse',
                                      temporary=temporary, destination=destination)
    result = Path(job['result'])
    result.mkdir()
    (result/'report.json').write_text('{}')
    modes = ('scientific', 'aesthetic') if mode == 'both' else (mode,)
    for item in modes:
        folder = result/item if mode == 'both' else result
        folder.mkdir(exist_ok=True)
        (folder/'report.json').write_text('{}')
        for ext in ('fits', 'tiff'):
            (folder/f'master.{ext}').write_bytes(f'{item}-{ext}'.encode())
        (folder/'support.fits').write_bytes(b'mask')
    jobs.publish_job(path)
    published = json.loads(path.read_text())
    names = {f'Eclipse_{item}.{ext}' for item in modes for ext in ('fits', 'tiff')}
    assert {p.name for p in destination.iterdir()} == names | {'unrelated.txt', 'Eclipse_settings.json'}
    assert jobs.is_published(published)
    for item in modes:
        for ext in ('fits', 'tiff'):
            target = destination/f'Eclipse_{item}.{ext}'
            original = (result/item if mode == 'both' else result)/f'master.{ext}'
            assert not target.is_symlink() and not target.samefile(original)
            original.write_bytes(b'changed temporary copy')
            assert target.read_bytes() == f'{item}-{ext}'.encode()
    assert jobs.is_published(published)
    import shutil
    shutil.rmtree(result)
    assert jobs.is_published(published)
    assert all((destination/name).is_file() for name in names)


def test_folder_publication_rejects_corrupt_copies(tmp_path, monkeypatch):
    import coronahdr.desktop_jobs as jobs
    result, destination = tmp_path/'result', tmp_path/'HDR_output'
    result.mkdir()
    (result/'report.json').write_text('{}')
    for ext in ('fits', 'tiff'):
        (result/f'master.{ext}').write_bytes(b'original')
    path = tmp_path/'job.json'
    path.write_text(json.dumps(dict(result=str(result), destination=str(destination), hdr_mode='aesthetic',
                                    publication='standalone_final_files_v1')))
    monkeypatch.setattr(jobs.shutil, 'copyfileobj', lambda src, dst, **kwargs: dst.write(b'corrupt'))
    with pytest.raises(OSError, match='failed verification'):
        jobs.publish_job(path)
    assert not list(destination.iterdir())


def test_folder_publication_rolls_back_its_own_partial_files(tmp_path, monkeypatch):
    import coronahdr.desktop_jobs as jobs
    result, destination = tmp_path/'result', tmp_path/'HDR_output'
    result.mkdir()
    destination.mkdir()
    (destination/'keep.txt').write_text('keep')
    (result/'report.json').write_text('{}')
    for ext in ('fits', 'tiff'):
        (result/f'master.{ext}').write_bytes(b'data')
    path = tmp_path/'job.json'
    path.write_text(json.dumps(dict(result=str(result), destination=str(destination), hdr_mode='scientific',
                                    publication='standalone_final_files_v1')))
    copy = jobs.shutil.copyfileobj
    calls = []
    def fail_second(src, dst, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError('simulated output failure')
        return copy(src, dst, **kwargs)
    monkeypatch.setattr(jobs.shutil, 'copyfileobj', fail_second)
    with pytest.raises(OSError, match='simulated'):
        jobs.publish_job(path)
    assert [p.name for p in destination.iterdir()] == ['keep.txt']


def test_folder_job_accepts_both_and_validates_output_base_name(tmp_path, monkeypatch):
    import coronahdr.desktop_jobs as jobs
    project = tmp_path/'project'
    project.mkdir()
    monkeypatch.setattr(jobs, '__file__', str(project/'src/coronahdr/desktop_jobs.py'))
    lights = tmp_path/'Lights'
    lights.mkdir()
    (lights/'image.CR3').write_bytes(b'fixture')
    for invalid in ('../outside', 'a/b', 'a\\b', '..', 'a\nname', '😀'*100):
        with pytest.raises(ValueError, match='base name'):
            jobs.create_folder_job({'light': lights}, mode='both', base_name=invalid,
                                   temporary=tmp_path, destination=tmp_path/'product')
    _, job = jobs.create_folder_job({'light': lights}, mode='both', base_name=' Eclipse 2026 ',
                                    temporary=tmp_path, destination=tmp_path/'product')
    assert job['hdr_mode'] == 'both'
    assert job['output_base_name'] == 'Eclipse 2026'
    _, visual_job = jobs.create_folder_job({'light': lights}, mode='both', alignment='ael_visual',
                                           temporary=tmp_path, destination=tmp_path/'visual-product')
    assert visual_job['alignment'] == 'ael_visual'


@pytest.mark.parametrize('mode', ['scientific', 'aesthetic', 'both'])
@pytest.mark.parametrize('output_format', ['fits', 'tiff', 'both'])
def test_selected_final_formats_are_independent_copies(tmp_path, monkeypatch, mode, output_format):
    import coronahdr.desktop_jobs as jobs
    lights = tmp_path/'Lights'
    lights.mkdir()
    (lights/'image.CR3').write_bytes(b'fixture')
    monkeypatch.setattr('coronahdr.folder_import.discover', lambda *args: [lights/'image.CR3'])
    destination = tmp_path/'output'
    path, job = jobs.create_folder_job({'light': lights}, mode=mode, temporary=tmp_path,
        destination=destination, base_name='Eclipse', output_format=output_format)
    result = Path(job['result'])
    result.mkdir()
    (result/'report.json').write_text('{}')
    modes = ('scientific', 'aesthetic') if mode == 'both' else (mode,)
    for item in modes:
        folder = result/item if mode == 'both' else result
        folder.mkdir(exist_ok=True)
        (folder/'report.json').write_text('{}')
        for ext in ('fits', 'tiff'):
            (folder/f'master.{ext}').write_bytes(f'{item}-{ext}'.encode())
    jobs.publish_job(path)
    published = json.loads(path.read_text())
    extensions = ('fits', 'tiff') if output_format == 'both' else (output_format,)
    expected = {f'Eclipse_{item}.{ext}' for item in modes for ext in extensions}
    assert {p.name for p in destination.iterdir()} == expected | {'Eclipse_settings.json'}
    assert jobs.is_published(published)
    for name in expected:
        assert not (destination/name).is_symlink()
    import shutil
    shutil.rmtree(result)
    assert jobs.is_published(published)


def test_selected_format_collision_checks_only_requested_files(tmp_path):
    from coronahdr.desktop_jobs import validate_folder_targets, OutputNameConflict
    (tmp_path/'Eclipse_aesthetic.tiff').write_bytes(b'keep')
    validate_folder_targets(tmp_path, 'aesthetic', 'Eclipse', standalone=True, output_format='fits')
    with pytest.raises(OutputNameConflict):
        validate_folder_targets(tmp_path, 'aesthetic', 'Eclipse', standalone=True, output_format='tiff')


def test_named_masters_do_not_replace_existing_files(tmp_path):
    from coronahdr.desktop_jobs import named_masters
    source, output = tmp_path/'source', tmp_path/'output'
    source.mkdir()
    output.mkdir()
    for ext in ('fits', 'tiff'):
        (source/f'master.{ext}').write_bytes(ext.encode())
    assert named_masters(source, output, 'Eclipse', 'scientific') == [
        'Eclipse_scientific.fits', 'Eclipse_scientific.tiff']
    with pytest.raises(FileExistsError):
        named_masters(source, output, 'Eclipse', 'scientific')
    assert (output/'Eclipse_scientific.fits').read_bytes() == b'fits'


def test_named_masters_copy_when_hard_links_are_unavailable(tmp_path, monkeypatch):
    import coronahdr.desktop_jobs as jobs
    source, output = tmp_path/'source', tmp_path/'output'
    source.mkdir()
    output.mkdir()
    for ext in ('fits', 'tiff'):
        (source/f'master.{ext}').write_bytes(ext.encode())
    def unsupported(*args):
        raise OSError('Filesystem does not support hard links')
    monkeypatch.setattr(jobs.os, 'link', unsupported)
    jobs.named_masters(source, output, 'Eclipse', 'aesthetic')
    assert (output/'Eclipse_aesthetic.fits').read_bytes() == b'fits'
    assert (output/'Eclipse_aesthetic.tiff').read_bytes() == b'tiff'


def test_folder_output_collisions_are_caught_before_starting(tmp_path):
    from coronahdr.desktop_jobs import validate_folder_targets
    (tmp_path/'Eclipse_aesthetic.tiff').write_bytes(b'keep')
    with pytest.raises(ValueError, match='Eclipse_aesthetic.tiff'):
        validate_folder_targets(tmp_path, 'both', 'Eclipse')
    assert (tmp_path/'Eclipse_aesthetic.tiff').read_bytes() == b'keep'
    with pytest.raises(ValueError, match='base name'):
        validate_folder_targets(tmp_path, 'both', '../outside')


def test_existing_output_log_requires_new_base_name(tmp_path):
    from coronahdr.desktop_jobs import OutputNameConflict, validate_folder_targets
    existing = tmp_path/'Eclipse_both.log'
    existing.write_text('keep')
    with pytest.raises(OutputNameConflict, match='different base name'):
        validate_folder_targets(tmp_path, 'both', 'Eclipse', standalone=True)
    assert existing.read_text() == 'keep'
    validate_folder_targets(tmp_path, 'scientific', 'Eclipse', standalone=True)


def test_raw_declaration_is_explicit_and_includes_calibrations(app, tmp_path):
    dialog = ImportDialog()
    dialog.add_paths('light', [tmp_path/'light.NEF', tmp_path/'linear.tiff'])
    dialog.add_paths('bias', [tmp_path/'bias.nef'])
    before = json.loads(dialog.editor.toPlainText())
    assert 'overrides' not in before['frames'][0]
    dialog.declare_raw_cfa()
    recipe = json.loads(dialog.editor.toPlainText())
    assert recipe['frames'][0]['overrides'] == dict(state='linear_cfa')
    assert recipe['frames'][1] == before['frames'][1]
    assert recipe['calibration']['bias'][0]['overrides'] == dict(role='bias', state='linear_cfa')
    dialog.close()


def test_raw_declaration_does_not_silently_replace_conflicting_state(app):
    dialog = ImportDialog()
    recipe = dict(frames=[dict(path='first.nef'), dict(path='second.nef', overrides=dict(state='linear_rgb'))])
    dialog.editor.setPlainText(json.dumps(recipe))
    before = dialog.editor.toPlainText()
    dialog.declare_raw_cfa()
    assert dialog.editor.toPlainText() == before
    assert 'Conflicting state' in dialog.status.text()
    dialog.close()


def test_response_group_declaration_preserves_exposure_and_calibrations(app):
    dialog = ImportDialog()
    recipe = dict(frames=[dict(path='a.nef', overrides=dict(exposure_s=.01)), dict(path='b.nef')],
                  calibration=dict(bias=[dict(path='bias.nef', overrides=dict(role='bias'))]))
    dialog.editor.setPlainText(json.dumps(recipe))
    dialog.declare_response_group()
    assert json.loads(dialog.editor.toPlainText()) == recipe
    dialog.response_group.setText(' Nikon-Z50II-ISO800-f8 ')
    dialog.declare_response_group()
    declared = json.loads(dialog.editor.toPlainText())
    assert all(item['overrides']['response_group'] == 'Nikon-Z50II-ISO800-f8' for item in declared['frames'])
    assert declared['frames'][0]['overrides']['exposure_s'] == .01
    assert declared['calibration'] == recipe['calibration']
    dialog.close()


def test_response_group_conflict_does_not_modify_recipe(app):
    dialog = ImportDialog()
    recipe = dict(frames=[dict(path='a.nef'), dict(path='b.nef', overrides=dict(response_group='other'))])
    dialog.editor.setPlainText(json.dumps(recipe))
    dialog.response_group.setText('new-group')
    dialog.declare_response_group()
    assert json.loads(dialog.editor.toPlainText()) == recipe
    assert 'Conflicting response_group' in dialog.status.text()
    dialog.close()


def test_default_import_only_exposes_folder_selection(app):
    from coronahdr.desktop_folders import FolderDialog
    window = CoronaWindow()
    assert window.form_widget.isHidden()
    assert window.table.isHidden()
    assert window.start_button.isHidden()
    dialog = FolderDialog(window)
    assert set(dialog.folders) == {'light', 'bias', 'dark', 'flat', 'dark_flat'}
    assert not hasattr(dialog, 'editor')
    assert not hasattr(dialog, 'response_group')
    dialog.close()
    window.close()


def test_folder_dialog_defaults_beside_lights_and_preserves_custom_destination(app, tmp_path, monkeypatch):
    import coronahdr.desktop_folders as module
    dialog = module.FolderDialog()
    first, second = tmp_path/'first/Lights', tmp_path/'second/Lights'
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    dialog.folders['light'].setText(str(first))
    assert Path(dialog.destination.text()) == first.parent/'HDR_output'
    dialog.folders['light'].setText(str(second))
    assert Path(dialog.destination.text()) == second.parent/'HDR_output'
    custom = tmp_path/'chosen'
    custom.mkdir()
    monkeypatch.setattr(module, 'choose_output_folder', lambda *args: str(custom))
    dialog.choose_storage(dialog.destination, False)
    dialog.folders['light'].setText(str(first))
    assert Path(dialog.destination.text()) == custom
    dialog.close()


def test_folder_dialog_asks_for_new_name_when_output_exists(app, tmp_path, monkeypatch):
    import coronahdr.desktop_folders as module
    lights, temporary, destination = tmp_path/'Lights', tmp_path/'temp', tmp_path/'HDR_output'
    for directory in (lights, temporary, destination):
        directory.mkdir()
    (lights/'image.CR3').write_bytes(b'fixture')
    monkeypatch.setattr('coronahdr.folder_import.discover', lambda *args: [lights/'image.CR3'])
    existing = destination/'CoronaHDR_aesthetic.tiff'
    existing.write_bytes(b'keep')
    prompts = []
    def choose_name(*args, **kwargs):
        prompts.append(args[2])
        return 'New name', True
    monkeypatch.setattr(module, 'ask_text', choose_name)
    dialog = module.FolderDialog()
    dialog.folders['light'].setText(str(lights))
    dialog.temporary.setText(str(temporary))
    dialog.submit()
    assert prompts and 'different base name' in prompts[0]
    assert dialog.job['output_base_name'] == 'New name'
    assert existing.read_bytes() == b'keep'
    dialog.close()


def test_folder_bias_fallback_and_mode_controls(app):
    from coronahdr.desktop_folders import FolderDialog
    dialog = FolderDialog()
    layout = dialog.alignment.parentWidget().layout()
    assert layout.indexOf(dialog.alignment) < layout.indexOf(dialog.detect_hot_pixels) < layout.indexOf(dialog.hdr_mode)
    assert not dialog.bias_fallback.isEnabled()
    dialog.folders['bias'].setText('/original/Bias')
    assert dialog.bias_fallback.isEnabled()
    dialog.bias_fallback.setChecked(True)
    assert 'Thermal current' in dialog.calibration_warning.text()
    assert 'You may continue' in dialog.calibration_warning.text()
    dialog.hdr_mode.setCurrentIndex(1)
    assert dialog.hdr_mode.currentData() == 'scientific'
    dialog.hdr_mode.setCurrentIndex(2)
    assert dialog.hdr_mode.currentData() == 'both'
    assert dialog.output_format.currentData() == 'both'
    dialog.output_format.setCurrentIndex(1)
    assert dialog.output_format.currentData() == 'fits'
    dialog.output_format.setCurrentIndex(2)
    assert dialog.output_format.currentData() == 'tiff'
    dialog.folders['dark'].setText('/original/Darks')
    assert not dialog.bias_fallback.isEnabled()
    assert not dialog.bias_fallback.isChecked()
    dialog.close()


def test_warnings_remain_visible_without_interrupting_processing(app):
    window = CoronaWindow()
    window.add_warning('No thermal correction')
    window.add_warning('No thermal correction')
    window.add_warning('Three frames rejected')
    assert len(window.warnings) == 2
    assert not window.warning_panel.isHidden()
    assert 'Three frames rejected' in window.warning_panel.text()
    window.close()


def test_related_folder_offer_preserves_existing_and_ambiguous_choices(app, monkeypatch, tmp_path):
    import coronahdr.desktop_folders as module
    from PySide6.QtWidgets import QComboBox, QDialog
    dialog = module.FolderDialog()
    monkeypatch.setattr(module, 'related_folders', lambda path: dict(
        bias=[tmp_path/'Bias'], flat=[tmp_path/'Flats'], dark=[tmp_path/'Dark1', tmp_path/'Dark2']))
    def accept_with_tooltips(self):
        choices = self.findChildren(QComboBox)
        assert choices and all(choice.toolTip() for choice in choices)
        assert all(choice.itemData(i, Qt.ItemDataRole.ToolTipRole) for choice in choices for i in range(choice.count()))
        return 1
    monkeypatch.setattr(QDialog, 'exec', accept_with_tooltips)
    dialog.folders['bias'].setText('existing-bias')
    dialog.offer_related(tmp_path/'Lights')
    assert dialog.folders['bias'].text() == 'existing-bias'
    assert dialog.folders['flat'].text() == str(tmp_path/'Flats')
    assert dialog.folders['dark'].text() == ''
    monkeypatch.setattr(QDialog, 'exec', lambda self: 0)
    dialog.folders['flat'].clear()
    dialog.offer_related(tmp_path/'Lights')
    assert dialog.folders['flat'].text() == ''
    dialog.close()


def test_intermediate_checkboxes_are_optional_and_frozen_in_job(app, monkeypatch, tmp_path):
    import coronahdr.desktop_folders as module
    captured = {}

    def create(folders, **options):
        captured.update(options)
        return tmp_path/'job.json', {'status': 'ready'}

    monkeypatch.setattr(module, 'create_folder_job', create)
    dialog = module.FolderDialog()
    assert not any(box.isChecked() for box in dialog.save_intermediates.values())
    dialog.save_intermediates['calibrated'].setChecked(True)
    dialog.save_intermediates['registered'].setChecked(True)
    dialog.detect_hot_pixels.setChecked(True)
    dialog.hdr_mode.setCurrentIndex(2)
    dialog.base_name.setText('Total Eclipse')
    dialog.temporary.setText(str(tmp_path))
    dialog.destination.setText(str(tmp_path/'chosen-output'))
    dialog.submit()
    assert captured['save_intermediates'] == ['calibrated', 'registered']
    assert captured['detect_hot_pixels'] is True
    assert captured['mode'] == 'both'
    assert captured['base_name'] == 'Total Eclipse'
    assert captured['temporary'] == str(tmp_path)
    assert captured['destination'] == str(tmp_path/'chosen-output')
    dialog.close()


def test_parameter_wheel_does_not_edit_but_keyboard_still_works(app):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent, QKeyEvent
    from coronahdr.desktop_style import ScrollSafeComboBox, ScrollSafeSpinBox
    combo = ScrollSafeComboBox()
    combo.addItems(['Aesthetic', 'Scientific'])
    spin = ScrollSafeSpinBox()
    spin.setRange(-4, 4)
    for widget in (combo, spin):
        wheel = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, -120),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(widget, wheel)
        assert not wheel.isAccepted()
    assert combo.currentIndex() == 0
    assert spin.value() == 0
    QApplication.sendEvent(combo, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier))
    QApplication.sendEvent(spin, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier))
    assert combo.currentIndex() == 1
    assert spin.value() == 1
    combo.close()
    spin.close()


def test_preview_refits_when_window_and_splitter_resize(app, source):
    from PySide6.QtWidgets import QSplitter
    root, review = source
    window = CoronaWindow(root, review)
    window.show()
    app.processEvents()
    before = window.preview.size()
    window.resize(1600, 1000)
    app.processEvents()
    assert window.preview.width() > before.width()
    splitter = window.findChild(QSplitter)
    splitter.setSizes([800, 800])
    app.processEvents()
    assert window.preview.image is not None
    window.close()


@pytest.mark.parametrize('seconds, decimal, nominal', [
    (0.002, '0.00200', '1/500 s'),
    (1 / 160, '0.00625', '1/160 s'),
    (0.0049, '0.00490', '1/200 s'),
    (0.0041, '0.00410', '1/250 s'),
    (1 / 60, '0.01667', '1/60 s'),
    (1 / 15, '0.06667', '1/15 s'),
    (2.45, '2.45000', '2.5 s'),
    (60, '60.00000', '60.00000 s'),
    (None, 'Unknown', 'Unknown'),
    (float('nan'), 'Unknown', 'Unknown'),
    (float('inf'), 'Unknown', 'Unknown'),
    (0, 'Unknown', 'Unknown'),
    (-1, 'Unknown', 'Unknown'),
])
def test_exposure_display(seconds, decimal, nominal):
    from coronahdr.desktop import exposure_labels
    assert exposure_labels(seconds) == (decimal, nominal)


def test_frame_table_shows_measured_and_nominal_exposure(app, source):
    root, review = source
    window = CoronaWindow(root, review)
    assert window.table.columnCount() == 7
    assert window.table.item(0, 3).text() == '1.00000'
    assert window.table.item(0, 4).text() == '1/1 s'
    assert window.table.item(0, 5).text() == 'linear_rgb'
    assert window.table.item(0, 6).text() == 'Reference'
    window.close()


def test_output_picker_returns_existing_folder(app, tmp_path, monkeypatch):
    import coronahdr.desktop_paths as module
    monkeypatch.setattr(module.QFileDialog, 'getExistingDirectory', lambda *args: str(tmp_path))
    assert module.choose_output_folder(None, str(tmp_path/'proposed')) == str(tmp_path)
    assert not list(tmp_path.iterdir())


def test_output_picker_cancel_keeps_selection(app, monkeypatch):
    import coronahdr.desktop_paths as module
    monkeypatch.setattr(module.QFileDialog, 'getExistingDirectory', lambda *args: '')
    window = CoronaWindow()
    window.destination.setText('/unchanged/result')
    window.browse_destination()
    assert window.destination.text() == '/unchanged/result'
    window.close()


@pytest.mark.parametrize('fail_publication', [False, True])
def test_publish_into_existing_folder_preserves_files(source, tmp_path, monkeypatch, fail_publication):
    from coronahdr.desktop_jobs import is_published
    root, review = source
    output = tmp_path/'existing-output'
    output.mkdir()
    (output/'notes.txt').write_text('keep me')
    path, job = create_job(root, review, output, [0, 1], 'aesthetic')
    result = Path(job['result'])
    result.mkdir()
    (result/'report.json').write_text('{}')
    for name in ('master.tiff', 'master.fits'):
        (result/name).write_bytes(b'fixture')
    assert not is_published(job)
    if fail_publication:
        original = Path.symlink_to
        calls = []
        def failing_link(target, *args, **kwargs):
            calls.append(target)
            if len(calls) == 2:
                raise OSError('Simulated publication failure')
            return original(target, *args, **kwargs)
        monkeypatch.setattr(Path, 'symlink_to', failing_link)
        with pytest.raises(OSError, match='Simulated'):
            publish_job(path)
        assert [p.name for p in output.iterdir()] == ['notes.txt']
        assert not is_published(job)
    else:
        publish_job(path)
        assert output.is_dir() and not output.is_symlink()
        assert (output/'master.tiff').read_bytes() == b'fixture'
        assert is_published(job)
        assert json.loads(path.read_text())['status'] == 'completed'
    assert (output/'notes.txt').read_text() == 'keep me'


def test_existing_output_with_result_is_rejected_before_job_creation(tmp_path):
    (tmp_path/'master.tiff').write_bytes(b'original')
    with pytest.raises(ValueError, match='never replaced'):
        output_path(tmp_path)
    assert (tmp_path/'master.tiff').read_bytes() == b'original'


def test_directory_picker_selects_existing_folder(app, tmp_path):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QFileDialog
    from coronahdr.desktop_paths import choose_output_folder
    observed = []
    def select_folder():
        dialog = app.activeModalWidget()
        observed.append(isinstance(dialog, QFileDialog))
        if isinstance(dialog, QFileDialog):
            observed.append(dialog.fileMode() == QFileDialog.FileMode.Directory)
            dialog.setDirectory(str(tmp_path))
            dialog.accept()
    QTimer.singleShot(50, select_folder)
    # Avoid a hanging test if the picker unexpectedly changes its modality.
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: app.activeModalWidget().reject() if app.activeModalWidget() else None)
    watchdog.start(3000)
    selected = choose_output_folder(None, str(tmp_path))
    watchdog.stop()
    assert observed == [True, True]
    assert Path(selected).resolve() == tmp_path.resolve()


@pytest.mark.parametrize('kind', ['window', 'folders', 'recipe'])
def test_desktop_labels_are_camera_neutral_and_controls_have_help(app, kind):
    from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QComboBox,
        QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QScrollArea, QScrollBar, QSplitter, QTableWidget, QWidget)
    from coronahdr.desktop_folders import FolderDialog
    widget = {'window': CoronaWindow, 'folders': FolderDialog, 'recipe': ImportDialog}[kind]()
    controls = (QAbstractButton, QAbstractSpinBox, QComboBox, QLabel,
                QLineEdit, QPlainTextEdit, QProgressBar, QScrollArea, QScrollBar, QSplitter, QTableWidget)
    for child in widget.findChildren(QWidget):
        if not isinstance(child, controls):
            continue
        assert child.toolTip(), (kind, type(child).__name__, child.objectName())
        if isinstance(child, (QLabel, QAbstractButton)):
            assert 'nikon' not in child.text().lower()
            assert 'canon' not in child.text().lower()
    if kind == 'window':
        assert all(widget.table.horizontalHeaderItem(i).toolTip() for i in range(widget.table.columnCount()))
        assert all(widget.method.itemData(i, Qt.ItemDataRole.ToolTipRole) for i in range(widget.method.count()))
        assert widget.findChild(QSplitter).handle(1).toolTip()
    if kind == 'folders':
        assert all(widget.hdr_mode.itemData(i, Qt.ItemDataRole.ToolTipRole) for i in range(widget.hdr_mode.count()))
    widget.close()


def test_folder_path_tooltips_follow_selection_and_default_output(app, tmp_path):
    from PySide6.QtWidgets import QAbstractButton
    from coronahdr.desktop_folders import FolderDialog
    dialog = FolderDialog()
    assert 'No path selected' in dialog.folders['light'].toolTip()
    lights = tmp_path/'an eclipse'/'Lights'
    dialog.folders['light'].setText(str(lights))
    assert str(lights) in dialog.folders['light'].toolTip()
    assert str(lights.parent/'HDR_output') in dialog.destination.toolTip()
    row = dialog.folders['light'].parentWidget()
    choose = next(button for button in row.findChildren(QAbstractButton) if button.text() == 'Choose…')
    assert str(lights) in choose.toolTip()
    dialog.folders['light'].clear()
    assert 'No path selected' in choose.toolTip()
    dialog.close()


def test_tooltip_paragraph_preserves_text_and_wraps_long_paths(app):
    from PySide6.QtGui import QTextDocument
    from coronahdr.desktop_style import tooltip_paragraph
    text = 'Output <folder> & reports:\n/Volumes/' + 'a'*200
    markup = tooltip_paragraph(text)
    document = QTextDocument()
    document.setHtml(markup)
    assert document.toPlainText().strip().replace('\u200b', '') == text
    assert '&lt;folder&gt;' in markup and '&amp;' in markup
    assert '\u200b' in markup


def test_folder_alignment_choices_are_labeled_and_explain_their_limits(app):
    from coronahdr.desktop_folders import FolderDialog
    dialog = FolderDialog()
    assert dialog.alignment.currentData() == 'conservative'
    assert 'shift-only' in dialog.alignment.currentText().lower()
    assert 'scientific' in dialog.alignment.toolTip().lower()
    dialog.alignment.setCurrentIndex(1)
    assert dialog.alignment.currentData() == 'ael_visual'
    assert 'experimental' in dialog.alignment.currentText().lower()
    dialog.close()


def test_about_window_shows_credits_and_project_links(app):
    from importlib.metadata import version
    from PySide6.QtWidgets import QDialog, QTextBrowser
    window = CoronaWindow()
    window.show()
    app.processEvents()
    assert [action.text() for action in window.more_button.menu().actions()] == ['Resume / revise…', 'Clean up temporary data…', 'About CoronaHDR']
    window.about_button.click()
    app.processEvents()
    dialog = next(child for child in window.findChildren(QDialog) if child.windowTitle() == 'About CoronaHDR')
    content = dialog.findChild(QTextBrowser)
    text, markup = content.toPlainText(), content.toHtml()
    assert dialog.isVisible() and content.openExternalLinks()
    assert 'color:#9cdcfe' in markup.lower().replace(' ', '')
    assert version('coronahdr') in text and 'Francisco Bitto' in text
    assert 'developed using Agentic Eclipse Library' in text
    assert 'Eclipse Image Processor' in text and 'Research and inspiration' in text
    assert 'Software dependencies' in text and 'numpy' in text and 'rawpy' in text
    assert 'https://github.com/PeterWar/Agentic-Eclipse-Library' in markup
    assert 'https://github.com/fbitto/Eclipse-Image-Processor' in markup
    dialog.resize(320, 240)
    app.processEvents()
    assert content.verticalScrollBar().maximum() > 0
    dialog.close()
    window.close()


def test_tooltip_rendering_wraps_controls_and_dropdown_items(app):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QHelpEvent, QTextDocument
    from coronahdr.desktop_style import apply_theme
    from coronahdr.desktop_folders import FolderDialog
    apply_theme(app)
    dialog = FolderDialog()
    dialog.show()
    app.processEvents()
    handler = app._coronahdr_tooltip_paragraphs

    def check(widget, position, expected):
        event = QHelpEvent(QEvent.Type.ToolTip, position, widget.mapToGlobal(position))
        QApplication.sendEvent(widget, event)
        app.processEvents()
        markup = handler.tip.text()
        assert markup.startswith('<qt><table')
        doc = QTextDocument()
        doc.setHtml(markup)
        assert doc.toPlainText().strip().replace('\u200b', '') == expected
        assert handler.tip.isVisible() and handler.tip.width() <= 450 and handler.tip.height() > 40

    check(dialog.detect_hot_pixels, dialog.detect_hot_pixels.rect().center(), dialog.detect_hot_pixels.toolTip())
    dialog.destination.setText('/Volumes/'+'a'*180)
    check(dialog.destination, dialog.destination.rect().center(), dialog.destination.toolTip())
    dialog.hdr_mode.showPopup()
    app.processEvents()
    view = dialog.hdr_mode.view()
    index = view.model().index(2, 0)
    check(view.viewport(), view.visualRect(index).center(), index.data(Qt.ItemDataRole.ToolTipRole))
    dialog.hdr_mode.hidePopup()
    handler.tip.hide()
    dialog.close()


def test_tooltip_rendering_uses_the_hovered_table_cell(app, source):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QHelpEvent, QTextDocument
    from coronahdr.desktop_style import apply_theme
    apply_theme(app)
    root, review = source
    window = CoronaWindow(root, review)
    window.show()
    app.processEvents()
    handler = app._coronahdr_tooltip_paragraphs
    item = window.table.item(2, 0)
    viewport = window.table.viewport()
    position = window.table.visualItemRect(item).center()
    QApplication.sendEvent(viewport, QHelpEvent(QEvent.Type.ToolTip, position, viewport.mapToGlobal(position)))
    app.processEvents()
    document = QTextDocument()
    document.setHtml(handler.tip.text())
    assert document.toPlainText().strip().replace('\u200b', '') == item.toolTip()
    assert handler.owner_rect == window.table.visualItemRect(item)
    header = window.table.horizontalHeader()
    header_viewport = header.viewport()
    position = header_viewport.rect().center()
    position.setX(header.sectionViewportPosition(0)+header.sectionSize(0)//2)
    QApplication.sendEvent(header_viewport, QHelpEvent(QEvent.Type.ToolTip, position, header_viewport.mapToGlobal(position)))
    app.processEvents()
    document.setHtml(handler.tip.text())
    assert document.toPlainText().strip().replace('\u200b', '') == window.table.horizontalHeaderItem(0).toolTip()
    assert handler.owner_rect.contains(position)
    handler.tip.hide()
    window.close()


def test_tooltip_on_view_scrollbar_uses_safe_coordinate_mapping(app, source):
    from PySide6.QtCore import QEvent, qInstallMessageHandler
    from PySide6.QtGui import QHelpEvent
    from coronahdr.desktop_style import apply_theme
    apply_theme(app)
    root, review = source
    window = CoronaWindow(root, review)
    window.show()
    app.processEvents()
    bar = window.table.verticalScrollBar()
    bar.setToolTip('Scroll this frame table vertically to see additional registration measurements and status details.')
    messages = []
    previous = qInstallMessageHandler(lambda kind, context, message: messages.append(message))
    try:
        position = bar.rect().center()
        QApplication.sendEvent(bar, QHelpEvent(QEvent.Type.ToolTip, position, bar.mapToGlobal(position)))
        app.processEvents()
    finally:
        qInstallMessageHandler(previous)
        window.close()
    assert not any('mapFrom' in message and 'parent' in message for message in messages)


def test_preview_ready_message_routes_images_and_skip_reasons(app, tmp_path, monkeypatch):
    from types import SimpleNamespace
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    events = [dict(role='corona', path='/tmp/main.fits'), dict(role='prominence', path=None, reason='No usable frames')]
    text = ''.join('CORONAHDR_PREVIEW_READY '+json.dumps(event)+'\n' for event in events)
    window.process = SimpleNamespace(readAllStandardOutput=lambda: text.encode())
    paths = []
    monkeypatch.setattr(window.hdr_preview, 'set_path', lambda role, path: paths.append((role, path)))
    window.read_process()
    assert paths == [('corona', '/tmp/main.fits')]
    window.process = None
    window.close()


def test_frame_tooltips_explain_reference_and_unavailable_geometry(app, source):
    root, review = source
    window = CoronaWindow(root, review)
    assert 'required' in window.table.item(0, 0).toolTip().lower()
    assert 'excluded' in window.table.item(2, 0).toolTip().lower()
    assert window.table.item(2, 6).toolTip() == window.table.item(2, 0).toolTip()
    window.close()


@pytest.mark.parametrize('accepted', [True, False])
def test_worker_decision_is_presented_and_answered(app, monkeypatch, tmp_path, accepted):
    from types import SimpleNamespace
    import coronahdr.desktop as module
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    replies = []
    request = dict(message='Choose the shared field', choices=['Crop', 'Keep'])
    window.process = SimpleNamespace(
        readAllStandardOutput=lambda: ('CORONAHDR_DECISION '+json.dumps(request)+'\n').encode(),
        write=replies.append)
    def choose(parent, title, message, choices):
        assert message == request['message'] and choices == request['choices']
        return 'Crop', accepted
    monkeypatch.setattr(module, 'ask_choice', choose)
    window.read_process()
    assert replies == [b'Crop\n' if accepted else b'\n']
    window.process = None
    window.close()


def test_worker_frame_review_sends_selected_candidates(app, monkeypatch, tmp_path):
    from types import SimpleNamespace
    import coronahdr.desktop as module
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    rows = [dict(index=0, filename='one.nef', status='accepted', reason='Passed'),
            dict(index=1, filename='two.nef', status='candidate', reason='Low correlation')]
    replies = []
    window.process = SimpleNamespace(readAllStandardOutput=lambda: ('CORONAHDR_FRAME_REVIEW '+json.dumps(rows)+'\n').encode(),
                                     write=replies.append)
    monkeypatch.setattr(module, 'review_alignment_frames', lambda parent, received: [1] if received == rows else None)
    window.read_process()
    assert replies == [b'[1]\n']
    window.process = None
    window.close()


def test_frame_review_table_only_allows_candidate_selection(app, monkeypatch):
    import coronahdr.desktop as module
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog, QTableWidget
    rows = [dict(index=0, filename='reference.nef', status='accepted', reason='Passed', reference=True),
            dict(index=1, filename='candidate.nef', status='candidate', reason='Low correlation', exposure_s=.125,
                 measurements=dict(heldout_correlation=.54, angle_deg=-.2, independent_check=dict(disagreement_native_px=1.23))),
            dict(index=2, filename='failed.nef', status='rejected', reason='Insufficient support')]
    def accept(dialog):
        table = dialog.findChild(QTableWidget)
        assert table.rowCount() == 3 and table.item(2, 7).text() == 'Insufficient support'
        assert [table.item(1, i).text() for i in (3, 4, 5, 6)] == ['1/8 s', '0.540', '-0.200', '1.230']
        assert table.item(0, 4).text() == 'Reference' and table.item(2, 4).text() == '—'
        assert not table.item(2, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable
        table.item(1, 0).setCheckState(Qt.CheckState.Checked)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, 'exec', accept)
    assert module.review_alignment_frames(None, rows) == [1]


def test_frame_review_dialog_keyboard_selection_and_cancel(app):
    import coronahdr.desktop as module
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QTableWidget
    rows = [dict(index=0, filename='reference.nef', status='accepted', reason='Passed'),
            dict(index=1, filename='candidate.nef', status='candidate', reason='Independent check failed'),
            dict(index=2, filename='failed.nef', status='rejected', reason='Insufficient support')]
    def choose():
        dialog = QApplication.activeModalWidget()
        table = dialog.findChild(QTableWidget)
        assert table.item(1, 0).checkState() == Qt.CheckState.Unchecked
        table.setCurrentCell(1, 0)
        table.setFocus()
        QTest.keyClick(table, Qt.Key.Key_Space)
        dialog.accept()
    QTimer.singleShot(0, choose)
    assert module.review_alignment_frames(None, rows) == [1]
    QTimer.singleShot(0, lambda: QApplication.activeModalWidget().reject())
    assert module.review_alignment_frames(None, rows) is None


def test_conservative_frame_table_allows_only_qualified_selection(app, monkeypatch):
    import coronahdr.desktop as module
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog, QTableWidget
    rows = [dict(index=0, filename='reference.nef', status='accepted', reference=True, reason='Passed'),
            dict(index=1, filename='accepted.nef', status='accepted', reason='Passed',
                 measurements=dict(correlation=.95, median_error=.12, regional_error=.19)),
            dict(index=2, filename='rejected.nef', status='rejected', reason='Radiometric disagreement')]
    def select(dialog):
        table = dialog.findChild(QTableWidget)
        assert table.item(0, 0).checkState() == Qt.CheckState.Checked
        assert not table.item(0, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable
        assert not table.item(2, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable
        assert table.item(2, 7).text() == 'Radiometric disagreement'
        assert table.item(1, 5).text() == '12.0%'
        table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, 'exec', select)
    assert module.review_alignment_frames(None, rows, conservative=True) == [0]


def test_resume_button_finds_saved_work_and_launches(app, monkeypatch, tmp_path):
    from test_resume import folder_job
    import coronahdr.desktop as module
    path, job = folder_job(tmp_path, monkeypatch)
    window = CoronaWindow()
    window.temporary.setText(str(tmp_path))
    monkeypatch.setattr(module, 'ask_choice', lambda parent, title, message, choices: (choices[0], True))
    launched = []
    monkeypatch.setattr(window, 'launch_job', lambda: launched.append(window.job_path))
    window.resume_button.click()
    assert len(launched) == 1 and window.job['resumed_from'] == str(path)
    assert window.job_path != path
    window.close()


def test_folder_dialog_offers_matching_unfinished_work(app, monkeypatch, tmp_path):
    from test_resume import folder_job
    import coronahdr.desktop_folders as module
    path, job = folder_job(tmp_path, monkeypatch)
    job['lunar_margin_px'] = 0
    path.write_text(json.dumps(job))
    dialog = module.FolderDialog()
    dialog.folders['light'].setText(job['folders']['light'])
    dialog.temporary.setText(str(tmp_path))
    monkeypatch.setattr(module, 'ask_choice', lambda parent, title, message, choices: (choices[0], True))
    def unexpected(*args, **kwargs):
        raise AssertionError('Matching work must be offered before new preparation')
    monkeypatch.setattr(module, 'create_folder_job', unexpected)
    dialog.submit()
    assert dialog.job['resumed_from'] == str(path)
    assert dialog.job_path != path
    dialog.close()


def test_open_standalone_hdr_image_without_settings(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    window = CoronaWindow()
    source = tmp_path/'final_aesthetic.fits'
    source.write_bytes(b'image')
    paths = []
    monkeypatch.setattr(window, 'show_preview', lambda path: paths.append(path))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (str(source), ''))
    window.open_image_button.click()
    assert paths == [source]
    assert window.hdr_preview.export_directory == tmp_path
    assert window.hdr_preview.export_base == 'final_aesthetic'
    assert window.job_path is None
    assert 'last job' in window.status.text()
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args, **kwargs: ('', ''))
    window.open_image_button.click()
    assert paths == [source]
    window.close()


def test_main_actions_fit_at_minimum_window_size(app):
    window = CoronaWindow()
    window.resize(960, 700)
    window.controls_scroll.setMaximumHeight(165)
    window.show()
    app.processEvents()
    viewport = window.controls_scroll.viewport()
    assert window.controls_scroll.horizontalScrollBar().maximum() == 0
    buttons = (window.import_button, window.open_image_button, window.more_button)
    assert max(button.height() for button in buttons) == min(button.height() for button in buttons)
    assert window.more_button.font().pointSizeF() == window.open_image_button.font().pointSizeF()
    assert max(button.width() for button in buttons) - min(button.width() for button in buttons) <= 1
    for button in buttons:
        top_left = button.mapTo(viewport, button.rect().topLeft())
        bottom_right = button.mapTo(viewport, button.rect().bottomRight())
        assert viewport.rect().contains(top_left) and viewport.rect().contains(bottom_right)
    assert window.hdr_preview.width() >= 400
    splitter = window.centralWidget()
    assert splitter.widget(0).isAncestorOf(window.hdr_preview)
    assert splitter.widget(1).isAncestorOf(window.import_button)
    window.close()


def test_progress_names_stage_and_reports_current_activity(app, tmp_path):
    from types import SimpleNamespace
    window = CoronaWindow()
    window.job_path = tmp_path/'job.json'
    window.stage_names = [('prepare', 'Prepare and align images'), ('scientific', 'Build scientific HDR'),
                          ('aesthetic', 'Build aesthetic HDR'), ('publish', 'Save outputs')]
    def feed(text):
        window.process = SimpleNamespace(readAllStandardOutput=lambda: text.encode())
        window.read_process()
    feed('CORONAHDR_STAGE prepare\nPrepared full field 12/26: light.nef\n')
    assert 'Step 1 of 4' in window.status.text()
    assert window.progress.value() == 12 and window.progress.maximum() == 26
    assert window.progress.format() == '%v / %m frames'
    feed('CORONAHDR_STAGE scientific\nHDR rows 100:200 of 500\n')
    assert 'Step 2 of 4' in window.status.text() and 'scientific' in window.status.text()
    assert window.progress.value() == 200 and window.progress.maximum() == 500
    assert window.progress.format() == '%v / %m rows'
    feed('CORONAHDR_STAGE aesthetic\n')
    assert 'Step 3 of 4' in window.status.text()
    assert window.progress.minimum() == window.progress.maximum() == 0
    feed('CORONAHDR_STAGE publish\n')
    assert 'Step 4 of 4' in window.status.text()
    window.process = None
    window.close()


def test_more_menu_opens_about_window(app):
    from PySide6.QtWidgets import QDialog
    window = CoronaWindow()
    window.show()
    window.more_button.menu().actions()[-1].trigger()
    app.processEvents()
    dialog = next(child for child in window.findChildren(QDialog) if child.windowTitle() == 'About CoronaHDR')
    assert dialog.isVisible()
    dialog.close()
    window.close()


def test_preview_adjustments_are_below_right_pane_actions(app):
    from PySide6.QtCore import QPoint
    window = CoronaWindow()
    window.resize(960, 700)
    window.show()
    app.processEvents()
    splitter = window.centralWidget()
    preview = window.hdr_preview
    assert splitter.widget(0).isAncestorOf(preview.views['corona'])
    assert splitter.widget(1).isAncestorOf(preview.adjustment_panel)
    assert not splitter.widget(0).isAncestorOf(preview.adjustment_panel)
    assert window.controls_scroll.mapToGlobal(QPoint(0, window.controls_scroll.height())).y() <= preview.adjustment_panel.mapToGlobal(QPoint(0, 0)).y()
    window.close()


def test_log_and_warnings_are_below_preview(app):
    from PySide6.QtCore import QPoint
    window = CoronaWindow()
    window.resize(1280, 800)
    window.show()
    window.log_header.show()
    window.add_warning('Example warning')
    window.log_toggle.click()
    app.processEvents()
    splitter = window.centralWidget()
    preview_pane = splitter.widget(0)
    for widget in (window.warning_panel, window.log_header, window.log):
        assert preview_pane.isAncestorOf(widget)
    bottom = window.hdr_preview.mapToGlobal(QPoint(0, window.hdr_preview.height())).y()
    assert bottom <= window.warning_panel.mapToGlobal(QPoint(0, 0)).y()
    assert bottom <= window.log_header.mapToGlobal(QPoint(0, 0)).y()
    assert window.log.isVisible()
    window.close()


def test_sidebar_controls_remain_grouped_near_top_in_tall_window(app):
    from PySide6.QtCore import QPoint
    window = CoronaWindow()
    window.resize(1280, 1000)
    window.show()
    app.processEvents()
    actions_bottom = window.controls_scroll.mapToGlobal(QPoint(0, window.controls_scroll.height())).y()
    panel_top = window.hdr_preview.adjustment_panel.mapToGlobal(QPoint(0, 0)).y()
    assert 0 <= panel_top-actions_bottom <= 12
    assert window.centralWidget().widget(1).width() <= 460
    window.close()


def test_preview_sliders_share_aligned_columns(app):
    from PySide6.QtCore import QPoint, Qt
    window = CoronaWindow()
    window.resize(1280, 800)
    window.show()
    preview = window.hdr_preview
    for section in ('stretch', 'color', 'filters'):
        preview.preview_sections[section].click()
    app.processEvents()
    sliders = [preview.sliders['corona'], *preview.level_controls.values(), preview.filter_strength]
    assert len({slider.mapToGlobal(QPoint(0, 0)).x() for slider in sliders}) == 1
    assert len({slider.width() for slider in sliders}) == 1
    assert all(label.alignment() & Qt.AlignmentFlag.AlignLeft for label in preview.adjustment_labels.values())
    window.close()


def test_tooltip_reuses_popup_and_dismisses_without_native_tooltip(app, monkeypatch):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtWidgets import QToolTip
    from coronahdr.desktop_style import apply_theme
    apply_theme(app)
    window = CoronaWindow()
    window.show()
    app.processEvents()
    handler = app._coronahdr_tooltip_paragraphs
    original_popup = handler.tip
    def forbidden(*args):
        raise AssertionError('Native tooltip display must not be used')
    monkeypatch.setattr(QToolTip, 'showText', forbidden)
    target = window.import_button
    position = target.rect().center()
    for _ in range(30):
        QApplication.sendEvent(target, QHelpEvent(QEvent.Type.ToolTip, position, target.mapToGlobal(position)))
        app.processEvents()
        assert handler.tip is original_popup and handler.tip.isVisible()
    QApplication.sendEvent(target, QEvent(QEvent.Type.Leave))
    assert not handler.tip.isVisible()
    window.close()
