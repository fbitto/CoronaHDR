"""External working data must not contaminate originals or lose migration records."""
import importlib.util
import json
import pytest
from pathlib import Path


def script(name):
    if not (Path(__file__).parents[1]/'scripts'/f'{name}.py').is_file():
        pytest.skip('Private research script is not included in the public checkout')
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_raw_inventory_ignores_nested_working_copies(tmp_path):
    original=[]
    for role in ('Bias','Flats','Lights'):
        folder=tmp_path/role;folder.mkdir();p=folder/'original.NEF';p.write_bytes(b'original placeholder');original.append(p)
    copy=tmp_path/'CoronaHDR_Work/artifacts/Lights';copy.mkdir(parents=True);(copy/'generated.NEF').write_bytes(b'working copy placeholder')
    assert script('validate_real_raw').source_raw_paths(tmp_path)==sorted(original)


def test_repeated_migration_preserves_original_checksum_manifest(tmp_path):
    root=tmp_path/'project';root.mkdir();work=tmp_path/'external';work.mkdir();(work/'artifacts').mkdir()
    (root/'artifacts').symlink_to(work/'artifacts',target_is_directory=True)
    (work/'storage-location.json').write_text('{}')
    manifest=work/'storage-migration.json';content=json.dumps(dict(state='completed',records=[dict(sha256='retained original digest')]))
    manifest.write_text(content)
    script('move_working_storage').relocate(root,work)
    assert manifest.read_text()==content and (root/'artifacts').is_symlink()
