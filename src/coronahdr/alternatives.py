"""Full-field longest-exposure comparison sharing immutable geometry diagnostics."""
import json,os,shutil,resource
from pathlib import Path
from contextlib import ExitStack
import numpy as np
from astropy.io import fits
from .streaming import load_cached_run,bands,mapped,sample_band,release_pages,export_map
from .empirical import soft_longest_weights
from .io import write_image


def export_longest(previous,output):
    previous,output=Path(previous),Path(output)
    if output.exists():raise FileExistsError(output)
    old=json.loads((previous/'report.json').read_text());shape=tuple(old['output']['shape_yx']);origin=np.array(old['output']['origin_reference_yx'])
    n=len(old['accepted_indices']);estimated=shape[0]*shape[1]*(4*n+36)
    if shutil.disk_usage(output.parent).free < estimated*1.2:
        raise OSError('Insufficient free space for alternate master, weights and bounded scratch')
    frames,_,_=load_cached_run(previous,old['recipe']);selected=[frames[i] for i in old['accepted_indices']]
    output.mkdir();work=output/'work';work.mkdir()
    channels=selected[0].data.shape[2:];master=mapped(work/'master.npy',(*shape,*channels),np.float32)
    weight=mapped(work/'weight.npy',shape,np.float32);count=mapped(work/'contributions.npy',shape,np.uint8 if n<=255 else np.uint32)
    individual=[mapped(work/f'weight_{i:03d}.npy',shape,np.float32) for i in range(n)]
    def link(source,target):target.symlink_to(os.path.relpath(source.resolve(),target.parent.resolve()))
    with ExitStack() as stack:
        raw_weights=[stack.enter_context(fits.open(previous/f'frame_{i:03d}'/'weight.fits',memmap=True))[0].data for i in range(n)]
        for start,end in bands(shape[0],old['memory']['tile_rows']):
            preferences=soft_longest_weights(np.stack([a[start:end] for a in raw_weights]),[f.scale for f in selected])
            total=preferences.sum(axis=0);numerator=np.zeros((end-start,shape[1],*channels))
            for i,(frame,transform) in enumerate(zip(selected,old['transformations'])):
                if 'matrix_yx' in transform:
                    delta=np.array(transform['matrix_yx']);delta[:2,2]-=origin
                else:delta=np.array(transform['shift_dy_dx'])-origin
                # Each alternative samples the prepared original once, never the weighted master.
                values=sample_band(frame.data,delta,start,end,shape[1])/frame.scale
                w=preferences[i];individual[i][start:end]=w
                numerator+=np.nan_to_num(values)*(w[...,None] if channels else w)
                release_pages(frame)
            result=np.full(numerator.shape,np.nan)
            np.divide(numerator,total[...,None] if channels else total,out=result,where=(total>0)[...,None] if channels else total>0)
            master[start:end]=result;weight[start:end]=total;count[start:end]=(preferences>0).sum(axis=0)
            print(f'Longest HDR rows {start}:{end} of {shape[0]}',flush=True)
    metadata=dict(old['output'],method='longest_valid_soft',weighting='source reliability times exp((exposure-longest_valid)/(0.1*longest_valid)); no variance model')
    for extension in ('fits','tiff'):write_image(output/f'master.{extension}',master,metadata)
    export_map(output/'weight.fits',weight,dict(meaning='summed heuristic longest-valid weight'))
    export_map(output/'contributions.fits',count,dict(meaning='contributing valid inputs'),'uint8' if n<=255 else 'int32')
    export_map(output/'support.fits',weight>0,dict(meaning='valid support'),'uint8')
    for key in ('coverage','lunar','saturation','low_signal','defective','unresolved_saturation'):
        link(previous/f'{key}.fits',output/f'{key}.fits')
    for i,a in enumerate(individual):
        folder=output/f'frame_{i:03d}';folder.mkdir();export_map(folder/'weight.fits',a,dict(meaning='longest-valid heuristic weight'))
        a._mmap.close();(work/f'weight_{i:03d}.npy').unlink()
        for key in ('coverage','lunar','saturation','low_signal','defective'):link(previous/f'frame_{i:03d}'/f'{key}.fits',folder/f'{key}.fits')
    registration=dict(old['registration'],hdr_method='longest_valid_soft')
    report=dict(old,output=metadata,registration=registration,hdr_method='longest_valid_soft',
        recipe=dict(old['recipe'],method='longest_valid_soft'),source_product=str(previous),
        memory=dict(old['memory'],work_directory=str(work),process_peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        shared_diagnostics=str(previous),warnings=[*old['warnings'],'Alternative uses relative symlinks to unchanged geometry masks; keep both product directories together',
        'Exposure preference is heuristic, not inverse variance; no sky or gain correction applied'])
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (output/'registration.json').write_text(json.dumps(registration,indent=2,allow_nan=False))
    (output/'recipe.json').write_text(json.dumps(dict(source_product=str(previous),method='longest_valid_soft',command='uv run python scripts/export_longest.py SOURCE --output DESTINATION'),indent=2))
    for a,name in [(master,'master'),(weight,'weight'),(count,'contributions')]:
        a._mmap.close();(work/f'{name}.npy').unlink()
    return report
