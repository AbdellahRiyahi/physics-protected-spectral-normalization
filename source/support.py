"""Run provenance, archived evidence and diagnostics for AICSET paper 240."""
import os, sys, json, math, hashlib, platform, socket, zipfile, traceback
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata as import_metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plain_json(value):
    if isinstance(value, dict): return {str(k): plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain_json(v) for v in value]
    if isinstance(value, np.ndarray): return plain_json(value.tolist())
    if isinstance(value, np.generic): return plain_json(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value): return None
    if value is pd.NA: return None
    return value


def json_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain_json(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def configuration_sha256(config):
    return hashlib.sha256(json.dumps(plain_json(config), sort_keys=True).encode()).hexdigest()


def initialize_run(config):
    """Read datasets under /data; allow explicitly configured /home results."""
    requested_base = Path(config['work_root']).expanduser()
    if not requested_base.is_absolute():
        raise ValueError('work_root must be an explicit absolute results path')
    base = requested_base.resolve()
    for name, root in config['data_roots'].items():
        p = Path(root).expanduser().resolve()
        if not str(p).startswith('/data/'):
            raise ValueError(f'{name}: dataset root must resolve under /data: {p}')
        if p == base or p in base.parents:
            raise ValueError('Outputs must be outside the read-only dataset tree')
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run = base / 'runs' / stamp; run.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ['numpy','scipy','pandas','scikit-learn','matplotlib','h5py']:
        try: versions[name] = import_metadata.version(name)
        except import_metadata.PackageNotFoundError: versions[name] = 'NOT_INSTALLED'
    runtime = {'run_id': stamp, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'implementation': 'NEW_AICSET240_REVISION_V5_FULL', 'python': sys.version,
               'platform': platform.platform(), 'host': socket.gethostname(), 'packages': versions,
               'configuration_sha256': configuration_sha256(config), 'configuration': config,
               'historical_pipeline_reproduced': False,
               'analysis_scope': 'RETROSPECTIVE_REANALYSIS_NOT_PROSPECTIVE_CONFIRMATION',
               'raw_inputs_written': False, 'new_scientific_results_available': False}
    json_write(run / 'run_manifest.json', runtime)
    json_write(run / 'configuration_frozen_before_computation.json', config)
    return run, runtime


def package_outputs(run, name='results_to_share.zip', diagnostics_only=False):
    """Only this run's derived outputs; never include dataset trees or raw arrays."""
    run = Path(run); target = run.parent / f'{run.name}_{name}'
    allowed = {'.csv','.json','.md','.txt','.png','.pdf','.svg','.html','.npz','.py'}
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(run.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in allowed: continue
            relative = p.relative_to(run)
            if 'cache' in relative.parts: continue
            if diagnostics_only and relative.parts[0] not in {'diagnostics','run_manifest.json','configuration_frozen_before_computation.json','reviewer_request_status.csv'}: continue
            archive.write(p, relative)
    return target


def legacy_reporting(snapshot, run, bootstrap_reps=10000, seed=20261113):
    """Recalculate reporting summaries from embedded public reports, not signals."""
    out = Path(run) / 'archived_reporting'; out.mkdir(parents=True, exist_ok=True)
    records = snapshot['records']
    o = records['openlab_2024_grouped_calibration.json']
    e = records['z24_ems_environmental_twin.json']
    a = records['openlab_2024_nuisance_ablation.json']
    rows = []
    raw = pd.Series(o['raw_variant']['heldout']['group_max_scores'], dtype=float)
    proposed = pd.Series(o['physics_protected_nuisance_variant']['heldout']['group_max_scores'], dtype=float)
    comparison = paired_group_comparison(raw, proposed, o['raw_variant']['conformal_threshold'],
        o['physics_protected_nuisance_variant']['conformal_threshold'], bootstrap_reps, seed)
    rows.append({'cohort':'OpenLab-2024','comparison':'raw_vs_protected_nnls', **comparison})
    scores = []
    for method, variant in [('raw',o['raw_variant']),('protected_nnls',o['physics_protected_nuisance_variant'])]:
        for group, score in variant['heldout']['group_max_scores'].items():
            scores.append({'dataset':'openlab_2024','group_id':group,'method':method,'score':score,'threshold':variant['conformal_threshold']})
    raw_e = e['variants']['raw_reference']
    x = pd.Series(raw_e['heldout']['group_max_scores'], dtype=float) / raw_e['conformal_threshold']
    for method in ['piecewise_temperature','conditional_environmental_twin']:
        variant = e['variants'][method]
        y = pd.Series(variant['heldout']['group_max_scores'], dtype=float) / variant['conformal_threshold']
        rows.append({'cohort':'Z24 EMS','comparison':'normalized_raw_vs_' + method,
                     **paired_group_comparison(x, y, 1., 1., bootstrap_reps, seed)})
    pd.DataFrame(rows).assign(evidence_origin='ARCHIVED_REPORTING_ONLY').to_csv(out/'paired_comparisons.csv', index=False)
    pd.DataFrame(scores).assign(evidence_origin='ARCHIVED_REPORTING_ONLY').to_csv(out/'group_scores.csv', index=False)
    json_write(out/'source_provenance.json', {'repository':snapshot['repository'], 'revision':snapshot['revision'],
        'source': 'Public JSON reports embedded in this notebook', 'evidence_origin':'ARCHIVED_REPORTING_ONLY',
        'raw_signal_extraction_reproduced':False,'historical_models_refitted':False,
        'snapshot_sha256':configuration_sha256(snapshot)})
    labels = ['Raw','Signed PCA','Protected signed','Protected NNLS']
    order = ['raw_distance','unprotected_rank3_signed_pca','physics_protected_rank3_signed_projection','physics_protected_rank3_nonnegative_cone']
    fig, axes = plt.subplots(1,2,figsize=(7.2,3.1),sharey=True)
    for ax,key,title,gate in zip(axes,['heldout_group_coverage','minimum_controlled_shift_retention'],['Reference-state coverage','Controlled-shift retention'],[.90,.95]):
        values = [a['variants'][v][key] for v in order]
        ax.scatter(values,np.arange(4),color=['#64748b','#b45309','#2563eb','#0f766e'],s=45)
        ax.axvline(gate,color='#b91c1c',linestyle='--',linewidth=1)
        ax.set(title=title,xlabel='Fraction',yticks=np.arange(4),yticklabels=labels,xlim=(.885,1.015))
        ax.grid(axis='x',alpha=.25)
    axes[0].invert_yaxis(); fig.suptitle('Archived evidence: no upstream recomputation', fontsize=10)
    fig.tight_layout(); fig.savefig(out/'archived_ablation.pdf',bbox_inches='tight');fig.savefig(out/'archived_ablation.png',dpi=300,bbox_inches='tight');plt.close(fig)
    return pd.DataFrame(rows)


def attach_covariates(manifest, covariate_path, tolerance_seconds=300):
    """Join external cov_* metadata by acquisition_id or nearest UTC timestamp.

    Timestamp joins are within dataset, reject duplicate covariate timestamps,
    and export the offset. Imputation is deferred to fitting-only model code.
    """
    frame = manifest.copy(); path = Path(covariate_path)
    if not path.exists(): return frame
    table = pd.read_csv(path, dtype={'dataset':str,'acquisition_id':str})
    cols = [c for c in table if c.startswith('cov_')]
    if not cols or 'dataset' not in table: raise ValueError('Covariate table needs dataset and cov_* columns')
    table['dataset']=table.dataset.fillna('').str.strip()
    if not table.dataset.isin(ALLOWED_DATASETS).all():raise ValueError('Covariate datasets must match explicit manifest cohort names')
    if set(cols).intersection(frame.columns): raise ValueError('Specify covariates in manifest OR covariate table, not both')
    if 'acquisition_id' in table:
        table['acquisition_id']=table.acquisition_id.fillna('').str.strip()
        if table.acquisition_id.fillna('').str.strip().eq('').any():
            raise ValueError('Covariate acquisition_id values must be explicit and nonmissing')
        if table.duplicated(['dataset','acquisition_id']).any(): raise ValueError('Duplicate covariate acquisition keys')
        return frame.merge(table[['dataset','acquisition_id']+cols],on=['dataset','acquisition_id'],how='left',validate='many_to_one')
    if 'timestamp_utc' not in table: raise ValueError('Covariate table requires acquisition_id or timestamp_utc')
    pieces = []
    for ds, part in frame.groupby('dataset',sort=False):
        left = part.copy();left['_join_time']=pd.to_datetime([_rio_timestamp(v) for v in left.timestamp_utc],utc=True)
        right = table[table.dataset==ds].copy()
        if right.empty:
            for col in cols:left[col]=np.nan
            pieces.append(left.drop(columns='_join_time'));continue
        right['_cov_time']=pd.to_datetime([_rio_timestamp(v) for v in right.timestamp_utc],utc=True)
        if right['_cov_time'].duplicated().any(): raise ValueError(f'Duplicate covariate timestamps for {ds}')
        merged=pd.merge_asof(left.sort_values('_join_time'),right[['_cov_time']+cols].sort_values('_cov_time'),
            left_on='_join_time',right_on='_cov_time',direction='nearest',tolerance=pd.Timedelta(seconds=tolerance_seconds))
        merged['covariate_time_delta_seconds']=(merged['_join_time']-merged['_cov_time']).dt.total_seconds().abs()
        pieces.append(merged.drop(columns=['_join_time','_cov_time']))
    return pd.concat(pieces,ignore_index=True)


def canonical_metadata(frame):
    """Use physical dates/weeks/scenarios, and official Z24 PDT state identities."""
    d=frame.copy();d['source_group_id']=d.group_id.astype(str)
    for dataset,part in d.groupby('dataset'):
        if dataset in ['openlab_2024','z24_ems']:
            stamp=pd.to_datetime(part.timestamp_utc,utc=True,errors='raise')
            d.loc[part.index,'group_id']=stamp.dt.strftime('%Y-%m-%d' if dataset=='openlab_2024' else '%G-W%V')
        elif dataset=='z24_pdt':
            for i,row in part.iterrows():
                scenario=f'{int(row.scenario):02d}'
                expected='reference' if scenario in {'01','02','08'} else 'altered'
                if row.state_label not in {expected,'unknown'}: raise ValueError(f'PDT state label contradicts official scenario {scenario}')
                d.at[i,'scenario']=scenario;d.at[i,'state_label']=expected;d.at[i,'group_id']='scenario_'+scenario
    for (ds,group),part in d.groupby(['dataset','group_id']):
        if part.state_label.nunique()>1:raise ValueError(f'Mixed state labels in one physical group: {ds}/{group}')
    return d


def assign_partitions(meta, config):
    d=meta.copy();d['partition']='excluded_unknown_state'
    for ds,part in d.groupby('dataset'):
        if ds=='z24_pdt':
            d.loc[part.index[part.scenario=='01'],'partition']='fit'
            d.loc[part.index[part.scenario=='02'],'partition']='calibration'
            d.loc[part.index[(part.scenario=='08')|(part.state_label=='altered')],'partition']='test'
        else:
            ref=part[part.state_label=='reference'].copy()
            ref['_time']=pd.to_datetime(ref.timestamp_utc,utc=True)
            order=ref.groupby('group_id')['_time'].min().sort_values().index.tolist()
            spec=config['splits'][ds];nfit=spec['fit'];ncal=spec['calibration']
            if min(nfit,ncal,spec['minimum_test'])<1:
                raise ValueError('Every partition must contain at least one physical group')
            if len(order)<nfit+ncal+spec['minimum_test']:
                raise ValueError(f'{ds}: need {nfit+ncal+spec["minimum_test"]} reference groups, found {len(order)}')
            for name,groups in [('fit',order[:nfit]),('calibration',order[nfit:nfit+ncal]),('test',order[nfit+ncal:])]:
                d.loc[part.index[part.group_id.isin(groups)],'partition']=name
            altered=part[part.state_label=='altered']
            if len(altered):
                cutoff=ref[ref.group_id.isin(order[:nfit+ncal])]['_time'].max()
                if (pd.to_datetime(altered.timestamp_utc,utc=True)<=cutoff).any():
                    raise ValueError(f'{ds}: altered test groups precede the frozen calibration cutoff')
                d.loc[altered.index,'partition']='test'
    assigned=d[d.partition.isin(['fit','calibration','test'])]
    if assigned.groupby(['dataset','group_id']).partition.nunique().gt(1).any():raise AssertionError('Group leakage across partitions')
    if assigned.groupby(['dataset','acquisition_id']).partition.nunique().gt(1).any():raise AssertionError('Acquisition leakage')
    for ds,part in assigned.groupby('dataset'):
        counts=part.groupby('partition').size()
        if any(int(counts.get(role,0))==0 for role in ['fit','calibration','test']):
            raise ValueError(f'{ds}: fitting, calibration and evaluation acquisitions are all required')
    return d


def export_documented_openlab_context(manifest, out):
    """Disclose a documented non-destructive intervention without relabelling."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    part=manifest[manifest.dataset.eq('openlab_2024')].copy()
    if part.empty:return {}
    stamps=pd.to_datetime(part.timestamp_utc,utc=True)
    selected=part[(stamps>=pd.Timestamp('2024-07-22',tz='UTC')) &
                  (stamps<pd.Timestamp('2024-08-03',tz='UTC'))].copy()
    columns=[c for c in ['dataset','record_id','acquisition_id','group_id','timestamp_utc','partition'] if c in selected]
    selected[columns].to_csv(out/'documented_non_destructive_prop_period.csv',index=False)
    calibration=selected[selected.partition.eq('calibration')]
    audit={'source':'https://pmc.ncbi.nlm.nih.gov/articles/PMC12145563/','section':'4.6 Test Concept',
        'period':'2024-07-22 to 2024-08-02','intervention':'Temporary midspan prop; changed boundary conditions without damage',
        'calibration_acquisitions':len(calibration),'calibration_dates':sorted(calibration.group_id.unique().tolist()),
        'labels_or_partitions_changed':False,'predictor_added':False,
        'interpretation':'The undamaged reference phase includes a known non-destructive boundary-condition intervention. These observations remain in the fixed chronological protocol.'}
    json_write(out/'documented_openlab_intervention.json',audit)
    return audit
