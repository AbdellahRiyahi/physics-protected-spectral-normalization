"""British-English publication outputs from frozen, derived AICSET results.

This module does not extract signals, fit models, select thresholds, or change
scientific results. It reads the exported CSV/JSON files and creates a separate
publication/ directory. Main-paper placement is suggested, not automatically
certified against the nine-page manuscript limit.
"""
from pathlib import Path
import html
import json
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib import font_manager


PUBLICATION_METHODS = {
    'raw': 'Uncorrected reference',
    'signed_pca': 'Unprotected signed projection',
    'protected_signed': 'Protected signed projection',
    'protected_nnls': 'Protected NNLS',
    'baseline_pca': 'Conventional PCA',
    'frequency_ratio': 'Spectral peak ratios',
    'temperature_ridge': 'Temperature regression',
    'piecewise_temperature': 'Piecewise temperature regression',
    'conditional_ridge': 'Multivariable regression',
}
PUBLICATION_COHORTS = {
    'openlab_2024': 'OpenLab-2024', 'z24_ems': 'Z24 environmental monitoring',
    'z24_pdt': 'Z24 progressive-damage tests',
}
PUBLICATION_COLOURS = {
    'raw':'#555555', 'protected_nnls':'#007D73', 'signed_pca':'#D55E00',
    'protected_signed':'#0072B2', 'baseline_pca':'#8B61A1',
    'frequency_ratio':'#A16C00', 'temperature_ridge':'#009E73',
    'piecewise_temperature':'#56B4E9', 'conditional_ridge':'#CC79A7',
}


def _pub_read_csv(path):
    path=Path(path)
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={
            'group_id':str, 'scenario':str, 'setup_id':str, 'record_id':str,
        })
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _pub_read_json(path, default=None):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def _pub_clean(value):
    if isinstance(value, dict): return {str(k):_pub_clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_pub_clean(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return _pub_clean(value.item())
    if isinstance(value, np.ndarray): return _pub_clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def _pub_label(method):
    return PUBLICATION_METHODS.get(str(method), str(method).replace('_',' ').capitalize())


def _pub_order(frame, field='method'):
    if frame.empty or field not in frame:return frame
    order={m:i for i,m in enumerate(PUBLICATION_METHODS)}
    return frame.assign(_order=frame[field].map(order).fillna(99)).sort_values('_order',kind='stable').drop(columns='_order')


def _pub_display(value):
    if value is None or (isinstance(value,(float,np.floating)) and not np.isfinite(value)):
        return 'Not estimable'
    if isinstance(value,(bool,np.bool_)): return 'Yes' if value else 'No'
    if isinstance(value,(int,np.integer)): return str(value)
    if isinstance(value,(float,np.floating)):
        if float(value).is_integer() and abs(value)<1e6:return str(int(value))
        return f'{value:.3g}'
    return str(value)


def _pub_markdown_table(frame):
    def cell(value):return _pub_display(value).replace('|',r'\|').replace('\n',' ')
    header='| '+' | '.join(cell(c) for c in frame.columns)+' |'
    rule='| '+' | '.join('---' for _ in frame.columns)+' |'
    rows=['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join([header,rule]+rows)


def _pub_table(frame,title,notes,stem,out,catalogue,role='Supplementary/reviewer response'):
    if frame.empty:return
    number=1+sum(item['kind']=='Table' for item in catalogue)
    caption=f'Table {number}. {title}'
    folder=Path(out)/'tables';folder.mkdir(parents=True,exist_ok=True)
    frame.to_csv(folder/(stem+'.csv'),index=False)
    md=caption+'\n\n'+_pub_markdown_table(frame)+'\n\n'+notes+'\n'
    (folder/(stem+'.md')).write_text(md,encoding='utf-8')
    display=frame.map(_pub_display) if hasattr(frame,'map') else frame.applymap(_pub_display)
    table=display.to_html(index=False,escape=True,border=0,classes='publication-table')
    body=f'<section id="{html.escape(stem)}"><p class="caption">{html.escape(caption)}</p>{table}<p class="note">{html.escape(notes)}</p></section>'
    (folder/(stem+'.html')).write_text(body,encoding='utf-8')
    catalogue.append({'kind':'Table','number':number,'stem':stem,'caption':caption,'notes':notes,
        'suggested_use':role,'files':[str((folder/(stem+ext)).relative_to(out)) for ext in ['.csv','.md','.html']],
        'evidence_origin':'NEW_RAW_SIGNAL_REANALYSIS'})


def _pub_figure(fig,caption,notes,stem,out,catalogue,dpi,role='Main paper candidate'):
    number=1+sum(item['kind']=='Figure' for item in catalogue)
    folder=Path(out)/'figures';folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    for ext in ['.pdf','.png']:
        p=folder/(stem+ext)
        fig.savefig(p,dpi=dpi,facecolor='white',bbox_inches='tight',pad_inches=.035)
        paths.append(str(p.relative_to(out)))
    plt.close(fig)
    catalogue.append({'kind':'Figure','number':number,'stem':stem,'caption':f'Fig. {number}. {caption}',
        'notes':notes,'suggested_use':role,'files':paths,'evidence_origin':'NEW_RAW_SIGNAL_REANALYSIS'})


def _pub_threshold(value):
    if not isinstance(value,dict) or not bool(value.get('finite_supported',False)):
        return None
    threshold=value.get('threshold')
    return float(threshold) if threshold is not None and np.isfinite(float(threshold)) else None


def _pub_partitions(run,out,catalogue):
    rows=[]
    for folder in sorted((Path(run)/'new_analysis').glob('*')):
        frame=_pub_read_csv(folder/'physical_group_partitions.csv')
        if frame.empty:continue
        for role in ['fit','calibration','test','excluded_unknown_state']:
            part=frame[frame.partition.eq(role)]
            if part.empty:continue
            stamp=pd.to_datetime(part.get('timestamp_utc'),utc=True,errors='coerce')
            date_range='Not used for scenario grouping'
            if hasattr(stamp,'notna') and stamp.notna().any():
                valid=stamp.dropna();date_range=f'{valid.min():%d %b %Y} to {valid.max():%d %b %Y}'
            rows.append({'Cohort':PUBLICATION_COHORTS.get(folder.name,folder.name),
                'Partition':{'fit':'Fitting','calibration':'Calibration','test':'Evaluation',
                             'excluded_unknown_state':'Unconfirmed state, excluded'}[role],
                'Physical groups':int(part.group_id.nunique()),'Acquisitions':len(part),'Date range (UTC)':date_range})
    _pub_table(pd.DataFrame(rows),'Physical-group partitions used in the new analysis.',
        'Counts are physical groups and acquisitions, not independent windows. Dates, ISO weeks and structural scenarios define OpenLab, environmental Z24 and progressive-test groups, respectively.',
        'physical_group_partitions',out,catalogue,'Main paper candidate')


def _pub_availability(run,out,catalogue):
    results=_pub_read_json(Path(run)/'cohort_results.json',[]) or []
    lookup={r['dataset']:r for r in results}
    rows=[]
    for dataset,label in PUBLICATION_COHORTS.items():
        result=lookup.get(dataset)
        if not result:
            rows.append({'Cohort':label,'Method':'All new analyses','Availability':'Not calculated',
                         'Reason':'Confirmed inputs or metadata were not available in this run.'})
            continue
        methods=result.get('methods',[])
        errors={entry.get('method'):entry.get('error','') for entry in result.get('unavailable_methods',[])}
        for method in PUBLICATION_METHODS:
            if method in methods:
                rows.append({'Cohort':label,'Method':_pub_label(method),'Availability':'Calculated','Reason':''})
            elif method in errors:
                reason=str(errors[method]).replace("ValueError('",'').rstrip("')")
                reason=reason.replace('NOT_ESTIMABLE_BY_PROTOCOL: ','Not estimable under the frozen protocol: ')
                rows.append({'Cohort':label,'Method':_pub_label(method),'Availability':'Not calculated','Reason':reason})
    _pub_table(pd.DataFrame(rows),'Availability of the analyses and compensation methods.',
        'An unavailable comparator is not treated as a negative experimental result. A single PDT fitting scenario cannot support group-based cross-validation of a temperature regression.',
        'analysis_availability',out,catalogue)


def _pub_documented_prop_note(folder):
    if folder.name!='openlab_2024':return ''
    d=_pub_read_csv(folder/'physical_group_partitions.csv')
    if d.empty or 'timestamp_utc' not in d:return ''
    stamps=pd.to_datetime(d.timestamp_utc,utc=True,errors='coerce')
    selected=d[(stamps>=pd.Timestamp('2024-07-22',tz='UTC')) &
               (stamps<pd.Timestamp('2024-08-03',tz='UTC')) & d.partition.eq('calibration')]
    if selected.empty:return ''
    return (' The frozen calibration includes '+str(len(selected))+' acquisitions on '+str(selected.group_id.nunique())+
        ' dates during the documented non-destructive temporary-prop intervention (22 July–2 August 2024; data article section 4.6). '
        'Undamaged reference phase does not mean unchanged boundary conditions. These records and thresholds are retained.')


def _pub_coverage(folder,out,catalogue,width,dpi):
    frame=_pub_order(_pub_read_csv(folder/'reference_coverage.csv'))
    if frame.empty:return
    rows=[]
    for _,r in frame.iterrows():
        finite=pd.notna(r.coverage)
        covered=f'{int(r.covered_groups)}/{int(r.n_groups)}' if pd.notna(r.covered_groups) else 'Abstain'
        bounds=f'{100*r.wilson_95_low:.1f}–{100*r.wilson_95_high:.1f}' if finite else 'Not estimable'
        rows.append({'Method':_pub_label(r.method),'Covered/reference groups':covered,
            'Coverage (%)':100*r.coverage if finite else np.nan,'Wilson 95% bounds (%)':bounds,
            'Alert groups':r.alert_groups})
    table=pd.DataFrame(rows)
    label=PUBLICATION_COHORTS.get(folder.name,folder.name)
    _pub_table(table,f'Reference-state evaluation for {label}.',
        'Each method uses its own frozen calibration threshold. Calibration-group counts and exact thresholds remain in the source CSV. Wilson bounds are descriptive 95% intervals; independence and exchangeability are not established. Abstain denotes an unsupported finite nominal threshold, not perfect coverage.'+_pub_documented_prop_note(folder),
        folder.name+'_reference_coverage',out,catalogue,'Main paper candidate')
    finite=frame[frame['coverage'].notna()&frame.wilson_95_low.notna()&frame.wilson_95_high.notna()]
    if finite.empty:return
    height=max(2.25,.31*len(finite)+.8)
    fig,ax=plt.subplots(figsize=(width,height))
    for y,(_,row) in enumerate(finite.iterrows()):
        colour=PUBLICATION_COLOURS.get(row.method,'#555555')
        ax.plot([row.wilson_95_low,row.wilson_95_high],[y,y],color=colour,lw=1.4)
        ax.scatter([row.coverage],[y],color=colour,s=22,zorder=3)
        ax.text(1.025,y,f'{int(row.covered_groups)}/{int(row.n_groups)}',va='center',fontsize=7.5)
    ax.set(yticks=np.arange(len(finite)),yticklabels=[_pub_label(m) for m in finite.method],
        xlabel='Observed reference-group coverage',xlim=(0,1.15),ylim=(-.6,len(finite)-.4))
    ax.set_xticks([0,.25,.5,.75,1]);ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.invert_yaxis();ax.grid(axis='x',color='#dddddd',lw=.5);ax.set_axisbelow(True)
    fig.tight_layout()
    _pub_figure(fig,f'Reference-group coverage for {label}; points show observed proportions and bars show Wilson 95% intervals.',
        'Counts are shown to the right. The intervals are descriptive and do not establish prospective validity.',
        folder.name+'_reference_coverage',out,catalogue,dpi)


def _pub_score_figures(folder,out,catalogue,width,dpi):
    frame=_pub_read_csv(folder/'group_scores.csv')
    if frame.empty:return
    methods=['raw','protected_nnls']
    if not set(methods)<=set(frame.method):return
    thresholds=_pub_read_json(folder/'thresholds.json',{}) or {}
    part=frame[frame.method.isin(methods)&frame.partition.eq('test')].copy()
    pivot=part.pivot(index='group_id',columns='method',values='score').dropna().sort_index()
    if pivot.empty:return
    fig,ax=plt.subplots(figsize=(width,2.8))
    xx=np.arange(len(pivot))
    is_pdt=folder.name=='z24_pdt'
    descriptive=_pub_read_csv(folder/'pdt_scenario_descriptive_decisions.csv') if is_pdt else pd.DataFrame()
    if is_pdt and 'scenario_08' in pivot.index:
        ref_index=pivot.index.tolist().index('scenario_08')
        ax.axvspan(ref_index-.4,ref_index+.4,color='#dbeafe',alpha=.65,zorder=0,label='Reference 08')
    for method,marker in [('raw','o'),('protected_nnls','s')]:
        colour=PUBLICATION_COLOURS[method]
        ax.plot(xx,pivot[method],marker=marker,markersize=3,lw=1.05,color=colour,label=_pub_label(method))
        q=_pub_threshold(thresholds.get(method))
        if q is None and is_pdt and not descriptive.empty:
            qs=descriptive.loc[descriptive.method.eq(method),'threshold'].dropna().unique()
            if len(qs)==1:q=float(qs[0])
        if q is not None:ax.axhline(q,color=colour,lw=.8,ls='--',label=('Uncorrected cutoff (desc.)' if method=='raw' else 'Protected cutoff (desc.)') if is_pdt else ('Uncorrected threshold' if method=='raw' else 'Protected threshold'))
    labels=pivot.index.tolist()
    if folder.name in {'openlab_2024','z24_ems'}:
        labels=[str(label).replace('2024-','') for label in labels]
    else:labels=[str(label).replace('scenario_','') for label in labels]
    every=1 if is_pdt else max(1,int(np.ceil(len(labels)/10)))
    ticks=xx[::every];ax.set_xticks(ticks,labels[::every],rotation=30,ha='right')
    ax.set(xlabel={'openlab_2024':'Evaluation date (month–day)','z24_ems':'Evaluation ISO week',
                  'z24_pdt':'Evaluation structural scenario'}.get(folder.name,'Evaluation physical group'),
           ylabel='Group-maximum\nspectral distance',xlim=(-.35,len(pivot)-.65))
    ax.legend(fontsize=7,ncol=2,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.015),borderaxespad=0)
    ax.grid(axis='y',color='#dddddd',lw=.5)
    fig.tight_layout()
    label=PUBLICATION_COHORTS.get(folder.name,folder.name)
    if is_pdt:
        notes='Dashed lines are descriptive maximum-score thresholds from reference scenario 02, without a nominal 95% guarantee. The nominal rule abstains. Blue shading identifies the held-out reference 08; an alert on this state is a reference false alert. Scores are scenario maxima, not independent damage replications.'
    else:notes='Only evaluation groups are shown. Dashed lines, when present, are method-specific frozen calibration thresholds. Score contraction is a model property and does not by itself establish better monitoring or retained damage sensitivity.'
    _pub_figure(fig,f'Paired uncorrected and protected-NNLS group scores for {label}.',
        notes,
        folder.name+'_paired_evaluation_scores',out,catalogue,dpi)

    if folder.name=='z24_pdt':return
    timeline=frame[frame.method.isin(methods)].pivot(index='group_id',columns='method',values='score').sort_index()
    metadata=frame.drop_duplicates('group_id').set_index('group_id').reindex(timeline.index)
    fig,ax=plt.subplots(figsize=(width,2.6))
    styles={'fit':('#edf2f7','Fitting'),'calibration':('#fff2cf','Calibration'),'test':('#e1f0ec','Evaluation')}
    for role,(colour,label_role) in styles.items():
        indices=np.flatnonzero(metadata.partition.eq(role).to_numpy())
        if len(indices):ax.axvspan(indices.min()-.5,indices.max()+.5,color=colour,label=label_role,zorder=0)
    xx=np.arange(len(timeline))
    for method in methods:ax.plot(xx,timeline[method],color=PUBLICATION_COLOURS[method],lw=.9,label=_pub_label(method))
    every=max(1,int(np.ceil(len(xx)/6)));ax.set_xticks(xx[::every],[str(v) for v in timeline.index[::every]],rotation=25,ha='right')
    ax.set(xlabel='Chronological physical group',ylabel='Group-maximum\nspectral distance',xlim=(-.5,len(timeline)-.5))
    ax.legend(fontsize=6.5,ncol=3,frameon=False);fig.tight_layout()
    _pub_figure(fig,f'Chronological fitting, calibration and evaluation groups for {label}.',
        'Background shading identifies the frozen partitions. Group sizes and identifiers are exported separately. This is a protocol diagnostic, not a prospective deployment result.'+_pub_documented_prop_note(folder),
        folder.name+'_partition_timeline',out,catalogue,dpi,'Supplementary/reviewer response')


def _pub_paired_table(folder,out,catalogue):
    frame=_pub_order(_pub_read_csv(folder/'paired_group_comparisons.csv'))
    if frame.empty:return
    rows=[];decisions=[]
    for _,r in frame.iterrows():
        same=r.get('comparison_scope')=='SAME_SPECTRAL_FEATURE_FAMILY'
        if same:
            lo=r.get('median_paired_relative_reduction_ci95_low',np.nan)
            hi=r.get('median_paired_relative_reduction_ci95_high',np.nan)
            bounds=f'{100*lo:.2f}–{100*hi:.2f}' if pd.notna(lo) and pd.notna(hi) else 'Not estimated'
            rows.append({'Method':_pub_label(r.method),'Paired groups':r.get('n_groups'),
                'Median score reduction (%)':100*r.get('median_paired_relative_reduction',np.nan),
                'Exploratory 95% bounds (%)':bounds})
        decisions.append({'Method':_pub_label(r.method),'Uncorrected-only alerts':r.get('raw_alert_only_groups'),
            'Corrected-only alerts':r.get('corrected_alert_only_groups'),
            'Exact McNemar p':r.get('mcnemar_exact_two_sided_p'),'Holm-adjusted p':r.get('mcnemar_p_holm')})
    title=f'Paired evaluation relative to the uncorrected reference for {PUBLICATION_COHORTS.get(folder.name,folder.name)}.'
    note='The score-reduction bounds are exploratory paired percentile-bootstrap bounds conditional on the frozen models and thresholds; they exclude fitting and calibration uncertainty. Only the common spectral feature space is included. Score contraction alone does not establish improved monitoring.'
    role='Main paper candidate'
    if folder.name=='z24_pdt':
        title='Score contraction on the sole held-out Z24 reference scenario 08.'
        note='This comparison uses one paired reference group (scenario 08), not all 15 evaluation states. No bootstrap interval is estimated for this single group. Score contraction alone does not establish damage sensitivity or improved monitoring.'
        role='Supplementary/reviewer response'
    _pub_table(pd.DataFrame(rows),title,note,folder.name+'_paired_score_reductions',out,catalogue,role)
    _pub_table(pd.DataFrame(decisions),f'Paired reference-state alert decisions for {PUBLICATION_COHORTS.get(folder.name,folder.name)}.',
        'Each method is compared with the uncorrected reference using its own frozen threshold. Binary decisions may be paired across feature spaces. McNemar inference requires independent paired groups, which are not established here; adjusted values use Holm correction within the cohort.',
        folder.name+'_paired_alert_decisions',out,catalogue,'Supplementary/reviewer response')


def _pub_retention(folder,out,catalogue,width,dpi):
    frame=_pub_read_csv(folder/'synthetic_shift_retention.csv')
    if frame.empty:return
    valid=frame[frame.retention_status.eq('OK')].copy()
    if valid.empty:return
    valid['Magnitude set']=np.where(valid.shift_set.eq('DESIGN_MAGNITUDE_ON_TEST_GROUP_SHAPE'),'Design magnitudes','Additional magnitudes')
    summary=valid.groupby(['method','Magnitude set'],sort=False).retention_ratio.agg(['count','min','median','max']).reset_index()
    summary=_pub_order(summary)
    table=summary.copy();table.method=table.method.map(_pub_label)
    for col in ['min','median','max']:table[col]*=100
    table.columns=['Method','Magnitude set','Perturbation cases','Minimum retention (%)','Median retention (%)','Maximum retention (%)']
    _pub_table(table,f'Controlled spectral-shift retention for {PUBLICATION_COHORTS.get(folder.name,folder.name)}.',
        'A case is a group–shift combination, not an independent damage experiment. Design and additional magnitudes are separated; ratios are relative to the uncorrected spectral response. The peak-ratio comparator has a different feature space and is excluded from these norm ratios.',
        folder.name+'_controlled_shift_retention',out,catalogue,'Main paper candidate')
    methods=[m for m in PUBLICATION_METHODS if m in set(summary.method)]
    fig,ax=plt.subplots(figsize=(width,max(2.3,.42*len(methods)+.6)))
    for i,method in enumerate(methods):
        for name,offset,marker in [('Design magnitudes',-.11,'o'),('Additional magnitudes',.11,'s')]:
            row=summary[summary.method.eq(method)&summary['Magnitude set'].eq(name)]
            if row.empty:continue
            r=row.iloc[0];colour=PUBLICATION_COLOURS.get(method,'#555555')
            ax.plot([r['min'],r['max']],[i+offset,i+offset],color=colour,lw=1.2,alpha=.75)
            ax.scatter([r['median']],[i+offset],color=colour,marker=marker,s=22,
                       facecolors=colour if marker=='o' else 'white',zorder=3,
                       label=name if i==0 else None)
    values=summary[['min','max']].to_numpy(float)
    lo=min(0.9,float(np.nanmin(values))-.025);hi=max(1.025,float(np.nanmax(values))+.025)
    ax.axvline(1,color='#777777',ls=':',lw=.8);ax.set(xlim=(max(0,lo),hi),yticks=np.arange(len(methods)),
        yticklabels=[_pub_label(m) for m in methods],xlabel='Synthetic response-norm retention')
    ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.grid(axis='x',color='#dddddd',lw=.5)
    ax.legend(fontsize=7,frameon=False,loc='best');fig.tight_layout()
    _pub_figure(fig,f'Controlled spectral-shift retention for {PUBLICATION_COHORTS.get(folder.name,folder.name)}; markers show medians and bars show full ranges.',
        'Ranges span held-out group shapes and shift magnitudes; they are not confidence intervals. Synthetic translations do not establish sensitivity to real structural damage.',
        folder.name+'_controlled_shift_retention',out,catalogue,dpi)


def _pub_pdt(folder,out,catalogue,width,dpi):
    side=_pub_read_csv(folder/'pdt_raw_vs_protected_by_scenario.csv')
    if not side.empty:
        fields=['scenario','phase','decision_raw','decision_protected','raw_alert_lost_after_correction','new_alert_after_correction']
        table=side[[c for c in fields if c in side]].copy()
        table['phase']=table.phase.map({'REFERENCE':'Reference','PIER_INTERVENTION':'Pier intervention','PROGRESSIVE_DAMAGE':'Damage phase'})
        table=table.rename(columns={'scenario':'Scenario','phase':'Phase',
            'decision_raw':'Uncorrected','decision_protected':'Protected NNLS',
            'raw_alert_lost_after_correction':'Lost alert','new_alert_after_correction':'New alert'})
        _pub_table(table,'Scenario-level decisions before and after protected normalisation.',
            'Thresholds are the maximum of the reference calibration scenario. These are descriptive decisions without a nominal 95% guarantee. Lost alert means an uncorrected alert becomes a protected non-alert; new alert means the converse. An alert lost on an altered state may indicate lost sensitivity; an alert on a reference state may be false.',
            'z24_pdt_scenario_decisions',out,catalogue,'Main paper candidate')
    phase=_pub_read_csv(folder/'pdt_progressive_damage_phase.csv')
    if not phase.empty:
        phase=phase[phase.threshold_rule.eq('DESCRIPTIVE_CALIBRATION_MAXIMUM')]
        def assessed_count(numerator,denominator):
            if pd.isna(numerator) or pd.isna(denominator) or denominator<=0:
                return 'Not estimable'
            return f'{int(numerator)}/{int(denominator)}'
        rows=[]
        for _,row in _pub_order(phase).iterrows():
            sensitivity=row.get('sensitivity',np.nan)
            rows.append({'Method':_pub_label(row.method),
                'Altered, detected/total':assessed_count(row.get('true_positive_groups',np.nan),row.get('n_altered_groups',np.nan)),
                'Reference, alerts/total':assessed_count(row.get('false_positive_groups',np.nan),row.get('n_reference_groups',np.nan)),
                'Observed sensitivity (%)':100*sensitivity if pd.notna(sensitivity) else np.nan,
                'Descriptive AUC':row.get('auc',np.nan)})
        table=pd.DataFrame(rows)
        _pub_table(table,'Reference scenario 08 versus progressive-damage scenarios 09–17.',
            'Decisions use the descriptive maximum score of calibration reference scenario 02, without a nominal 95% guarantee; the nominal 95% rule abstains. Models are frozen. Altered counts are detected/assessed damage-phase scenarios; reference counts are alerts/assessed reference scenarios. Mechanisms and severities differ and may be cumulative; scenarios are not independent replications. Rates and AUC are descriptive. A missing class leaves its count-based rate and AUC not estimable. The full source CSV retains all confusion counts, specificity and decision statuses.',
            'z24_pdt_progressive_damage_phase',out,catalogue,'Main paper candidate')
    mismatch=_pub_read_csv(folder/'pdt_excitation_mismatch_summary.csv')
    if not mismatch.empty:
        reference=mismatch[mismatch.scenario.isin(['02','08'])].copy()
        if not reference.empty:
            fields=['scenario','method','matched_setups','incomplete_setups','raw_mean_mismatch_equal_setup_weight',
                    'corrected_mean_mismatch_equal_setup_weight','ratio_of_means_reduction']
            table=_pub_order(reference)[fields].copy();table['method']=table.method.map(_pub_label)
            table['ratio_of_means_reduction']*=100
            table.columns=['Reference scenario','Method','Matched setups','Incomplete setups','Uncorrected mean mismatch',
                           'Corrected mean mismatch','Ratio-of-means reduction (%)']
            _pub_table(table,'Setup-balanced ambient–forced excitation mismatch in the Z24 reference states.',
                'Distances compare AVT and FVT centroids within the same setup. Aggregation gives each matched setup equal weight. This is a descriptive normalisation metric, not a damage-detection rate.',
                'z24_pdt_reference_excitation_mismatch',out,catalogue,'Main paper candidate')
            proposed=reference[reference.method.eq('protected_nnls')].sort_values('scenario')
            if not proposed.empty:
                x=np.arange(len(proposed));fig,ax=plt.subplots(figsize=(width,2.4))
                ax.bar(x-.16,proposed.raw_mean_mismatch_equal_setup_weight,width=.3,color=PUBLICATION_COLOURS['raw'],label='Uncorrected')
                ax.bar(x+.16,proposed.corrected_mean_mismatch_equal_setup_weight,width=.3,color=PUBLICATION_COLOURS['protected_nnls'],label='Protected NNLS')
                ax.set(xticks=x,xticklabels=['Reference '+str(v) for v in proposed.scenario],ylabel='Mean within-setup AVT–FVT distance')
                ax.legend(frameon=False,fontsize=7.5);ax.grid(axis='y',color='#dddddd',lw=.5);ax.set_axisbelow(True);fig.tight_layout()
                _pub_figure(fig,'Ambient–forced excitation mismatch before and after protected normalisation in the Z24 reference states.',
                    'Each bar is an equal-weight mean over matched setups. Missing pairs are reported in the accompanying table. Bars are descriptive and have no independence-based uncertainty estimate.',
                    'z24_pdt_reference_excitation_mismatch',out,catalogue,dpi)
    sampling=_pub_read_csv(folder/'pdt_sampling_balance.csv')
    if not sampling.empty:
        table=sampling.drop(columns=['note'],errors='ignore').rename(columns={
            'scenario':'Scenario','phase':'Phase','acquisitions':'Acquisitions','setups':'Setups',
            'same_setup_excitation_pairs_as_source01':'Same setup–excitation pairs as scenario 01',
            'same_setup_excitation_counts_as_source01':'Same counts as scenario 01',
            'missing_source01_pairs':'Missing source pairs','additional_pairs':'Additional pairs'})
        _pub_table(table,'Acquisition and setup balance across Z24 structural scenarios.',
            'Unequal group sizes can affect group maxima. No balancing or deletion is performed using evaluation scores.',
            'z24_pdt_sampling_balance',out,catalogue)


def export_publication_outputs(run,config=None):
    """Create British-English tables, captions and vector/raster figures.

    Returns a JSON-safe catalogue. Only actual newly computed cohorts are used;
    archived-reporting results remain in archived_reporting/. Configurable style
    choices are implementation settings, not asserted conference obligations.
    """
    run=Path(run).resolve();config={} if config is None else config
    style=config.get('publication_style',{})
    width_mm=float(style.get('figure_width_mm',122.0));dpi=int(style.get('raster_dpi',300))
    font=float(style.get('font_size_pt',9.0))
    if not 60<=width_mm<=200 or not 150<=dpi<=1200 or not 6<=font<=12:
        raise ValueError('Invalid publication figure width, raster resolution or font size')
    width=width_mm/25.4;out=run/'publication';out.mkdir(parents=True,exist_ok=True)
    catalogue=[];errors=[]
    requested_font=str(style.get('font_family','Times New Roman'))
    selected_font=None
    for candidate_font in [requested_font,'Liberation Serif','DejaVu Serif']:
        try:
            font_manager.findfont(font_manager.FontProperties(family=candidate_font),fallback_to_default=False)
            selected_font=candidate_font;break
        except ValueError:
            continue
    if selected_font is None:raise RuntimeError('No suitable serif font is available for publication outputs')
    settings={'font.family':'serif','font.serif':[selected_font],'font.size':font,
        'axes.labelsize':font,'xtick.labelsize':font-1,'ytick.labelsize':font-1,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':dpi}
    with plt.rc_context(settings):
        _pub_partitions(run,out,catalogue);_pub_availability(run,out,catalogue)
        for folder in sorted((run/'new_analysis').glob('*')):
            if not folder.is_dir() or not (folder/'group_scores.csv').exists():continue
            try:
                _pub_coverage(folder,out,catalogue,width,dpi)
                _pub_score_figures(folder,out,catalogue,width,dpi)
                _pub_paired_table(folder,out,catalogue)
                _pub_retention(folder,out,catalogue,width,dpi)
                if folder.name=='z24_pdt':_pub_pdt(folder,out,catalogue,width,dpi)
            except Exception as exc:
                errors.append({'cohort':folder.name,'error':repr(exc)})
    caption_rows=[{k:item[k] for k in ['kind','number','stem','caption','notes','suggested_use']} for item in catalogue]
    pd.DataFrame(caption_rows,columns=['kind','number','stem','caption','notes','suggested_use']).to_csv(out/'captions.csv',index=False)
    captions='\n\n'.join('### '+item['kind']+' '+str(item['number'])+'\n\n'+item['caption']+
        '\n\n**Scientific note (for the manuscript text or table footnote):** '+item['notes'] for item in catalogue)
    (out/'captions.md').write_text('# Figure and table captions\n\nUse the short captions below. Scientific notes are supplied separately and need not be appended verbatim to every caption.\n\n'+captions+'\n',encoding='utf-8')
    sections=[]
    for item in catalogue:
        if item['kind']=='Table':
            sections.append((out/'tables'/(item['stem']+'.html')).read_text(encoding='utf-8'))
        else:
            png=next(p for p in item['files'] if p.endswith('.png'))
            sections.append('<figure><img src="'+html.escape(png)+'" alt="'+html.escape(item['caption'])+'">'
                '<figcaption>'+html.escape(item['caption'])+'</figcaption><p class="note">'+html.escape(item['notes'])+'</p></figure>')
    caution=('Only newly computed evidence is shown here. Archived evidence remains separate. '
        'Unavailable analyses are reported explicitly. Score contraction and synthetic retention are not proof of damage detection. '
        'Table titles are placed above tables and figure captions below figures. Figure numbering is a working catalogue; '
        'renumber all items and citations consistently when selecting material for the final paper. '
        'The final manuscript must be checked in the official AICSET template and within its page limit.')
    css='''body{font:15px/1.5 "Times New Roman","Liberation Serif",serif;max-width:1150px;margin:35px auto;padding:0 24px;color:#17202a;background:#fff}
        h1{font-size:26px}section,figure{margin:35px 0;break-inside:avoid}figure{max-width:850px}img{width:100%;height:auto}
        .caption,figcaption{font-size:14px;color:#111;margin:8px 0}.note{font-size:12px;color:#475569}
        table{border-collapse:collapse;font-size:12px;width:100%;margin:8px 0;display:block;overflow-x:auto}
        th{text-align:left;border-top:1.3px solid #111;border-bottom:1px solid #111}td,th{padding:6px 9px;vertical-align:top}
        tbody tr:last-child td{border-bottom:1.3px solid #111}td{border:0}a{color:#006b65}
        @media print{body{max-width:122mm;margin:0;padding:0}.caption,figcaption,table{font-size:9pt}img{max-width:122mm}section,figure{max-width:122mm}}'''
    page='<!doctype html><html lang="en-GB"><meta charset="utf-8"><title>AICSET 240 — publication outputs</title><style>'+css+'</style>'
    page+='<h1>AICSET 240: publication outputs</h1><p>'+html.escape(caution)+'</p>'+''.join(sections)+'</html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    manifest={'language':'British English','evidence_origin':'NEW_RAW_SIGNAL_REANALYSIS_ONLY',
        'raw_signals_reprocessed':False,'models_refitted':False,'thresholds_changed':False,
        'figure_width_mm':width_mm,'raster_dpi':dpi,'font_size_pt':font,
        'font_family':selected_font,'requested_font_family':requested_font,
        'style_status':'AUTHOR_PRESENTATION_SETTINGS; verify final placement against the official conference template',
        'table_title_position':'above','figure_caption_position':'below','items':catalogue,
        'figures':sum(x['kind']=='Figure' for x in catalogue),'tables':sum(x['kind']=='Table' for x in catalogue),
        'errors':errors,'preview':'publication/index.html'}
    (out/'manifest.json').write_text(json.dumps(_pub_clean(manifest),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return manifest
