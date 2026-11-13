"""New raw-signal reanalysis orchestrator; archived findings are never overwritten."""
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt


def source_file_identity(path):
    stat=Path(path).stat()
    return {'device':stat.st_dev,'inode':stat.st_ino,'size_bytes':stat.st_size,
            'mtime_ns':stat.st_mtime_ns,'ctime_ns':stat.st_ctime_ns}


def spectral_cache_record(row, config, cache_dir, hashes, source_identities):
    path=Path(row['path'])
    identity=source_file_identity(path)
    if str(path) in source_identities and source_identities[str(path)]!=identity:
        raise RuntimeError('Raw source changed after its first hash in this run: '+str(path))
    if str(path) not in hashes:
        digest=file_sha256(path)
        if source_file_identity(path)!=identity:raise RuntimeError('Raw source changed during hashing: '+str(path))
        hashes[str(path)]=digest;source_identities[str(path)]=identity
    selector={k:row.get(k,'') for k in ['record_id','fs_hz','data_key','channel_keys','channel_columns','channel_names','sample_axis','delimiter','decimal','skiprows','header','time_column']}
    key=configuration_sha256({'file_sha256':hashes[str(path)],'selector':selector,
        'nperseg':config['nperseg'],'overlap':config['overlap'],'implementation':'V4_CPSD'})
    dest=Path(cache_dir)/(key+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        with np.load(dest,allow_pickle=False) as z:
            cached=(z['frequency'].copy(),z['singular'].copy(),json.loads(str(z['qc_json'])))
        if source_file_identity(path)!=identity:raise RuntimeError('Raw source changed during cache lookup: '+str(path))
        return cached
    signal,fs,channels=load_acquisition(row)
    f,s=cpsd_singular_spectra(signal,fs,nperseg=config['nperseg'],overlap=config['overlap'])
    if source_file_identity(path)!=identity:raise RuntimeError('Raw source changed between hash and spectral extraction: '+str(path))
    qc={'record_id':row['record_id'],'acquisition_id':row['acquisition_id'],'status':'PASS',
        'samples':len(signal),'channels':channels,'fs_hz':fs,'source_sha256':hashes[str(path)],
        'duration_seconds':len(signal)/fs,'selector_sha256':configuration_sha256(selector)}
    step=int(config['nperseg'])-int(np.floor(config['overlap']*config['nperseg']))
    qc['welch_segments']=1+(len(signal)-int(config['nperseg']))//step
    np.savez_compressed(dest,frequency=f,singular=s,qc_json=np.array(json.dumps(qc)))
    return f,s,qc


def source_pdt_displacements(Xfit,fitmeta):
    vectors=[];audit=[]
    for setup,part in fitmeta.groupby('setup_id',sort=True):
        avt=part.index[part.excitation.str.upper()=='AVT'].to_numpy()
        fvt=part.index[part.excitation.str.upper()=='FVT'].to_numpy()
        audit.append({'setup_id':setup,'avt_records':len(avt),'fvt_records':len(fvt)})
        if not len(avt) or not len(fvt):raise ValueError(f'PDT source01 setup {setup} lacks paired AVT/FVT records')
        vectors.append(Xfit[fvt].mean(axis=0)-Xfit[avt].mean(axis=0))
    return np.asarray(vectors),pd.DataFrame(audit)


# Short paraphrases of the official KU Leuven scenario descriptions.
# https://bwk.kuleuven.be/bwm/z24 — chronology, not inferred signal labels.
PDT_SCENARIOS={
    '01':'Undamaged baseline','02':'Settlement apparatus installed',
    '03':'Pier settlement 20 mm','04':'Pier settlement 40 mm',
    '05':'Pier settlement 80 mm','06':'Pier settlement 95 mm',
    '07':'Pier lifted; foundation tilted','08':'New reference',
    '09':'Soffit spalling 12 m2','10':'Soffit spalling 24 m2',
    '11':'Abutment slide 1 m','12':'Concrete hinge damaged',
    '13':'Two anchor heads failed','14':'Four anchor heads failed',
    '15':'Two of 16 tendons ruptured','16':'Four of 16 tendons ruptured',
    '17':'Six of 16 tendons ruptured'}


def pdt_phase(scenario):
    n=int(scenario)
    return 'REFERENCE' if n in {1,2,8} else ('PIER_INTERVENTION' if n<=7 else 'PROGRESSIVE_DAMAGE')


def pdt_sampling_audit(d,out):
    """Describe layout/excitation imbalance without selecting on outcome scores."""
    source=d[d.scenario.eq('01')].groupby(['setup_id','excitation']).size().to_dict()
    rows=[]
    for scenario,part in d.groupby('scenario',sort=True):
        pattern=part.groupby(['setup_id','excitation']).size().to_dict()
        rows.append({'scenario':scenario,'phase':pdt_phase(scenario),'acquisitions':len(part),
            'setups':part.setup_id.nunique(),'same_setup_excitation_pairs_as_source01':set(pattern)==set(source),
            'same_setup_excitation_counts_as_source01':pattern==source,
            'missing_source01_pairs':json.dumps(sorted(set(source)-set(pattern))),
            'additional_pairs':json.dumps(sorted(set(pattern)-set(source))),
            'note':'Unequal counts can change group maxima; no outcome-based balancing or deletion performed.'})
    frame=pd.DataFrame(rows);frame.to_csv(Path(out)/'pdt_sampling_balance.csv',index=False)
    return frame


def export_pdt_mismatch(d,L,C,responders,family,out):
    """Within-scenario AVT/FVT centroids: same setup, equal setup aggregation.

    A mismatch is a feature-space distance, not a damage detection statistic.
    Missing excitation pairs remain explicit; no raw acquisition is dropped.
    """
    vectors={m:r(L,C) for m,r in responders.items() if family[m]=='NORMALIZED_LOG_SINGULAR_SPECTRA'}
    rows=[]
    for (scenario,setup),part in d.groupby(['scenario','setup_id'],sort=True):
        avt=part.index[part.excitation.str.upper().eq('AVT')].to_numpy()
        fvt=part.index[part.excitation.str.upper().eq('FVT')].to_numpy()
        paired=bool(len(avt) and len(fvt))
        raw_distance=float(np.linalg.norm(vectors['raw'][fvt].mean(axis=0)-vectors['raw'][avt].mean(axis=0))) if paired else np.nan
        for method,Z in vectors.items():
            distance=float(np.linalg.norm(Z[fvt].mean(axis=0)-Z[avt].mean(axis=0))) if paired else np.nan
            rows.append({'scenario':scenario,'scenario_description':PDT_SCENARIOS[scenario],
                'phase':pdt_phase(scenario),'partition':part.partition.iloc[0],
                'setup_id':setup,'method':method,'avt_acquisitions':len(avt),'fvt_acquisitions':len(fvt),
                'pairing_status':'MATCHED' if paired else 'MISSING_AVT_OR_FVT',
                'raw_mismatch':raw_distance,'corrected_mismatch':distance,
                'relative_mismatch_reduction':1-distance/raw_distance if raw_distance>1e-12 else np.nan,
                'evidence_origin':'NEW_RAW_SIGNAL_REANALYSIS','interpretation':'DESCRIPTIVE_EXCITATION_MISMATCH'})
    frame=pd.DataFrame(rows);frame.to_csv(Path(out)/'pdt_excitation_mismatch_by_setup.csv',index=False)
    summaries=[]
    for (scenario,method),part in frame.groupby(['scenario','method'],sort=True):
        pairs=part[part.pairing_status.eq('MATCHED')]
        raw=float(pairs.raw_mismatch.mean()) if len(pairs) else np.nan
        corrected=float(pairs.corrected_mismatch.mean()) if len(pairs) else np.nan
        summaries.append({'scenario':scenario,'scenario_description':PDT_SCENARIOS[scenario],
            'phase':pdt_phase(scenario),'partition':part.partition.iloc[0],'method':method,
            'matched_setups':len(pairs),'incomplete_setups':len(part)-len(pairs),
            'raw_mean_mismatch_equal_setup_weight':raw,'corrected_mean_mismatch_equal_setup_weight':corrected,
            'ratio_of_means_reduction':1-corrected/raw if raw>1e-12 else np.nan,
            'median_paired_setup_relative_reduction':float(pairs.relative_mismatch_reduction.median()) if len(pairs) else np.nan,
            'interpretation':'DESCRIPTIVE_NOT_AN_INDEPENDENT_SCENARIO_TEST'})
    pd.DataFrame(summaries).to_csv(Path(out)/'pdt_excitation_mismatch_summary.csv',index=False)


def export_group_decisions(groups,thresholds,d,out,dataset):
    """Expose every physical-group decision, including abstentions and losses."""
    records=[]
    for group,part in d.groupby('group_id',sort=True):
        scenario=part.scenario.iloc[0]
        for method,series in groups.items():
            q=thresholds[method];score=float(series.loc[group]);finite=bool(q['finite_supported'])
            records.append({'dataset':dataset,'group_id':group,'scenario':scenario,
                'scenario_description':PDT_SCENARIOS.get(scenario,'') if dataset=='z24_pdt' else '',
                'phase':pdt_phase(scenario) if dataset=='z24_pdt' else '',
                'state_label':part.state_label.iloc[0],'partition':part.partition.iloc[0],
                'method':method,'acquisitions_in_group':len(part),'score':score,
                'threshold':q['threshold'],'decision':'ALERT' if finite and score>q['threshold'] else ('NO_ALERT' if finite else 'ABSTAIN'),
                'decision_scope':'NOMINAL_ORDER_STATISTIC_RULE_CONDITIONAL_ON_UNVERIFIED_ASSUMPTIONS',
                'evidence_origin':'NEW_RAW_SIGNAL_REANALYSIS'})
    decisions=pd.DataFrame(records)
    decisions.to_csv(Path(out)/'physical_group_decisions.csv',index=False)
    if dataset!='z24_pdt':return
    calids=d.loc[d.partition.eq('calibration'),'group_id'].unique()
    long=[]
    for item in records:
        if item['partition']!='test':continue
        q=float(groups[item['method']].reindex(calids).max())
        long.append({**item,'threshold':q,'decision':'ALERT' if item['score']>q else 'NO_ALERT',
            'decision_scope':'DESCRIPTIVE_CALIBRATION_MAXIMUM_NO_NOMINAL_GUARANTEE'})
    desc=pd.DataFrame(long);desc.to_csv(Path(out)/'pdt_scenario_descriptive_decisions.csv',index=False)
    if 'protected_nnls' in groups:
        columns=['group_id','scenario','scenario_description','phase','state_label']
        raw=desc[desc.method.eq('raw')][columns+['score','threshold','decision']]
        proposed=desc[desc.method.eq('protected_nnls')][['group_id','score','threshold','decision']]
        side=raw.merge(proposed,on='group_id',suffixes=('_raw','_protected'),validate='one_to_one')
        side['raw_alert_lost_after_correction']=side.decision_raw.eq('ALERT')&side.decision_protected.eq('NO_ALERT')
        side['new_alert_after_correction']=side.decision_raw.eq('NO_ALERT')&side.decision_protected.eq('ALERT')
        side['interpretation']='DESCRIPTIVE; an alert on altered state may indicate sensitivity; an alert on reference may be false'
        side.to_csv(Path(out)/'pdt_raw_vs_protected_by_scenario.csv',index=False)


def compute_cohort(dataset, metadata, config, run):
    out=Path(run)/'new_analysis'/dataset;out.mkdir(parents=True,exist_ok=True)
    d=assign_partitions(metadata,config).reset_index(drop=True)
    d.to_csv(out/'physical_group_partitions.csv',index=False)
    d=d[d.partition.isin(['fit','calibration','test'])].reset_index(drop=True)
    if dataset=='z24_pdt':
        inventory=d.groupby(['scenario','partition','setup_id','excitation']).size().rename('acquisitions').reset_index()
        inventory['phase']=inventory.scenario.map(pdt_phase)
        inventory.to_csv(out/'pdt_scenario_inventory.csv',index=False)
        sampling_audit=pdt_sampling_audit(d,out)
    if not len(d):raise ValueError('No confirmed records in fitting/calibration/test partitions')
    if d[d.partition=='fit'].state_label.ne('reference').any():raise AssertionError('Altered states entered fitting')
    if d[d.partition=='calibration'].state_label.ne('reference').any():raise AssertionError('Altered states entered calibration')
    specs=[];qc=[];failures=[];hashes={};source_identities={};signatures=[]
    for i,row in d.iterrows():
        try:
            f,s,q=spectral_cache_record(row.to_dict(),config,Path(run)/'cache',hashes,source_identities)
            specs.append((f,s));qc.append(q);signatures.append(tuple(q['channels']))
        except Exception as exc:
            failures.append({'record_id':row.record_id,'path':row.path,'error':repr(exc)})
            specs.append(None)
        if (i+1)%10==0 or i+1==len(d):print(f'{dataset}: spectral QC {i+1}/{len(d)}',flush=True)
    pd.DataFrame(qc).to_csv(out/'acquisition_qc.csv',index=False)
    json_write(out/'raw_source_hashes.json',hashes)
    json_write(out/'raw_source_identity.json',source_identities)
    if failures:
        pd.DataFrame(failures).to_csv(out/'acquisition_failures.csv',index=False)
        raise RuntimeError('QC failed: inspect acquisition_failures.csv. No outcome-dependent row deletion is performed.')
    if len(set(signatures))!=1:
        raise ValueError('Channel names/order differ within cohort. Provide explicit consistent channel selectors; do not merge roving layouts.')
    if d.fs_hz.astype(float).nunique()!=1:raise ValueError('Mixed sampling rates require a separately specified harmonization protocol')
    fit=d.partition.eq('fit').to_numpy();cal=d.partition.eq('calibration').to_numpy();test=d.partition.eq('test').to_numpy()
    # Reference selection gives every fitting physical group one spectral summary.
    training=[]
    for group,part in d[fit].groupby('group_id'):
        arrays=[specs[i] for i in part.index];f0=arrays[0][0]
        if any(not np.array_equal(f0,f) for f,_ in arrays):raise ValueError('Incompatible fitting frequency grids')
        training.append((f0,np.mean([s for _,s in arrays],axis=0)))
    override=config['f1_overrides_hz'].get(dataset)
    if override is None:
        f1,audit=estimate_reference_f1(training,band=tuple(config['f1_peak_band_hz']))
        audit['physical_mode_identification']='SPECTRAL_HEURISTIC_REQUIRES_PHYSICAL_INTERPRETATION_CHECK'
    else:
        f1=float(override);audit={'selection':'USER_SUPPLIED_FITTING_ONLY_OVERRIDE','selected_f1_hz':f1}
    f1_confirmed=bool(config.get('f1_review_confirmed',{}).get(dataset,False))
    f1_note=str(config.get('f1_review_notes',{}).get(dataset,'')).strip()
    if f1_confirmed and not f1_note:raise ValueError('Physical f1 confirmation requires a fitting-only provenance note')
    audit.update({'physical_mode_review_confirmed_by_user':f1_confirmed,'physical_mode_review_note':f1_note,
                  'physical_mode_identity_established_by_code':False})
    json_write(out/'reference_frequency_audit.json',audit)
    if 'frequency_hz' in audit:
        fig,ax=plt.subplots(figsize=(7,2.6));ax.plot(audit['frequency_hz'],audit['median_unit_height_sigma1'])
        ax.axvline(f1,color='#b91c1c',ls='--');ax.set(xlabel='Frequency (Hz)',ylabel='Fitting spectral summary',title=f'{dataset}: frozen reference frequency {f1:.4g} Hz')
        fig.tight_layout();fig.savefig(out/'reference_frequency_check.png',dpi=200);plt.close(fig)
    upper=config['frequency_upper'][dataset];rho=np.linspace(.25,upper,config['bins_per_curve'])
    L=np.stack([grid_log_curves(f,s,f1,upper=upper,bins=config['bins_per_curve']) for f,s in specs])
    X=normalize_curves(L);fitgroups=d.loc[fit,'group_id'].to_numpy()
    fitL=L[fit];fitX=X[fit];fitmeta=d.loc[fit].reset_index(drop=True)
    groupL=np.stack([fitL[fitgroups==g].mean(axis=0) for g in np.unique(fitgroups)])
    reference=groupL.mean(axis=0);U=build_protected_basis(reference,shifts=config['design_shifts'],grid=rho)
    modeldir=out/'fitted_models';modeldir.mkdir(exist_ok=True)
    extra={'dataset':dataset,'f1_hz':f1,'grid':rho.tolist(),'fit_groups':sorted(set(fitgroups)),
           'configuration_sha256':configuration_sha256(config),'new_implementation_not_historical':True}
    mean=group_centroids(fitX,fitgroups)[0].mean(axis=0)
    np.savez_compressed(modeldir/'raw_reference.npz',mean=mean,grid=rho,protected_basis=U,reference_log_curves=reference)
    scorers={'raw':lambda Z,C:np.linalg.norm(normalize_curves(Z)-mean,axis=1)}
    responders={'raw':lambda Z,C:normalize_curves(Z)-mean}
    family={'raw':'NORMALIZED_LOG_SINGULAR_SPECTRA'};errors=[]
    normalizer=None
    try:
        vectors=None
        if dataset=='z24_pdt':
            vectors,pairs=source_pdt_displacements(fitX,fitmeta);pairs.to_csv(out/'source01_excitation_pairs.csv',index=False)
        normalizer=SpectralNormalizer(rank=config['rank']).fit(fitX,fitgroups,protected_basis=U,nuisance_vectors=vectors)
        corrected=normalizer.transform(X,'protected_nnls')
        coefficients=normalizer.coefficients(X,'protected_nnls')
        constraint_audit={
            'protected_orthonormality_max_abs_error':float(np.max(np.abs(U.T@U-np.eye(U.shape[1])))),
            'dictionary_protection_max_abs_inner_product':float(np.max(np.abs(U.T@normalizer.B_))),
            'protected_coordinate_max_abs_change':float(np.max(np.abs((corrected-X)@U))),
            'minimum_nnls_coefficient':float(coefficients.min()),
            'largest_residual_norm_increase':float(np.max(np.linalg.norm(corrected-mean,axis=1)-np.linalg.norm(X-mean,axis=1))),
            'synthetic_response_retention_is_not_real_damage_validation':True}
        json_write(out/'constraint_audit.json',constraint_audit)
        for key in ['protected_orthonormality_max_abs_error','dictionary_protection_max_abs_inner_product',
                    'protected_coordinate_max_abs_change','largest_residual_norm_increase']:
            if constraint_audit[key]>1e-8:raise ArithmeticError('Protected model constraint failed: '+key)
        if constraint_audit['minimum_nnls_coefficient'] < -1e-12:raise ArithmeticError('Negative NNLS coefficient')
        normalizer.save(modeldir/'protected_dictionary',extra=extra)
        for method in ['signed_pca','protected_signed','protected_nnls']:
            scorers[method]=lambda Z,C,m=method,obj=normalizer:obj.score(normalize_curves(Z),m)
            responders[method]=lambda Z,C,m=method,obj=normalizer:obj.transform(normalize_curves(Z),m)-obj.mean_
            family[method]='NORMALIZED_LOG_SINGULAR_SPECTRA'
    except Exception as exc:errors.append({'method':'protected_dictionary_family','error':repr(exc)})
    try:
        conventional=ConventionalPCA(rank=config['rank']).fit(fitX,fitgroups)
        conventional.save(modeldir/'baseline_pca',extra=extra)
        scorers['baseline_pca']=lambda Z,C,obj=conventional:obj.score(normalize_curves(Z))
        responders['baseline_pca']=lambda Z,C,obj=conventional:obj.transform(normalize_curves(Z))-obj.mean_
        family['baseline_pca']='NORMALIZED_LOG_SINGULAR_SPECTRA'
    except Exception as exc:errors.append({'method':'baseline_pca','error':repr(exc)})
    try:
        ratios=FrequencyRatioBaseline(grid=rho).fit(fitL,fitgroups);ratios.save(modeldir/'frequency_ratio',extra=extra)
        tracked,tracking=ratios.tracked_frequencies(L)
        trackrows=[]
        for i,row in d.iterrows():
            for band in range(tracked.shape[1]):
                trackrows.append({'record_id':row.record_id,'group_id':row.group_id,'partition':row.partition,
                    'band_index':band,'frequency_over_f1':tracked[i,band],'frequency_hz':tracked[i,band]*f1,
                    'band_edge_maximum':bool(tracking['band_edge_maximum'][i,band]),
                    'quadratic_interpolation_used':bool(tracking['quadratic_interpolation_used'][i,band]),
                    'physical_modal_identification':False})
        pd.DataFrame(trackrows).to_csv(out/'frequency_ratio_peak_tracking.csv',index=False)
        scorers['frequency_ratio']=lambda Z,C,obj=ratios:obj.score(Z)
        responders['frequency_ratio']=lambda Z,C,obj=ratios:obj.transform(Z)
        family['frequency_ratio']='TRAINING_SELECTED_PEAK_FREQUENCY_RATIOS'
    except Exception as exc:errors.append({'method':'frequency_ratio','error':repr(exc)})
    covcols=[c for c in d if c.startswith('cov_')]
    if 'cov_temperature' in covcols:covcols=['cov_temperature']+[c for c in covcols if c!='cov_temperature']
    C=d[covcols].apply(pd.to_numeric,errors='raise').to_numpy(float) if covcols else np.empty((len(d),0))
    for method,piecewise in [('temperature_ridge',False),('piecewise_temperature',True)]:
        try:
            if len(np.unique(fitgroups))<3:raise ValueError('NOT_ESTIMABLE_BY_PROTOCOL: grouped temperature regression requires at least three fitting groups; PDT source01 provides only one')
            if 'cov_temperature' not in covcols:raise ValueError('Provide measured cov_temperature metadata; no temperature values are fabricated')
            # A simple temperature baseline uses one temperature predictor only.
            cov=C[:,:1]
            obj=TemperatureRidge(piecewise=piecewise,alpha_grid=config['ridge_alphas']).fit(fitX,cov[fit],fitgroups)
            obj.save(modeldir/method,extra={**extra,'covariate_columns':['cov_temperature']})
            scorers[method]=lambda Z,V,obj=obj:obj.score(normalize_curves(Z),V[:,:1])
            responders[method]=lambda Z,V,obj=obj:obj.transform(normalize_curves(Z),V[:,:1])-obj.mean_
            family[method]='NORMALIZED_LOG_SINGULAR_SPECTRA'
        except Exception as exc:errors.append({'method':method,'error':repr(exc)})
    if len(covcols)>1:
        try:
            obj=TemperatureRidge(alpha_grid=config['ridge_alphas']).fit(fitX,C[fit],fitgroups)
            obj.save(modeldir/'conditional_ridge',extra={**extra,'covariate_columns':covcols})
            scorers['conditional_ridge']=lambda Z,V,obj=obj:obj.score(normalize_curves(Z),V)
            responders['conditional_ridge']=lambda Z,V,obj=obj:obj.transform(normalize_curves(Z),V)-obj.mean_
            family['conditional_ridge']='NORMALIZED_LOG_SINGULAR_SPECTRA'
        except Exception as exc:errors.append({'method':'conditional_ridge','error':repr(exc)})
    json_write(out/'unavailable_methods.json',errors)
    rows=[];groups={};thresholds={};coverage=[];damage=[];descriptive=[];rocframes=[];paired=[]
    labels=d.groupby('group_id').state_label.first()
    testids=d.loc[test,'group_id'].unique();refids=labels.index[labels.eq('reference')].intersection(testids)
    for method,scorer in scorers.items():
        values=np.asarray(scorer(L,C),float)
        if not np.isfinite(values).all():raise ValueError(f'{method}: nonfinite scores')
        for i,value in enumerate(values):rows.append({**d.loc[i,['record_id','acquisition_id','group_id','state_label','partition','scenario','excitation','setup_id']].to_dict(),
            'dataset':dataset,'method':method,'feature_family':family[method],'score':float(value),'evidence_origin':'NEW_RAW_SIGNAL_REANALYSIS'})
        groups[method]=grouped_max(values,d.group_id)
        q=conformal_threshold(grouped_max(values[cal],d.loc[cal,'group_id']),alpha=config['alpha']);thresholds[method]=q
        summary=coverage_summary(groups[method].reindex(refids),q)
        coverage.append({'dataset':dataset,'method':method,**q,**summary})
        if labels.reindex(testids).eq('altered').any():
            summary,roc=damage_discrimination(groups[method].reindex(testids),labels.reindex(testids),q)
            damage.append({'dataset':dataset,'method':method,'threshold_rule':'NOMINAL_CONFORMAL_ORDER_STATISTIC',**summary})
            rocframes.append(roc.assign(dataset=dataset,method=method))
            # A separately labelled descriptive rule supplies detection counts
            # when scarce PDT calibration cannot support a finite 95% rule.
            calvalues=grouped_max(values[cal],d.loc[cal,'group_id'])
            if len(calvalues):
                desc={'threshold':float(calvalues.max()),'finite_supported':True,
                      'reason':'DESCRIPTIVE_CALIBRATION_MAXIMUM_NOT_A_95_PERCENT_GUARANTEE'}
                item,_=damage_discrimination(groups[method].reindex(testids),labels.reindex(testids),desc)
                descriptive.append({'dataset':dataset,'method':method,'calibration_groups':len(calvalues),
                    'threshold_rule':'DESCRIPTIVE_CALIBRATION_MAXIMUM','nominal_coverage_supported':q['finite_supported'],'nominal_reference_coverage_target':1-config['alpha'],
                    'population_detection_claim_supported':False,**item})
    pd.DataFrame(rows).to_csv(out/'acquisition_scores.csv',index=False)
    grow=[]
    for method,series in groups.items():
        for group,value in series.items():grow.append({'dataset':dataset,'method':method,'feature_family':family[method],'group_id':group,'score':value,
            'state_label':labels[group],'partition':d.loc[d.group_id.eq(group),'partition'].iloc[0],'evidence_origin':'NEW_RAW_SIGNAL_REANALYSIS'})
    pd.DataFrame(grow).to_csv(out/'group_scores.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'reference_coverage.csv',index=False);json_write(out/'thresholds.json',thresholds)
    export_group_decisions(groups,thresholds,d,out,dataset)
    if dataset=='z24_pdt':
        export_pdt_mismatch(d,L,C,responders,family,out)
        # Predefined phase subset: test08 reference vs test09--17 damage phase.
        phaseids=d.loc[test & (d.scenario.eq('08')|d.scenario.isin([f'{i:02d}' for i in range(9,18)])),'group_id'].unique()
        phase_rows=[]
        if len(phaseids):
            for method,series in groups.items():
                nominal,_=damage_discrimination(series.reindex(phaseids),labels.reindex(phaseids),thresholds[method])
                descriptive_q={'threshold':float(series.reindex(d.loc[cal,'group_id'].unique()).max()),'finite_supported':True,
                    'reason':'DESCRIPTIVE_CALIBRATION_MAXIMUM_NOT_A_NOMINAL_GUARANTEE'}
                descriptive_item,_=damage_discrimination(series.reindex(phaseids),labels.reindex(phaseids),descriptive_q)
                for rule,summary in [('NOMINAL_CONFORMAL_ORDER_STATISTIC',nominal),('DESCRIPTIVE_CALIBRATION_MAXIMUM',descriptive_item)]:
                    phase_rows.append({'dataset':dataset,'method':method,'subset':'REFERENCE08_VS_PROGRESSIVE_DAMAGE09_TO17',
                        'threshold_rule':rule,'population_detection_claim_supported':False,**summary})
        pd.DataFrame(phase_rows).to_csv(out/'pdt_progressive_damage_phase.csv',index=False)
    for method in scorers:
        if method=='raw' or len(refids)==0:continue
        if family[method]!=family['raw']:
            if not thresholds['raw']['finite_supported'] or not thresholds[method]['finite_supported']:continue
            # Binary decisions can be paired across feature families without
            # dividing by a possibly zero calibration threshold.
            v=paired_decision_comparison(groups['raw'].reindex(refids),groups[method].reindex(refids),thresholds['raw'],thresholds[method])
            paired.append({'dataset':dataset,'method':method,'comparison_scope':'BINARY_DECISIONS_ACROSS_DIFFERENT_FEATURE_FAMILIES',**v})
        else:
            v=paired_group_comparison(groups['raw'].reindex(refids),groups[method].reindex(refids),thresholds['raw'],thresholds[method],
                config['bootstrap_replicates'] if len(refids)>=3 else 0,config['seed'])
            paired.append({'dataset':dataset,'method':method,'comparison_scope':'SAME_SPECTRAL_FEATURE_FAMILY',**v})
    if paired:
        p=pd.DataFrame(paired)
        valid=p.mcnemar_exact_two_sided_p.notna() if 'mcnemar_exact_two_sided_p' in p else pd.Series(False,index=p.index)
        if valid.any():p.loc[valid,'mcnemar_p_holm']=holm_adjust(p.loc[valid,'mcnemar_exact_two_sided_p'].to_numpy())
        p.to_csv(out/'paired_group_comparisons.csv',index=False)
    if damage:pd.DataFrame(damage).to_csv(out/'damage_discrimination.csv',index=False)
    if descriptive:pd.DataFrame(descriptive).to_csv(out/'damage_descriptive_detection.csv',index=False)
    if rocframes:pd.concat(rocframes,ignore_index=True).to_csv(out/'damage_roc.csv',index=False)
    retention=[]
    for group,part in d[test].groupby('group_id'):
        base=L[part.index].mean(axis=0)[None,...];cov=C[part.index].mean(axis=0,keepdims=True) if C.shape[1] else np.empty((1,0))
        for shift in config['design_shifts']+config['additional_test_shifts']:
            shifted=shift_log_curves(base,shift,grid=rho)
            for method,response in responders.items():
                a0=response(base,cov);a1=response(shifted,cov)
                raw_delta=normalize_curves(shifted)-normalize_curves(base)
                same_family=family[method]=='NORMALIZED_LOG_SINGULAR_SPECTRA'
                denom=np.linalg.norm(raw_delta) if same_family else np.nan
                ratio=float(np.linalg.norm(a1-a0)/denom) if denom>1e-10 else np.nan
                retention.append({'dataset':dataset,'group_id':group,'method':method,'relative_frequency_shift':shift,
                    'shift_set':'DESIGN_MAGNITUDE_ON_TEST_GROUP_SHAPE' if shift in config['design_shifts'] else 'ADDITIONAL_MAGNITUDE_ON_TEST_GROUP_SHAPE',
                    'retention_ratio':ratio,'zero_raw_response':bool(denom<=1e-10) if same_family else None,
                    'retention_status':'OK' if same_family and denom>1e-10 else ('ZERO_SYNTHETIC_RESPONSE' if same_family else 'DIFFERENT_FEATURE_FAMILY_NOT_COMPARABLE'),
                    'feature_family':family[method],'real_damage_simulation':False})
    pd.DataFrame(retention).to_csv(out/'synthetic_shift_retention.csv',index=False)
    # These arrays enable reproduction of new transformations without refitting.
    np.savez_compressed(out/'derived_features.npz',X=X,log_curves=L,record_ids=d.record_id.to_numpy(dtype=str),grid=rho,f1_hz=f1)
    required={'raw','protected_nnls','baseline_pca','frequency_ratio'}
    if dataset!='z24_pdt':required.add('temperature_ridge')
    missing_required=sorted(required-set(scorers))
    status={'dataset':dataset,'status':'PARTIAL_NEW_ANALYSIS' if missing_required else 'COMPLETED_NEW_ANALYSIS','records':len(d),'methods':list(scorers),
        'required_methods_missing':missing_required,
        'fit_groups':int(d[fit].group_id.nunique()),'calibration_groups':int(d[cal].group_id.nunique()),
        'test_groups':int(d[test].group_id.nunique()),'reference_test_groups':len(refids),
        'altered_test_groups':int(labels.reindex(testids).eq('altered').sum()),'unavailable_methods':errors,
        'f1_hz':f1,'f1_physical_identity_review_required':not f1_confirmed,
        'independent_group_assumptions_established':False,'historical_pipeline_reproduced':False}
    if dataset=='z24_pdt':
        status['observed_scenarios']=sorted(d.scenario.unique().tolist())
        status['missing_official_scenarios']=sorted(set(PDT_SCENARIOS)-set(d.scenario))
        status['temperature_regression_estimable_under_frozen_protocol']=False
        status['progressive_damage_test_groups']=int(d.loc[test & d.scenario.isin([f'{i:02d}' for i in range(9,18)]),'group_id'].nunique())
        status['all_scenario_sampling_counts_match_source01']=bool(sampling_audit.same_setup_excitation_counts_as_source01.all())
    json_write(out/'cohort_status.json',status)
    plot_new_results(out,dataset)
    return status


def plot_new_results(out,dataset):
    out=Path(out);frame=pd.read_csv(out/'group_scores.csv');test=frame[frame.partition=='test']
    families=test.feature_family.unique();fig,axes=plt.subplots(len(families),1,figsize=(8,3.1*len(families)),squeeze=False)
    for ax,feature_family in zip(axes.ravel(),families):
        for method,part in test[test.feature_family.eq(feature_family)].groupby('method',sort=False):
            part=part.sort_values('group_id');ax.plot(np.arange(1,len(part)+1),part.score,marker='o',ms=3,label=method)
        ax.set(xlabel='Physical test group index (CSV identifiers)',ylabel='Distance within feature family',title=dataset+' / '+feature_family)
        ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'new_group_scores.png',dpi=300);fig.savefig(out/'new_group_scores.pdf');plt.close(fig)
    path=out/'damage_roc.csv'
    if path.exists():
        roc=pd.read_csv(path);fig,ax=plt.subplots(figsize=(4.7,4))
        for method,part in roc.groupby('method',sort=False):ax.plot(part.false_positive_rate,part.true_positive_rate,marker='.',label=method)
        ax.plot([0,1],[0,1],'k--',lw=.7);ax.set(xlabel='Reference-group false-positive fraction',ylabel='Altered-group true-positive fraction',title='Descriptive damage discrimination')
        ax.legend(fontsize=6);fig.tight_layout();fig.savefig(out/'new_damage_roc.png',dpi=300);fig.savefig(out/'new_damage_roc.pdf');plt.close(fig)
