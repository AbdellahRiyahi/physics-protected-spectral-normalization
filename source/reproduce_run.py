from pathlib import Path
import json,os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
SOURCE_FOLDER=Path(__file__).resolve().parent
CONFIG=json.loads((SOURCE_FOLDER.parent/'configuration_frozen_before_computation.json').read_text())
MODULE_NAMES=['raw_io.py','environment.py','models.py','statistics.py','support.py','pipeline.py','publication.py']
IMPLEMENTATION_SOURCE={name:(SOURCE_FOLDER/name).read_text() for name in MODULE_NAMES}
for module_name,module_source in IMPLEMENTATION_SOURCE.items():
    exec(compile(module_source,str(SOURCE_FOLDER/module_name),'exec'),globals())
ARCHIVED_SNAPSHOT=json.loads((SOURCE_FOLDER/'archived_evidence.json').read_text())
LOCAL_IMPLEMENTATION_VERIFICATION=json.loads((SOURCE_FOLDER/'local_development_controls.json').read_text())
UNIT_IMPLEMENTATION_VERIFICATION=json.loads((SOURCE_FOLDER/'unit_development_controls.json').read_text())
AICSET_FORMAT_REQUIREMENTS=json.loads((SOURCE_FOLDER/'aicset_format_requirements.json').read_text())
PREVIOUS_RUN_AUDIT=json.loads((SOURCE_FOLDER/'previous_run_audit.json').read_text())
REGRESSION_TEST_SOURCE={name:(SOURCE_FOLDER/name).read_text() for name in ['test_io_v4.py','test_models_v4.py','test_statistics_v4.py','test_publication_v4.py','test_environment_v5.py']}
REPRODUCTION_DRIVER=Path(__file__).read_text()


RUN, RUNTIME = initialize_run(CONFIG)
print('Run directory:', RUN)
SOURCE_DIR = RUN / 'source'; SOURCE_DIR.mkdir()
for module_name, module_source in IMPLEMENTATION_SOURCE.items():
    (SOURCE_DIR / module_name).write_text(module_source, encoding='utf-8')
for test_name, test_source in REGRESSION_TEST_SOURCE.items():
    (SOURCE_DIR / test_name).write_text(test_source, encoding='utf-8')
(SOURCE_DIR / 'reproduce_run.py').write_text(REPRODUCTION_DRIVER, encoding='utf-8')
json_write(SOURCE_DIR / 'archived_evidence.json', ARCHIVED_SNAPSHOT)
json_write(SOURCE_DIR / 'local_development_controls.json', LOCAL_IMPLEMENTATION_VERIFICATION)
json_write(SOURCE_DIR / 'unit_development_controls.json', UNIT_IMPLEMENTATION_VERIFICATION)
json_write(SOURCE_DIR / 'aicset_format_requirements.json', AICSET_FORMAT_REQUIREMENTS)
json_write(SOURCE_DIR / 'previous_run_audit.json', PREVIOUS_RUN_AUDIT)
json_write(SOURCE_DIR / 'source_sha256.json', {n:hashlib.sha256(s.encode()).hexdigest() for n,s in IMPLEMENTATION_SOURCE.items()})
json_write(RUN / 'local_development_controls.json', LOCAL_IMPLEMENTATION_VERIFICATION)
archived_table = legacy_reporting(ARCHIVED_SNAPSHOT, RUN, CONFIG['bootstrap_replicates'], CONFIG['seed'])
print('Archived reporting complete; no historical raw pipeline was reproduced.')

DIAGNOSTICS = RUN / 'diagnostics'; DIAGNOSTICS.mkdir(exist_ok=True)
cohort_results = []; run_errors = []
expected_datasets = ['openlab_2024', 'z24_ems', 'z24_pdt']
existing_roots = {name:Path(root) for name,root in CONFIG['data_roots'].items() if Path(root).is_dir()}
missing_roots = {name:str(root) for name,root in CONFIG['data_roots'].items() if not Path(root).is_dir()}
json_write(DIAGNOSTICS / 'missing_roots.json', missing_roots)
if missing_roots: print('Missing input roots:', missing_roots)
inventory = pd.DataFrame(); candidate = pd.DataFrame(); validated = None
try:
    if existing_roots:
        inventory = inventory_inputs(existing_roots, DIAGNOSTICS)
        candidate = make_manifest_template(inventory)
        json_write(DIAGNOSTICS / 'z24_pdt_schema_audit.json',candidate.attrs.get('z24_pdt_schema_audit',{}))
        candidate.to_csv(DIAGNOSTICS / 'acquisitions_TEMPLATE_ALL_CANDIDATES.csv', index=False)
        (DIAGNOSTICS / 'manifest_field_guide.txt').write_text(
            'One row = one synchronous acquisition. Confirm dataset, record_id, acquisition_id, group_id, '
            'timestamp_utc, state_label, scenario/excitation/setup_id for PDT, path, format, fs_hz, '
            'and explicit channel/data selectors. candidate_* fields are hints, never confirmed metadata. '
            'Use common physical reference channels across all PDT layouts. Never invent timestamps or climate values.\n'
            'Copy only completed confirmed acquisitions into the configured acquisitions.csv. '
            'Do not import every candidate: inventory also contains metadata/non-signal files.\n', encoding='utf-8')
    manifest_path = Path(CONFIG['manifest_file'])
    if manifest_path.exists():
        manifest_input = pd.read_csv(manifest_path, dtype=object, keep_default_na=False)
        print('Using explicit manifest:', manifest_path)
    elif not candidate.empty:
        verified = candidate.metadata_status.astype(str).str.startswith('VERIFIED') & candidate.dataset.ne('')
        manifest_input = candidate.loc[verified].copy()
        manifest_input.to_csv(DIAGNOSTICS / 'acquisitions_VERIFIED_AUTO.csv', index=False)
        print(f'No explicit manifest. Verified automatic OpenLab/PDT rows: {len(manifest_input)}. Other rows require metadata.')
    else:
        manifest_input = pd.DataFrame()
    if not manifest_input.empty:
        validated = validate_manifest(manifest_input, CONFIG['data_roots'])
        if Path(CONFIG['covariates_file']).exists():
            validated = attach_covariates(validated, CONFIG['covariates_file'], CONFIG['covariate_join_tolerance_seconds'])
        validated = canonical_metadata(validated)
        partitioned=[]
        for dataset,part in validated.groupby('dataset',sort=False):
            try:
                partitioned.append(assign_partitions(part,CONFIG))
            except Exception as exc:
                part=part.copy();part['partition']='UNASSIGNED_INCOMPLETE_GROUPS'
                partitioned.append(part)
                json_write(DIAGNOSTICS/(dataset+'_partition_preflight.json'),{'error':repr(exc)})
        validated=pd.concat(partitioned).sort_index()
        export_documented_openlab_context(validated,DIAGNOSTICS)
        if CONFIG.get('auto_openlab_temperature',True) and validated.dataset.eq('openlab_2024').any():
            if 'cov_temperature' in validated.columns:
                print('Retaining the explicitly supplied temperature covariate.')
                json_write(DIAGNOSTICS/'temperature_source.json',{'source':'EXPLICIT_MANIFEST_OR_COVARIATE_TABLE','automatic_overwrite':False})
            else:
                try:
                    validated,temperature_audit=attach_openlab_2024_temperature(validated,CONFIG['data_roots']['openlab'],DIAGNOSTICS,
                        tolerance_seconds=CONFIG.get('temperature_join_tolerance_seconds',300),
                        require_complete=CONFIG.get('require_complete_temperature_matches',False))
                    print('Measured OpenLab air-temperature metadata joined; see the coverage audit.')
                except Exception as exc:
                    json_write(DIAGNOSTICS/'temperature_metadata_failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
                    print('Temperature metadata unavailable:',repr(exc),'; continuing with the other methods.')
        validated.to_csv(RUN / 'validated_acquisitions.csv', index=False)
        if CONFIG['run_new_analysis']:
            for dataset in expected_datasets:
                subset = validated[validated.dataset.eq(dataset)].copy()
                if subset.empty: continue
                try:
                    print('Starting new analysis:', dataset, '/', len(subset), 'acquisitions', flush=True)
                    cohort_results.append(compute_cohort(dataset, subset, CONFIG, RUN))
                except Exception as exc:
                    error = {'dataset':dataset,'status':'FAILED_NEW_ANALYSIS','error':repr(exc),'traceback':traceback.format_exc()}
                    run_errors.append(error); json_write(DIAGNOSTICS / (dataset+'_failure.json'),error)
                    print(dataset, 'incomplete:', repr(exc))
        else: print('New raw analysis disabled in configuration; archived reporting only.')
    else: print('No confirmed acquisitions available. Complete metadata using diagnostics before raw analysis.')
except ManifestValidationError as exc:
    exc.errors.to_csv(DIAGNOSTICS / 'manifest_validation_errors.csv', index=False)
    run_errors.append({'stage':'manifest','error':str(exc)})
    print('Manifest rejected. See manifest_validation_errors.csv; no metadata were guessed.')
except Exception as exc:
    run_errors.append({'stage':'preflight','error':repr(exc),'traceback':traceback.format_exc()})
    json_write(DIAGNOSTICS / 'preflight_failure.json',run_errors[-1])
    print('Preflight incomplete:', repr(exc))

json_write(RUN / 'cohort_results.json',cohort_results)
json_write(DIAGNOSTICS / 'run_errors.json',run_errors)
present = {item['dataset']:item for item in cohort_results}
missing_cohorts = [ds for ds in expected_datasets if ds not in present]
json_write(DIAGNOSTICS / 'cohorts_not_completed.json',missing_cohorts)
print('Cohorts with derived results:', list(present))
print('Cohorts without completed results:', missing_cohorts)


review_rows=[]
def review_row(request,status,evidence,limitation):
    review_rows.append({'reviewer_request':request,'status':status,'evidence':evidence,'remaining_limit':limitation})
pdt=present.get('z24_pdt',{})
pdt_methods=set(pdt.get('methods',[]))
damage_ready=pdt.get('progressive_damage_test_groups',0)>0 and pdt.get('reference_test_groups',0)>0 and 'protected_nnls' in pdt_methods
review_row('R1_known_structural_alterations','NEW_DESCRIPTIVE_EVIDENCE' if damage_ready else 'INCOMPLETE',
    'new_analysis/z24_pdt/pdt_progressive_damage_phase.csv; pdt_raw_vs_protected_by_scenario.csv; pdt_excitation_mismatch_summary.csv' if damage_ready else 'Provide verified PDT progressive-damage acquisitions from scenarios 09--17, reference08 and source01 AVT/FVT pairs.',
    'Retrospective heterogeneous states; one calibration scenario cannot support finite nominal 95% decisions. '+('Missing scenarios: '+str(pdt['missing_official_scenarios']) if pdt else 'PDT not analysed; scenario completeness unavailable.'))
chronological=[ds for ds in ['openlab_2024','z24_ems'] if ds in present and 'protected_nnls' in present[ds].get('methods',[])]
review_row('R2_forward_group_holdout_and_paired_statistics','CALCULATED' if chronological else 'INCOMPLETE',
    '; '.join(chronological)+' / physical_group_partitions.csv, reference_coverage.csv, paired_group_comparisons.csv',
    'No artificial increase in physical group counts; sequential dependence and small samples limit inference.')
baseline_details={ds:sorted(({'baseline_pca','frequency_ratio'}|({'temperature_ridge'} if ds!='z24_pdt' else set()))-set(item.get('methods',[]))) for ds,item in present.items()}
baseline_complete={'openlab_2024','z24_pdt'}<=set(present) and all(not values for values in baseline_details.values())
review_row('R3_established_comparators','CALCULATED_FOR_AVAILABLE_COHORTS' if baseline_complete else 'PARTIAL_OR_INCOMPLETE',
    json.dumps(baseline_details)+'; fitted_models/ and unavailable_methods.json in each computed cohort.',
    'Temperature baseline applies to environmental cohorts. PDT temperature cross-validation is not estimable with one fitting group; no extra evaluation data enter fitting. Peak ratios are not verified modal invariants.')
review_row('R4_raw_pipeline_and_fitted_matrices','NEW_IMPLEMENTATION_PROVIDED' if present else 'CODE_PROVIDED_NOT_YET_RUN',
    'source/*.py; validated_acquisitions.csv; raw_source_hashes.json; fitted_models/*.npz and *.json.',
    'This does not recover the unavailable historical fitting code. Raw inputs remain on MARWAN and are not redistributed.')
review_row('R5_language_notation_and_captions','MANUSCRIPT_STEP_PENDING',
    'Notebook produces separate labelled tables and figures for subsequent Word revision.',
    'The submitted Word article has not been changed by this notebook.')
review_row('R6_learning_and_scope','METHOD_SCOPE_EXPLICIT',
    'Training-only estimation of reference, spectral dictionary, PCA components and Ridge predictors; model metadata exported.',
    'Constrained dictionary estimation and regression; no neural network or new AI performance claim.')
review_table=pd.DataFrame(review_rows)
review_table.to_csv(RUN / 'reviewer_request_status.csv',index=False)
RUNTIME.update({'finished_utc':datetime.now(timezone.utc).isoformat(),
    'new_scientific_results_available':bool(present),
    'new_analysis_complete_for_all_cohorts':not missing_cohorts and all(x['status']=='COMPLETED_NEW_ANALYSIS' for x in cohort_results),
    'cohort_statuses':cohort_results,'missing_cohorts':missing_cohorts,'errors':run_errors,
    'physical_frequency_review_pending':[x['dataset'] for x in cohort_results if x.get('f1_physical_identity_review_required',True)],
    'manual_scientific_review_required':True,
    'historical_pipeline_reproduced':False})
json_write(RUN / 'run_manifest.json',RUNTIME)
(RUN / 'README_results.md').write_text(
    '# AICSET 240 revision outputs\n\n'
    'archived_reporting/ contains reporting-only reanalysis of prior public outputs.\n'
    'new_analysis/ contains newly computed evidence, never presumed identical to historical outputs.\n'
    'Review reviewer_request_status.csv and unavailable_methods.json before drafting claims.\n'
    'Reference frequency selected by heuristic requires physical review using fitting data.\n'
    'PDT nominal infinite thresholds mean abstention, not perfect coverage. The separate calibration-maximum rule is descriptive.\n'
    'PDT scenario decisions, progressive-damage phase09--17, excitation mismatch and sampling balance remain separate descriptive outputs.\n'
    'PDT temperature cross-validation is not estimable from one fitting scenario; environmental comparisons require measured covariates.\n'
    'Do not tune configuration on evaluation outcomes or interpret score contraction as detection improvement.\n'
    'Raw acquisitions and spectral cache are excluded. Derived features and fitted transformations are included.\n'
    'source/ contains the exact new implementation; configuration_frozen_before_computation.json records parameters.\n',encoding='utf-8')
publication_manifest=export_publication_outputs(RUN,CONFIG)
json_write(RUN/'publication_export_status.json',publication_manifest)
diagnostic_zip=package_outputs(RUN,'diagnostics.zip',diagnostics_only=True)
result_zip=package_outputs(RUN,'results_to_share.zip')
print(review_table.to_string(index=False))
print('\nRESULTS TO RETURN:', result_zip)
print('DIAGNOSTICS IF METADATA ARE MISSING:', diagnostic_zip)
print('PHYSICAL FREQUENCY REVIEW STILL PENDING:', RUNTIME['physical_frequency_review_pending'])
print('No raw datasets were modified. Inspect incomplete requests before revising the paper.')
try:
    from IPython.display import display, FileLink
    display(FileLink(str(result_zip)))
    display(FileLink(str(diagnostic_zip)))
except ImportError:
    pass
