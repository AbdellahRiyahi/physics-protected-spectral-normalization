"""Publication controls use artificial fixtures, never bridge observations."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

SPEC=importlib.util.spec_from_file_location('aicset_publication',Path(__file__).with_name('publication.py'))
PUB=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(PUB)


class PublicationControls(unittest.TestCase):
    def test_compact_pdt_phase_table_preserves_counts_and_missing_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);folder=base/'z24_pdt';folder.mkdir();out=base/'publication';out.mkdir()
            source=folder/'pdt_progressive_damage_phase.csv'
            pd.DataFrame([
                {'method':'raw','threshold_rule':'DESCRIPTIVE_CALIBRATION_MAXIMUM','n_reference_groups':1,
                 'n_altered_groups':9,'true_positive_groups':8,'false_positive_groups':0,'sensitivity':8/9,'auc':.9},
                {'method':'protected_nnls','threshold_rule':'DESCRIPTIVE_CALIBRATION_MAXIMUM','n_reference_groups':0,
                 'n_altered_groups':9,'true_positive_groups':7,'false_positive_groups':np.nan,'sensitivity':7/9,'auc':np.nan},
            ]).to_csv(source,index=False)
            original=source.read_bytes();catalogue=[]
            PUB._pub_pdt(folder,out,catalogue,122/25.4,300)
            table=pd.read_csv(out/'tables/z24_pdt_progressive_damage_phase.csv')
            self.assertEqual(table.shape,(2,5))
            self.assertIn('8/9',table.iloc[0].astype(str).tolist())
            self.assertTrue(pd.isna(table['Descriptive AUC'].iloc[1]))
            self.assertEqual(source.read_bytes(),original)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.run=Path(self.tmp.name)
        self.folder=self.run/'new_analysis/openlab_2024';self.folder.mkdir(parents=True)
        self.groups=[]
        for method in ['raw','protected_nnls','frequency_ratio']:
            for i,partition in enumerate(['fit','calibration','test']):
                self.groups.append({'dataset':'openlab_2024','method':method,'group_id':f'2024-01-0{i+1}',
                    'score':1.+i*.2 if method!='protected_nnls' else .8+i*.2,'partition':partition,
                    'state_label':'reference','feature_family':'RATIOS' if method=='frequency_ratio' else 'SPECTRA'})
        pd.DataFrame(self.groups).to_csv(self.folder/'group_scores.csv',index=False)
        self.thresholds={m:{'threshold':2.,'finite_supported':True} for m in ['raw','protected_nnls','frequency_ratio']}
        (self.folder/'thresholds.json').write_text(json.dumps(self.thresholds))
        pd.DataFrame([{'method':m,'m':24,'n_groups':1,'covered_groups':1,'alert_groups':0,'coverage':1.,
            'wilson_95_low':.2,'wilson_95_high':1.,'status':'OK'} for m in self.thresholds]).to_csv(self.folder/'reference_coverage.csv',index=False)
        pd.DataFrame([{'method':'protected_nnls','comparison_scope':'SAME_SPECTRAL_FEATURE_FAMILY','n_groups':1,
            'median_paired_relative_reduction':.12,'median_paired_relative_reduction_ci95_low':.05,
            'median_paired_relative_reduction_ci95_high':.2,'raw_alert_only_groups':0,'corrected_alert_only_groups':0,
            'mcnemar_exact_two_sided_p':1.,'mcnemar_p_holm':1.},
            {'method':'frequency_ratio','comparison_scope':'BINARY_DECISIONS_ACROSS_DIFFERENT_FEATURE_FAMILIES','n_groups':1,
             'raw_alert_only_groups':0,'corrected_alert_only_groups':0,'mcnemar_exact_two_sided_p':1.,'mcnemar_p_holm':1.}]
        ).to_csv(self.folder/'paired_group_comparisons.csv',index=False)
        (self.run/'cohort_results.json').write_text(json.dumps([{'dataset':'openlab_2024',
            'methods':list(self.thresholds),'unavailable_methods':[{'method':'temperature_ridge','error':'Measured covariate unavailable.'}]}]))

    def tearDown(self):self.tmp.cleanup()

    def export(self):return PUB.export_publication_outputs(self.run,{'publication_style':{'raster_dpi':150}})

    def test_frozen_source_files_are_not_changed(self):
        before={p.name:p.read_bytes() for p in self.folder.iterdir() if p.is_file()}
        report=self.export();self.assertFalse(report['errors'])
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.folder.iterdir() if p.is_file()})
        self.assertFalse(report['models_refitted']);self.assertFalse(report['thresholds_changed'])

    def test_cross_family_distances_are_excluded_from_score_table(self):
        self.export();folder=self.run/'publication/tables'
        scores=pd.read_csv(folder/'openlab_2024_paired_score_reductions.csv')
        self.assertEqual(scores['Method'].tolist(),['Protected NNLS'])
        self.assertEqual(scores['Median score reduction (%)'].iloc[0],12.)
        decisions=pd.read_csv(folder/'openlab_2024_paired_alert_decisions.csv')
        self.assertIn('Spectral peak ratios',decisions.Method.tolist())

    def test_missing_cohorts_are_not_fabricated(self):
        report=self.export();table=pd.read_csv(self.run/'publication/tables/analysis_availability.csv')
        absent=table[table.Cohort.str.startswith('Z24')]
        self.assertTrue(absent.Availability.eq('Not calculated').all())
        self.assertFalse(any('z24_' in i['stem'] for i in report['items']))

    def test_caption_positions_and_british_html_language(self):
        self.export();text=(self.run/'publication/index.html').read_text()
        self.assertIn('lang="en-GB"',text)
        self.assertLess(text.index('class="caption"'),text.index('<table'))
        self.assertLess(text.index('<img '),text.index('<figcaption>'))

    def test_unsupported_threshold_does_not_become_perfect_coverage(self):
        rows=[{'method':m,'m':1,'n_groups':1,'covered_groups':np.nan,'alert_groups':np.nan,'coverage':np.nan,
            'wilson_95_low':np.nan,'wilson_95_high':np.nan,'status':'ABSTAIN'} for m in self.thresholds]
        pd.DataFrame(rows).to_csv(self.folder/'reference_coverage.csv',index=False)
        report=self.export();table=pd.read_csv(self.run/'publication/tables/openlab_2024_reference_coverage.csv')
        self.assertTrue(table['Coverage (%)'].isna().all())
        self.assertTrue(table['Covered/reference groups'].eq('Abstain').all())
        self.assertFalse(any(i['kind']=='Figure' and i['stem'].endswith('reference_coverage') for i in report['items']))

    def test_empty_run_has_no_experimental_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=PUB.export_publication_outputs(tmp)
            self.assertEqual(report['figures'],0);self.assertFalse(report['errors'])
            self.assertEqual(report['tables'],1)

    def test_invalid_style_rejected(self):
        with self.assertRaises(ValueError):
            PUB.export_publication_outputs(self.run,{'publication_style':{'figure_width_mm':0}})


if __name__=='__main__':unittest.main()
