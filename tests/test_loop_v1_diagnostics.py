"""CPU arithmetic audit; no model or torch import required."""
import copy
import unittest

from experiments.diagnose_loop_v1 import (
    bootstrap_document_ratio, diagnose, quantile, summarize, validate_record, verify_same_records)


def row(name='a', native=(1.,3.,2.,4.), candidate=(.5,4.,1.,4.)):
    return dict(document_id=name,tokens=len(native),native_ce=list(native),
        candidate_ce=list(candidate),native_nll=sum(native)/len(native),
        candidate_nll=sum(candidate)/len(candidate),
        valid_position_indices=list(range(len(native))),continue_mask=[True]*len(native))


class PriorOracleTests(unittest.TestCase):
    def test_oracle_and_forced_are_different(self):
        result=summarize([row()],repetitions=20)
        self.assertAlmostEqual(result['native_nll'],2.5)
        self.assertAlmostEqual(result['forced_nll'],2.375)
        self.assertAlmostEqual(result['oracle_nll'],2.125)
        self.assertAlmostEqual(result['oracle_gain_nats'],.375)
        self.assertEqual(result['oracle_continue_tokens'],2)
        self.assertEqual(result['beneficial_fraction'],.5)
        self.assertEqual(result['harmful_fraction'],.25)
        self.assertAlmostEqual(result['positive_contribution_nats'],.375)
        self.assertAlmostEqual(result['negative_contribution_nats'],-.25)
        self.assertAlmostEqual(result['oracle_at_most_call_fraction']['0.25']['oracle_gain_nats'],.25)

    def test_oracle_never_spends_on_harmful_tokens(self):
        result=summarize([row(native=(1.,2.),candidate=(3.,4.))],repetitions=20)
        self.assertEqual(result['oracle_nll'],result['native_nll'])
        self.assertEqual(result['oracle_at_most_call_fraction']['1.0']['oracle_calls'],0)

    def test_token_weighted_cluster_bootstrap(self):
        result=summarize([row(),row('b',native=(10.,),candidate=(0.,))],repetitions=20)
        self.assertAlmostEqual(result['forced_gain_nats'],10.5/5)
        self.assertEqual(bootstrap_document_ratio([(10,5.)],repetitions=20),[.5,.5])
        self.assertEqual(bootstrap_document_ratio([(10,5.),(2,-1.)],repetitions=50),
                         bootstrap_document_ratio([(10,5.),(2,-1.)],repetitions=50))

    def test_tampering_lengths_and_duplicate_docs_rejected(self):
        for key,value in [('tokens',5),('native_nll',1.),('native_ce',[float('nan')]*4),
                          ('valid_position_indices',[0,0,2,3]),('continued_supervised_tokens',3)]:
            bad=row();bad[key]=value
            with self.assertRaises(ValueError):validate_record(bad)
        with self.assertRaises(ValueError):summarize([row(),row()],repetitions=20)

    def test_duplicate_pointer_verification(self):
        left=[row(),row('b')]
        verify_same_records(left,list(reversed(copy.deepcopy(left))),'same')
        right=copy.deepcopy(left);right[0]['candidate_ce'][0]=0.
        with self.assertRaises(ValueError):verify_same_records(left,right,'tamper')
        with self.assertRaises(ValueError):verify_same_records(left,right[:1],'missing')

    def test_quantile(self):
        self.assertEqual(quantile([0.,10.],.25),2.5)
        with self.assertRaises(ValueError):quantile([], .5)

    def test_actual_historical_evidence_recomputes_without_duplicate_counting(self):
        result=diagnose(repetitions=20)
        self.assertEqual(result['raw_evidence_records'],680)
        self.assertEqual(result['canonical_checkpoint_record_count'],576)
        r2=result['branches']['r2']
        self.assertEqual(r2['selected_epoch'],1)
        self.assertEqual(r2['updates'],1480)
        dev=r2['fixed_checkpoints'][1]['development']
        self.assertEqual((dev['documents'],dev['tokens']),(8,1809))
        self.assertAlmostEqual(dev['forced_nll'],2.2567136344820504)
        self.assertLess(r2['fixed_checkpoints'][7]['development']['oracle_nll'],dev['oracle_nll'])
        self.assertEqual(result['gate_choice']['mode'],'stop')
        self.assertTrue(all(r['gain_nats']<0 for r in result['gate_candidates']))


if __name__=='__main__':
    unittest.main()
