"""Contract checks; no real world accuracy claims."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from v3_adapt import frame_plan, calc_scores, choose_balanced, HOLDOUT_SLOTS

class V3AdaptationChecks(unittest.TestCase):
    def test_holdout_ids_fixed(self):
        self.assertEqual(HOLDOUT_SLOTS,('P003','P010','P012'))
    def test_nonoverlap_timestamps(self):
        values=frame_plan(283,7,10)
        self.assertEqual(len(set(values)),7)
        self.assertEqual(values,sorted(values))
    def test_metrics(self):
        x=[(10,'A','empty','empty'),(20,'B','occupied','empty'),(30,'C','occupied','occupied')]
        m=calc_scores(x)
        self.assertEqual(m['correct'],2)
        self.assertEqual(m['confusion_matrix']['occupied']['empty'],1)
    def test_balancing(self):
        rows=[({'true_status':'empty','id':f'E{i}'},None) for i in range(3)]+[({'true_status':'occupied','id':f'O{i}'},None) for i in range(7)]
        balanced=choose_balanced(rows)
        self.assertEqual(len(balanced),6)

if __name__=='__main__':unittest.main()
