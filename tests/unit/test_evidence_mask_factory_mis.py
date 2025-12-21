import unittest
import torch
import numpy as np

from bpropy.clique_cover import clique_cover
from bpropy.co.generators.mis import random_mis
from bpropy.mrf.factor_graph import FactorGraph
from bpropy.mrf.evidence_factory import EvidenceMaskFactory

def slow_oracle_mask(fg, evidence: dict[int,int], *, device, dtype=torch.bool):
    mask = torch.ones(fg.number_sufficient_statistics(), dtype=dtype, device=device)
    c = 0
    for F in fg.iter_factors():
        scope = fg.factor_variables(F)
        nF = len(scope)
        pos = {v: scope.index(v) for v in evidence if v in scope}

        for idx in range(1 << nF):
            xF = [(idx >> k) & 1 for k in range(nF)]
            ok = all(xF[pos[v]] == evidence[v] for v in pos)
            if not ok:
                mask[c + idx] = False
        c += 1 << nF
    return mask

class TestEvidenceMaskFactoryMIS(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_pb = random_mis()
        self.random_pb_cc = random_mis()
        self.random_pb_cc.C = clique_cover(self.random_pb_cc.g)
    def test_evidence_mask_factory_mis_1(self):
        fg = FactorGraph(self.random_pb)
        theta = torch.arange(fg.number_sufficient_statistics(), dtype=torch.float32)
        evidence_factory = EvidenceMaskFactory(fg, theta=theta, neg_inf=-10.)

        # apply evidence
        evidence = {0: 1, 1: 0}
        evidence_factory.add_evidence(evidence)
        # create masked theta
        masked_theta = evidence_factory.apply(theta)
        # loop over all factor and variable assignments
        c = 0
        for F in fg.iter_factors():
            scope = fg.factor_variables(F)
            nF = len(scope)

            pos = {v: scope.index(v) for v in evidence.keys() if v in scope}
            for idx in range(1 << nF):
                xF = np.array([(idx >> k) & 1 for k in range(nF)], dtype=np.uint8)
                expected_allowed = all(xF[pos[v]] == evidence[v] for v in pos)
                if expected_allowed:
                    self.assertTrue(evidence_factory.mask[c+idx].item())
                    self.assertEqual(masked_theta[c+idx].item(), theta[c+idx].item())
                else:
                    self.assertFalse(evidence_factory.mask[c+idx].item())
                    self.assertEqual(masked_theta[c+idx].item(), -10.0)
            c += 1<<nF
    def test_evidence_mask_factory_mis_cc(self):
        fg = FactorGraph(self.random_pb_cc)
        theta = torch.arange(fg.number_sufficient_statistics(), dtype=torch.float32)
        evidence_factory = EvidenceMaskFactory(fg, theta=theta, neg_inf=-10.)

        # apply evidence
        evidence = {0: 1, 1: 0}
        evidence_factory.add_evidence(evidence)
        # create masked theta
        masked_theta = evidence_factory.apply(theta)
        # loop over all factor and variable assignments
        c = 0
        for F in fg.iter_factors():
            scope = fg.factor_variables(F)
            nF = len(scope)

            pos = {v: scope.index(v) for v in evidence.keys() if v in scope}
            for idx in range(1 << nF):
                xF = np.array([(idx >> k) & 1 for k in range(nF)], dtype=np.uint8)
                expected_allowed = all(xF[pos[v]] == evidence[v] for v in pos)
                if expected_allowed:
                    self.assertTrue(evidence_factory.mask[c+idx].item())
                    self.assertEqual(masked_theta[c+idx].item(), theta[c+idx].item())
                else:
                    self.assertFalse(evidence_factory.mask[c+idx].item())
                    self.assertEqual(masked_theta[c+idx].item(), -10.0)
            c += 1<<nF

    def test_evidence_mask_factory_incremental(self):
        fg = FactorGraph(self.random_pb)
        theta = torch.arange(fg.number_sufficient_statistics(), dtype=torch.float32)
        evidence_factory = EvidenceMaskFactory(fg, theta=theta, neg_inf=-10.)

        # apply evidence
        evidence_factory.add_evidence({0: 1, 1: 0})
        # create masked theta
        m0 = evidence_factory.mask.clone()

        # reset and apply evidence bit by bit
        evidence_factory.reset()
        evidence_factory.add_evidence({0:1})
        m1 = evidence_factory.mask.clone()
        evidence_factory.add_evidence({1:0})
        m2 = evidence_factory.mask.clone()
        self.assertTrue((m2 == m0).all())
    def test_evidence_mask_factory_contradiction(self):
        fg = FactorGraph(self.random_pb)
        theta = torch.arange(fg.number_sufficient_statistics(), dtype=torch.float32)
        evidence_factory = EvidenceMaskFactory(fg, theta=theta, neg_inf=-10.)
        evidence_factory.add_evidence({0:1})
        with self.assertRaises(ValueError):
            evidence_factory.add_evidence({0:0})
    def test_evidence_mask_factory_random_evidence(self):
        fg = FactorGraph(self.random_pb)
        theta = torch.arange(fg.number_sufficient_statistics(), dtype=torch.float32)
        neg_inf = -10.0
    
        rng = np.random.default_rng(0)
        nV = fg.number_of_variables()
    
        for _ in range(50):
            # random partial evidence: choose k vars and clamp them randomly
            k = int(rng.integers(0, min(10, nV) + 1))  # keep reasonable
            vars_ = rng.choice(nV, size=k, replace=False)
            evidence = {int(v): int(rng.integers(0, 2)) for v in vars_}
    
            emf = EvidenceMaskFactory(fg, theta=theta, neg_inf=neg_inf)
            emf.add_evidence(evidence)
    
            oracle = slow_oracle_mask(fg, evidence, device=theta.device)
            self.assertTrue(torch.equal(emf.mask, oracle))
    
            masked_theta = emf.apply(theta)
            expected_masked = torch.where(oracle, theta, torch.full_like(theta, neg_inf))
            self.assertTrue(torch.equal(masked_theta, expected_masked))


if __name__=="__main__":
    unittest.main()
