"""Fast CPU checks for the proposed training intervention (no data or GPU)."""
import math
import unittest

import torch

from weakflow.time_sampling import off_time_score, sample_imf_times


class IntervalMixtureTest(unittest.TestCase):
    def test_score_is_derivative_of_marginal_log_density(self):
        cfg = dict(time_mean=-0.4, time_std=1.)
        r = torch.tensor([1e-6, .02, .15, .4, .9], dtype=torch.float64)
        t = torch.tensor([.03, .25, .5, .9, .999999], dtype=torch.float64,
                         requires_grad=True)
        def p(s):
            a = (torch.logit(s) - cfg["time_mean"]) / cfg["time_std"]
            return torch.exp(-a.square()/2) / (math.sqrt(2*math.pi)*cfg["time_std"]*s*(1-s))
        for rho in (0., .25, 1.):
            q = 2*((1-rho)*p(r)*p(t) + rho)
            expected, = torch.autograd.grad(q.log().sum(), t)
            got = off_time_score(r, t, dict(cfg, off_uniform_prob=rho))
            torch.testing.assert_close(got, expected, rtol=2e-10, atol=1e-10)

    def test_paired_sampling_and_long_interval_counts(self):
        cfg = dict(time_mean=-.4, time_std=1.)
        n = 400_000
        torch.manual_seed(311)
        t0, r0 = sample_imf_times(n, cfg, "cpu")
        after_a = torch.get_rng_state()
        torch.manual_seed(311)
        t1, r1 = sample_imf_times(n, dict(cfg, off_uniform_prob=.25), "cpu",
                                   torch.Generator().manual_seed(19))
        self.assertTrue(torch.equal(after_a, torch.get_rng_state()))
        off = n // 2
        self.assertTrue(torch.equal(t0[:off], t1[:off]))
        self.assertTrue(torch.equal(r0[:off], r1[:off]))
        self.assertTrue(bool(((r1 > 0) & (r1 <= t1) & (t1 < 1)).all()))
        h = (t1-r1)[off:]
        # Values for mu=-0.4, sigma=1: .75 P_LN(h>=a)+.25(1-a)^2
        for threshold, target in ((.5, .1259692), (.9, .0025108)):
            observed = (h >= threshold).double().mean().item()
            se = math.sqrt(target*(1-target) / len(h))
            self.assertLess(abs(observed - target), 6*se)
            print(f"h >= {threshold}: observed={observed:.6%}, expected={target:.6%}")


if __name__ == "__main__":
    unittest.main()
