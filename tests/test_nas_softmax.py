import numpy as np
import pytest

from cryptollm.fhe.nas_softmax import FheSoftmax, emulate, exact_softmax, make_plan, precision_bits


def synthetic_rows(rng, m, n=16):
    """Rows with wide row-dependent offsets and a sink-like first column (like real LMs)."""
    base = rng.normal(0, 3, (m, n)) - rng.uniform(0, 40, (m, 1))
    base[:, 0] = base.max(1) + rng.uniform(-1, 0.5, m)
    return base


@pytest.mark.parametrize("kind,rule", [("constant", "M"), ("constant", "G"), ("fitted", "G")])
def test_emulated_pipeline_is_accurate(kind, rule):
    rng = np.random.default_rng(0)
    cal, test = synthetic_rows(rng, 3000), synthetic_rows(rng, 200)
    plan = make_plan(cal, kind, rule)
    probs, ood = emulate(plan, test)
    assert ood.mean() <= 0.01
    assert np.median(precision_bits(probs[~ood], exact_softmax(test[~ood]))) > 12


def test_fitted_shift_needs_fewer_rounds_than_current_practice():
    rng = np.random.default_rng(1)
    cal = synthetic_rows(rng, 3000)
    current, fitted = make_plan(cal, "constant", "M"), make_plan(cal, "fitted", "G")
    assert fitted.k < current.k and fitted.depth() < current.depth()


def test_encrypted_pipeline_matches_emulation_and_depth_plan():
    pytest.importorskip("openfhe")
    rng = np.random.default_rng(2)
    cal, test = synthetic_rows(rng, 3000), synthetic_rows(rng, 4)
    plan = make_plan(cal, "fitted", "G")
    fhe = FheSoftmax(plan, rows_per_ct=4, secure=False)   # tiny insecure ring: correctness only
    probs, levels, _ = fhe.softmax(test)
    assert np.abs(probs - emulate(plan, test)[0]).max() < 1e-6
    assert levels <= plan.depth()
