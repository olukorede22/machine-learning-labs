"""pytest lab05/src -q   (torch tests are skipped automatically if torch is missing)"""
import math
import numpy as np
import pytest
from engine import Value, vsum, topological_order, graph_summary


def f_engine(a, b, c):
    return vsum([((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c)])


def f_numpy(a, b, c):
    return float(np.sum(np.maximum((a * b + c) ** 2, 0)))


def num_grad(f, arrs, eps=1e-6):
    grads = []
    for k in range(len(arrs)):
        g = np.zeros_like(arrs[k])
        for i in range(len(arrs[k])):
            hi = [x.copy() for x in arrs]; lo = [x.copy() for x in arrs]
            hi[k][i] += eps; lo[k][i] -= eps
            g[i] = (f(*hi) - f(*lo)) / (2 * eps)
        grads.append(g)
    return grads


def test_required_expression_matches_numerical():
    rng = np.random.default_rng(42)
    arrs = [rng.normal(size=8) for _ in range(3)]
    vals = [[Value(x) for x in a] for a in arrs]
    out = f_engine(*vals)
    out.backward()
    assert out.data == pytest.approx(f_numpy(*arrs))
    ng = num_grad(f_numpy, arrs)
    for vs, g in zip(vals, ng):
        assert max(abs(v.grad - gi) for v, gi in zip(vs, g)) < 1e-6


def test_forward_builds_graph():
    a, b = Value(2.0), Value(3.0)
    c = (a * b + 1).relu()
    assert c._parents and graph_summary(c)["nodes"] >= 5
    assert a in topological_order(c) and topological_order(c)[-1] is c


def test_gradients_accumulate_not_overwrite():
    a = Value(3.0)
    y = a * a + a * 2 + a          # a is used 4 times -> dy/da = 2a + 2 + 1 = 9
    y.backward()
    assert a.grad == pytest.approx(9.0)


def test_pow_relu_and_inactive_relu():
    a = Value(-2.0)
    y = (a ** 3).relu()            # negative -> relu inactive -> grad 0
    y.backward()
    assert y.data == 0.0 and a.grad == 0.0
    b = Value(2.0)
    z = (b ** 3).relu(); z.backward()
    assert b.grad == pytest.approx(12.0)


def test_sigmoid_log_derivatives():
    x = Value(0.3)
    y = (x.sigmoid()).log(); y.backward()
    s = 1 / (1 + math.exp(-0.3))
    assert x.grad == pytest.approx(1 - s, abs=1e-9)        # d/dx log(sigmoid(x)) = 1 - sigmoid(x)


def test_deep_chain_no_recursion_error():
    x = Value(1.0); y = x
    for _ in range(20000):
        y = y + 1e-6
    y.backward()
    assert x.grad == pytest.approx(1.0)


def test_matches_torch_if_available():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    arrs = [rng.normal(size=6) for _ in range(3)]
    ts = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in arrs]
    ((ts[0] * ts[1] + ts[2]) ** 2).relu().sum().backward()
    vals = [[Value(x) for x in a] for a in arrs]
    f_engine(*vals).backward()
    for vs, t in zip(vals, ts):
        assert max(abs(v.grad - g) for v, g in zip(vs, t.grad.numpy())) < 1e-6
