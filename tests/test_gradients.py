import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network import NeuralNetwork
from src.layers import Linear, ReLU, SoftmaxCrossEntropy

RTOL = 1e-5
EPS = 1e-5
SEED = 42

def relative_error(analytic, numeric):
    return np.abs(analytic - numeric) / np.maximum(1e-12, np.abs(analytic) + np.abs(numeric))


def numerical_gradient(f, x, eps=EPS):
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]

        x[idx] = orig + eps
        fxp = f()

        x[idx] = orig - eps
        fxm = f()

        x[idx] = orig  # restore
        grad[idx] = (fxp - fxm) / (2 * eps)
        it.iternext()
    return grad


@pytest.fixture
def small_batch():
    rng = np.random.default_rng(SEED)
    N, D, H, C = 6, 5, 4, 3
    x = rng.standard_normal((N, D))
    y = rng.integers(0, C, size=N)
    return x, y, D, H, C


def test_linear_layer_gradients(small_batch):
    x, _, D, H, _ = small_batch
    rng = np.random.default_rng(1)
    layer = Linear(D, H, rng=rng)
    dout = rng.standard_normal((x.shape[0], H))

    out = layer.forward(x)
    dx_analytic = layer.backward(dout)
    dW_analytic, db_analytic = layer.dW, layer.db

    def loss_fn_w():
        return np.sum(layer.forward(x) * dout)

    def loss_fn_x():
        return np.sum(layer.forward(x) * dout)

    dW_numeric = numerical_gradient(loss_fn_w, layer.W)
    db_numeric = numerical_gradient(loss_fn_w, layer.b)
    dx_numeric = numerical_gradient(loss_fn_x, x)

    for name, analytic, numeric in [
        ("dW", dW_analytic, dW_numeric),
        ("db", db_analytic, db_numeric),
        ("dx", dx_analytic, dx_numeric),
    ]:
        err = relative_error(analytic, numeric).max()
        print(f"Linear layer {name}: max relative error = {err:.2e}")
        assert err < RTOL, f"Linear layer {name} gradient check FAILED (err={err:.2e})"


def test_relu_gradients(small_batch):
    x, _, D, _, _ = small_batch
    rng = np.random.default_rng(2)
    z = rng.standard_normal((x.shape[0], D))
    dout = rng.standard_normal(z.shape)

    layer = ReLU()
    layer.forward(z)
    dz_analytic = layer.backward(dout)

    def loss_fn():
        return np.sum(ReLU().forward(z) * dout)

    dz_numeric = numerical_gradient(loss_fn, z)
    err = relative_error(dz_analytic, dz_numeric).max()
    print(f"ReLU dz: max relative error = {err:.2e}")
    assert err < RTOL, f"ReLU gradient check FAILED (err={err:.2e})"


def test_softmax_cross_entropy_gradients(small_batch):
    x, y, D, _, C = small_batch
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((x.shape[0], C))

    layer = SoftmaxCrossEntropy()
    layer.forward(logits, y)
    dlogits_analytic = layer.backward()

    def loss_fn():
        loss, _ = SoftmaxCrossEntropy().forward(logits, y)
        return loss

    dlogits_numeric = numerical_gradient(loss_fn, logits)
    err = relative_error(dlogits_analytic, dlogits_numeric).max()
    print(f"SoftmaxCrossEntropy dlogits: max relative error = {err:.2e}")
    assert err < RTOL, f"SoftmaxCrossEntropy gradient check FAILED (err={err:.2e})"


def test_full_network_gradients(small_batch):
    x, y, D, H, C = small_batch
    net = NeuralNetwork(D, H, C, seed=SEED)

    loss, _ = net.forward(x, y)
    net.backward()

    def loss_fn():
        l, _ = net.forward(x, y)
        return l

    for name, (param, grad) in zip(["W1", "b1", "W2", "b2"], net.parameters()):
        numeric = numerical_gradient(loss_fn, param)
        err = relative_error(grad, numeric).max()
        print(f"Full network {name}: max relative error = {err:.2e}")
        assert err < RTOL, f"Full network {name} gradient check FAILED (err={err:.2e})"


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
