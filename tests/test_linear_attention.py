"""Correctness checks against the explicit quadratic kernel computation."""
import torch
import torch.nn.functional as F

from models.linear_attention import LinearSelfAttention


def test_matches_explicit_kernel_and_gradients():
    torch.manual_seed(5)
    model = LinearSelfAttention(d_model=8, n_heads=2).double()
    x = torch.randn(2, 7, 8, dtype=torch.double, requires_grad=True)
    got = model(x)
    qkv = model.qkv(x).reshape(2, 7, 3, 2, 4).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = F.elu(q) + 1, F.elu(k) + 1
    weights = q @ k.transpose(-1, -2)
    expected = model.out(((weights @ v) / weights.sum(-1, keepdim=True).clamp_min(model.eps))
                         .transpose(1, 2).reshape(2, 7, 8))
    assert torch.allclose(got, expected, atol=1e-10, rtol=1e-10)
    grad_got = torch.autograd.grad(got.square().sum(), (x, *model.parameters()), retain_graph=True)
    grad_expected = torch.autograd.grad(expected.square().sum(), (x, *model.parameters()))
    for a, b in zip(grad_got, grad_expected):
        assert torch.allclose(a, b, atol=1e-9, rtol=1e-9)


def test_invalid_dimensions():
    try:
        LinearSelfAttention(7, 2)
    except ValueError:
        return
    raise AssertionError('non-divisible head dimension accepted')


if __name__ == '__main__':
    test_matches_explicit_kernel_and_gradients()
    test_invalid_dimensions()
    print('linear attention tests passed')
