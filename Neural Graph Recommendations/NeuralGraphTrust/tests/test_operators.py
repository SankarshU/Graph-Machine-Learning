"""The attention operator and the propagation identities stated in the paper."""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from conftest import graph_inputs
from src.graphtrust import edge_table, fast_norm_adj
from src.models import AttentiveLightGCN, LightGCN
from src.models._attention import MaskedNeighborAttention
from src.utils import build_norm_adj


def naive_attention(attn, E, neighbors, mask):
    """s_vj = v_a^T tanh(W_a [e_v || e_j] + b), masked softmax, weighted sum (concatenation form)."""
    N, K = neighbors.shape
    e_j = E[neighbors]                                          # (N, K, d)
    cat = torch.cat([E.unsqueeze(1).expand(-1, K, -1), e_j], dim=-1)
    s = attn.v_a(torch.tanh(attn.W_a(cat))).squeeze(-1)
    s = s.masked_fill(mask < 0.5, float("-inf"))
    empty = mask.sum(dim=1, keepdim=True) < 0.5
    s = torch.where(empty.expand_as(s), torch.zeros_like(s), s)
    alpha = F.softmax(s, dim=1) * (1.0 - empty.double())
    return (alpha.unsqueeze(-1) * e_j).sum(dim=1)


def test_split_attention_matches_concatenation():
    torch.manual_seed(0)
    N, K, d = 25, 6, 8
    attn = MaskedNeighborAttention(d, attn_hidden=5).double()
    with torch.no_grad():
        attn.W_a.bias.normal_()
    neighbors = torch.randint(0, N, (N, K))
    mask = (torch.rand(N, K) > 0.3).double()
    mask[3] = 0.0                                              # a node without neighbours
    E1 = torch.randn(N, d, dtype=torch.float64, requires_grad=True)
    E2 = E1.detach().clone().requires_grad_(True)
    out1 = attn(E1, neighbors, mask)
    g1 = torch.autograd.grad(out1.square().sum(), [E1, *attn.parameters()])
    out2 = naive_attention(attn, E2, neighbors, mask)
    g2 = torch.autograd.grad(out2.square().sum(), [E2, *attn.parameters()])
    assert torch.allclose(out1, out2, atol=1e-6, rtol=0)
    assert out1[3].abs().max() == 0
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-6, rtol=0)


def test_fast_norm_adj_matches_build_norm_adj():
    R = sp.random(17, 23, density=0.2, random_state=2, format="lil", dtype=np.float32)
    R[5, :] = 0.0                                              # an isolated user (degree 0)
    R = R.tocsr(); R.eliminate_zeros(); R.data[:] = 1.0
    a = build_norm_adj(17, 23, R).toarray()
    b = fast_norm_adj(17, 23, R).toarray()
    assert np.abs(a - b).max() < 1e-12


def test_closed_gate_recovers_lightgcn(data):
    """Prop. recovery: as gamma -> 0 the gated layer is LightGCN's."""
    norm_adj, pad = graph_inputs(data)
    torch.manual_seed(0)
    att = AttentiveLightGCN(data.n_users, data.n_items, embed_dim=16, n_layers=3, norm_adj=norm_adj, neighbor_pad=pad)
    lgcn = LightGCN(data.n_users, data.n_items, embed_dim=16, n_layers=3, norm_adj=norm_adj)
    lgcn.user_embedding.weight.data.copy_(att.user_embedding.weight.data)
    lgcn.item_embedding.weight.data.copy_(att.item_embedding.weight.data)
    with torch.no_grad():
        att.gamma.raw.fill_(-60.0)                             # sigmoid(-60) ~ 1e-26
        ua, ia = att.propagate()
        ul, il = lgcn.propagate()
    assert torch.allclose(ua, ul, atol=1e-6) and torch.allclose(ia, il, atol=1e-6)


def test_uniform_attention_gives_unit_reliability(data):
    """Prop. uniform attention: alpha = 1/n~_v on the valid slots, so r = n~_v alpha = 1 on every scored edge."""
    norm_adj, pad = graph_inputs(data, K=8)
    att = AttentiveLightGCN(data.n_users, data.n_items, embed_dim=16, n_layers=2, norm_adj=norm_adj, neighbor_pad=pad)
    with torch.no_grad():
        att.attn.v_a.weight.zero_()                            # constant scores -> uniform softmax
    att.eval()
    t = edge_table(att, data, torch.device("cpu"))
    assert len(t) == int(pad.valid_mask[: data.n_users].sum())
    assert np.abs(t.r - 1.0).max() < 1e-6
    assert (t.label == 1).all()
