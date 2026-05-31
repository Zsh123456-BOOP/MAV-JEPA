import math

from mavjepa.edge_sampler import EdgeSampler


EDGES = [
    {"name": "low", "src": "Q", "tgt": "R", "quality": 1.0},
    {"name": "high", "src": "R", "tgt": "A", "quality": 1.0},
    {"name": "quality", "src": "Q", "tgt": "A", "quality": 2.0},
]


def test_empty_edge_list_returns_empty_selection():
    sampler = EdgeSampler(mode="adaptive", edge_budget=1)
    selected, probs = sampler.sample([])
    assert selected == []
    assert probs == {}


def test_sampler_respects_edge_budget():
    sampler = EdgeSampler(mode="random", edge_budget=2, seed=7)
    selected, _ = sampler.sample(EDGES)
    assert len(selected) == 2


def test_budget_larger_than_edges_selects_all():
    sampler = EdgeSampler(mode="adaptive", edge_budget=10)
    selected, probs = sampler.sample(EDGES)
    assert [edge["name"] for edge in selected] == [edge["name"] for edge in EDGES]
    assert set(probs) == {edge["name"] for edge in EDGES}


def test_no_edge_starvation_from_min_probability():
    sampler = EdgeSampler(mode="adaptive", edge_budget=1, p_min=0.05)
    sampler.update_loss("high", 100.0)
    sampler.update_loss("low", 0.0)
    sampler.update_loss("quality", 0.0)
    probs = sampler.probabilities(EDGES)
    assert set(probs) == {edge["name"] for edge in EDGES}
    assert all(value > 0.0 for value in probs.values())
    assert math.isclose(sum(probs.values()), 1.0)


def test_adaptive_sampler_prefers_high_loss_high_quality_edges():
    sampler = EdgeSampler(mode="adaptive", edge_budget=1, p_min=0.01)
    sampler.update_loss("high", 25.0)
    sampler.update_loss("low", 1.0)
    sampler.update_loss("quality", 4.0)
    probs = sampler.probabilities(EDGES)
    assert probs["high"] > probs["low"]
    assert probs["quality"] > probs["low"]


def test_zero_or_nan_scores_fallback_to_uniform():
    sampler = EdgeSampler(mode="adaptive", edge_budget=1, p_min=0.0)
    sampler.ema_loss = {"low": float("nan"), "high": float("nan"), "quality": float("nan")}
    probs = sampler.probabilities(EDGES)
    assert all(math.isclose(value, 1 / 3) for value in probs.values())


def test_repeated_nan_temporarily_blacklists_edge():
    sampler = EdgeSampler(mode="none", edge_budget=10, nan_blacklist_threshold=2, blacklist_steps=5)
    sampler.update_loss("high", float("nan"))
    sampler.update_loss("high", float("nan"))
    selected, probs = sampler.sample(EDGES)
    assert "high" not in {edge["name"] for edge in selected}
    assert "high" not in probs
