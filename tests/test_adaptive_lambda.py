import math

from mavjepa.adaptive_lambda import AdaptiveLambda


def test_lambdas_are_finite_and_clamped():
    controller = AdaptiveLambda(lambda_base=1.0, lambda_min=0.5, lambda_max=1.5, warmup_steps=0)
    controller.update_many({"large": 100.0, "small": 0.001})
    assert controller.lambda_for("large") == 1.5
    assert controller.lambda_for("small") == 0.5
    assert all(math.isfinite(value) for value in controller.lambdas().values())


def test_unseen_edges_use_lambda_base():
    controller = AdaptiveLambda(lambda_base=0.75, warmup_steps=0)
    controller.update_many({"seen": 2.0})
    assert controller.lambda_for("unseen") == 0.75


def test_warmup_uses_lambda_base():
    controller = AdaptiveLambda(lambda_base=1.0, warmup_steps=2)
    controller.update_many({"a": 10.0, "b": 1.0})
    assert controller.lambda_for("a") == 1.0
    controller.update_many({"a": 10.0, "b": 1.0})
    assert controller.lambda_for("a") != 1.0


def test_all_zero_emas_fallback_to_lambda_base():
    controller = AdaptiveLambda(lambda_base=1.25, warmup_steps=0)
    controller.update_many({"a": 0.0, "b": 0.0})
    assert controller.lambda_for("a") == 1.25
    assert controller.lambda_for("b") == 1.25
