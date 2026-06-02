from argparse import Namespace
from collections import Counter

from mavjepa.edge_sampler import EdgeSampler
from mavjepa.trainer_mv import MultiViewTrainer, split_main_weak_edges, strip_final_answer, tokenize_view


class ListTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False, **kwargs):
        tokens = str(text).split()
        if add_special_tokens:
            tokens = ["<s>"] + tokens
        return {"input_ids": list(range(1, len(tokens) + 1))}


def trainer_for_candidate_tests() -> MultiViewTrainer:
    trainer = MultiViewTrainer.__new__(MultiViewTrainer)
    trainer.args = Namespace(
        view_max_length=256,
        min_target_tokens=8,
        strip_answer_from_reasoning=False,
        disable_answer_target_edges=False,
        lambda_base=0.05,
        weak_edge_step_prob=0.03,
        weak_edge_start_step=3000,
    )
    trainer.tokenizer = ListTokenizer()
    trainer.view_max_lengths = {"Q": 256, "R": 256, "A": 64, "R_SUF": 128}
    trainer.view_truncation_sides = {"Q": "right", "R": "left"}
    trainer.config_edges = {}
    trainer.allowed_edges = None
    trainer.lambda_mode = "fixed"
    trainer.filtered_edge_counts = Counter()
    trainer.last_mv_loss_info = {}
    return trainer


def test_answer_gating_happens_before_sampling():
    trainer = trainer_for_candidate_tests()
    views = {"Q": "what is two plus two", "R": " ".join(["reason"] * 32), "A": "4"}
    edges = [
        {"name": "Q_to_R", "src": "Q", "tgt": "R"},
        {"name": "Q_to_A", "src": "Q", "tgt": "A"},
        {"name": "R_to_A", "src": "R", "tgt": "A"},
    ]

    candidates = trainer.candidate_edges(views, edges, step=1000)

    assert {edge["name"] for edge in candidates} == {"Q_to_R"}
    assert trainer.filtered_edge_counts["Q_to_A:short_target"] == 1
    assert trainer.filtered_edge_counts["R_to_A:short_target"] == 1


def test_edge_budget_one_cannot_sample_unusable_short_answer_edge():
    trainer = trainer_for_candidate_tests()
    views = {"Q": "what is two plus two", "R": " ".join(["reason"] * 32), "A": "4"}
    edges = [
        {"name": "Q_to_R", "src": "Q", "tgt": "R", "prior": 0.1},
        {"name": "Q_to_A", "src": "Q", "tgt": "A", "prior": 0.9},
    ]
    candidates = trainer.candidate_edges(views, edges, step=1000)
    sampler = EdgeSampler(mode="prior", edge_budget=1, seed=7)

    selected, _ = sampler.sample(candidates)

    assert [edge["name"] for edge in selected] == ["Q_to_R"]


def test_per_edge_lambda_overrides_global_lambda():
    trainer = trainer_for_candidate_tests()
    edge = {"name": "QRPRE_to_RSUF", "lambda": 0.03}

    assert trainer.lambda_for_edge(edge) == 0.03


def test_strip_answer_from_reasoning_does_not_clear_useful_reasoning():
    text = "We compute 2 + 2 = 4. Therefore, the answer is 4."

    cleaned = strip_final_answer(text)

    assert cleaned.strip()
    assert "2 + 2" in cleaned


def test_weak_only_edges_do_not_enter_main_sampler():
    candidates = [
        {"name": "Q_to_R_MASKANS", "src": "Q", "tgt": "R_MASKANS"},
        {"name": "QR_MASKANS_to_A_STMT", "src": "QR_MASKANS", "tgt": "A_STMT", "weak_only": True},
    ]

    main_edges, weak_edges = split_main_weak_edges(candidates)

    assert [edge["name"] for edge in main_edges] == ["Q_to_R_MASKANS"]
    assert [edge["name"] for edge in weak_edges] == ["QR_MASKANS_to_A_STMT"]


def test_per_view_right_truncation_keeps_prefix():
    tokenizer = ListTokenizer()
    tokenized = tokenize_view(tokenizer, "Question: keep " + " ".join(f"tail{i}" for i in range(20)), 5, "right")

    assert tokenized["input_ids"][0, 1].item() == 2
