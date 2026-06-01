import json
from pathlib import Path

import torch

from mavjepa.losses import jepa_loss, last_token_hidden
from mavjepa.trainer_mv import MVJEPADataset


def test_last_token_hidden_shape():
    hidden = torch.randn(2, 5, 7)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    out = last_token_hidden(hidden, mask)
    assert out.shape == (2, 7)
    assert torch.equal(out[0], hidden[0, 2])
    assert torch.equal(out[1], hidden[1, 4])


def test_jepa_losses_are_scalar_and_finite():
    src = torch.randn(3, 8)
    tgt = torch.randn(3, 8)
    for loss_type in ["cosine", "mse", "l2"]:
        loss = jepa_loss(src, tgt, loss_type=loss_type)
        assert loss.shape == ()
        assert torch.isfinite(loss)


def test_detach_target_blocks_target_grad():
    src = torch.randn(2, 4, requires_grad=True)
    tgt = torch.randn(2, 4, requires_grad=True)
    loss = jepa_loss(src, tgt, detach_target=True)
    loss.backward()
    assert src.grad is not None
    assert tgt.grad is None


class TinyTokenizer:
    chat_template = None

    def __call__(self, text, truncation=True, max_length=32, padding=False, return_tensors="pt"):
        ids = list(range(1, min(len(str(text)), max_length) + 1)) or [1]
        if padding == "max_length":
            ids = ids + [0] * (max_length - len(ids))
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": (tensor != 0).long()}


def test_mv_dataset_reads_jsonl_with_unicode_line_separator(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    record = {
        "id": "x",
        "messages": [
            {"role": "system", "content": "Answer."},
            {"role": "user", "content": "contains\u2028separator"},
            {"role": "assistant", "content": "ok"},
        ],
        "views": {},
        "edges": [],
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    dataset = MVJEPADataset(path, TinyTokenizer(), max_length=32, view_max_length=16, model_name="tiny")

    assert len(dataset) == 1
    assert dataset.records[0]["messages"][1]["content"] == "contains\u2028separator"
