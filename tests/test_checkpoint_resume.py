from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import utils


class ResumeModel(nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.backbone = nn.Linear(width, width)
        self.head = nn.Linear(width, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


def _args(tmp_path, resume=""):
    return SimpleNamespace(
        output_dir=str(tmp_path),
        resume=str(resume),
        auto_resume=False,
        eval=False,
        start_epoch=0,
        model_ema=False,
        save_ckpt_num=2,
        save_ckpt_freq=1,
    )


def test_strict_resume_rejects_architecture_mismatch(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"model": {"unknown.weight": torch.randn(2, 2)}}, path)
    with pytest.raises(RuntimeError):
        utils.load_model_for_resume(_args(tmp_path, path), ResumeModel(), strict=True)


def test_strict_resume_restores_exact_model_state(tmp_path):
    source = ResumeModel()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.uniform_(-0.2, 0.2)
    path = tmp_path / "resume.pth"
    torch.save({"model": source.state_dict(), "epoch": 2}, path)

    target = ResumeModel()
    checkpoint = utils.load_model_for_resume(_args(tmp_path, path), target, strict=True)
    assert checkpoint["epoch"] == 2
    for left, right in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(left, right)


def test_restore_optimizer_state_advances_start_epoch(tmp_path):
    model = ResumeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 3,
        "training_state": {"best_epoch": 2, "best_val_metric": 77.0},
    }
    args = _args(tmp_path)
    state = utils.restore_optimizer_state(args, checkpoint, optimizer, loss_scaler=None)
    assert args.start_epoch == 4
    assert state["best_epoch"] == 2
    assert state["best_val_metric"] == 77.0
