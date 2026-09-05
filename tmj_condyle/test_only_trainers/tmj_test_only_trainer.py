"""A deliberately tiny, clearly named nnU-Net v2 trainer for test-only runs."""

from __future__ import annotations

import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_TMJTestOnly_1epoch(nnUNetTrainer):
    """Run real nnU-Net code for one epoch and two train iterations.

    This class is intentionally not suitable for scientific training. The
    environment guard prevents accidental use outside the isolated runner.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cpu"),
    ):
        if os.environ.get("TMJ_TEST_ONLY") != "1":
            raise RuntimeError(
                "nnUNetTrainer_TMJTestOnly_1epoch is restricted to TMJ_TEST_ONLY=1"
            )
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1
        self.save_every = 1
