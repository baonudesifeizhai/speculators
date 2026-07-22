from pathlib import Path

import torch

from speculators.train import dataloader


def test_explicit_validation_dataset_uses_full_disjoint_arrow_data(monkeypatch):
    datasets = []

    def fake_arrow_dataset(**kwargs):
        datasets.append(kwargs)
        return kwargs

    monkeypatch.setattr(dataloader, "ArrowDataset", fake_arrow_dataset)
    monkeypatch.setattr(
        dataloader,
        "_setup_dataloader",
        lambda dataset, *_args, **_kwargs: dataset,
    )

    train, validation = dataloader.create_train_val_loaders(
        data_path="/data/train",
        validation_data_path="/data/validation",
        train_data_ratio=0.9,
        total_seq_len=8192,
        hidden_states_dtype=torch.bfloat16,
        noise_std=0.05,
        legacy_data=False,
        hidden_states_path="/data/hidden_states",
        vllm_endpoint="http://localhost:18000/v1",
        on_missing="generate",
        on_generate="delete",
        verifier_name_or_path="target",
        request_timeout=180,
        max_retries=3,
        hidden_size=2048,
        num_target_layers=5,
        num_workers=0,
        prefetch_factor=4,
        preprocess=None,
    )

    assert train["datapath"] == "/data/train"
    assert train["split_ratio"] == 1.0
    assert train["hidden_states_path"] == Path("/data/hidden_states/train")
    assert validation["datapath"] == "/data/validation"
    assert validation["split_ratio"] == 1.0
    assert validation["hidden_states_path"] == Path("/data/hidden_states/validation")
    assert len(datasets) == 2
