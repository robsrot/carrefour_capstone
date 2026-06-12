"""Autoencoder feature learning for nonlinear customer structure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.utils import file_fingerprint, frame_to_numpy, numeric_feature_columns, should_use_cache, write_artifact_metadata


def build_autoencoder_latents(
    feature_path: str | Path,
    latent_size: int,
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Train a compact autoencoder and persist its latent representation."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = (
        Path(output_path)
        if output_path
        else cfg.data_processed / f"{cfg.get('autoencoder.output_prefix')}_{latent_size}.parquet"
    )
    model_path = cfg.models / f"autoencoder_latent_{latent_size}.pt"
    cache_metadata = {
        "stage": "autoencoder_latents",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "latent_size": latent_size,
        "autoencoder": cfg.get("autoencoder", {}),
        "random_seed": cfg.random_seed,
    }
    if (
        should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        and model_path.exists()
    ):
        return output

    import torch
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(cfg.random_seed)
    df = pl.read_parquet(feature_path).sort("cliente")
    feature_cols = numeric_feature_columns(df)
    X = frame_to_numpy(df, feature_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    input_dim = X_scaled.shape[1]
    hidden_dim = max(latent_size * int(cfg.get("autoencoder.hidden_multiplier", 2)), min(128, input_dim))

    class CustomerAutoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    model = CustomerAutoencoder(input_dim, hidden_dim, latent_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("autoencoder.learning_rate", 0.001)))
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(torch.from_numpy(X_scaled))
    generator = torch.Generator()
    generator.manual_seed(cfg.random_seed)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("autoencoder.batch_size", 1024)),
        shuffle=True,
        generator=generator,
    )

    best_loss = float("inf")
    stale_epochs = 0
    patience = int(cfg.get("autoencoder.patience", 5))
    for _ in range(int(cfg.get("autoencoder.epochs", 25))):
        model.train()
        epoch_losses = []
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(epoch_losses))
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.eval()
    with torch.no_grad():
        latent = model.encoder(torch.from_numpy(X_scaled)).cpu().numpy().astype(np.float32)

    out = {"cliente": df["cliente"].to_list()}
    for idx in range(latent.shape[1]):
        out[f"latent_{idx:03d}"] = latent[:, idx]
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out).write_parquet(output)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_size": latent_size,
            "feature_columns": feature_cols,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "best_loss": best_loss,
        },
        model_path,
    )
    write_artifact_metadata(output, cache_metadata)
    write_artifact_metadata(model_path, cache_metadata)
    return output
