from __future__ import annotations

try:
    import lightning as L
    from lightning import LightningDataModule
except ModuleNotFoundError:
    try:
        import pytorch_lightning as L
        from pytorch_lightning import LightningDataModule
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing Lightning dependency. Install 'lightning' or 'pytorch-lightning' in the active environment."
        ) from exc

__all__ = [
    "L",
    "LightningDataModule",
]