__all__ = ["MalariaDataset", "DataPipeline"]


def __getattr__(name: str):
    if name == "MalariaDataset":
        from .dataset import MalariaDataset
        return MalariaDataset
    if name == "DataPipeline":
        from .pipeline import DataPipeline
        return DataPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
