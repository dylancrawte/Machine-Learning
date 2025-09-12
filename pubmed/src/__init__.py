"""PubMed relevance pipeline (pandas + scikit-learn)."""

__all__ = ["PubmedPipelineSetup", "PubmedPipelineUpdate"]


def __getattr__(name: str):
    if name in __all__:
        from .pipeline import PubmedPipelineSetup, PubmedPipelineUpdate

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
