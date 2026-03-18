from pydantic import BaseModel, ConfigDict

from surrox.analysis.summary import Summary


class AnalysisResult(BaseModel):
    """Result of the automatic post-optimization analysis.

    Attributes:
        summary: High-level summary including solution overview, baseline comparison,
            constraint status, surrogate quality, and extrapolation warnings.
    """

    model_config = ConfigDict(frozen=True)

    summary: Summary
