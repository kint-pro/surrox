from pydantic import BaseModel, ConfigDict

from surrox.analysis.summary import Summary


class AnalysisResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: Summary
