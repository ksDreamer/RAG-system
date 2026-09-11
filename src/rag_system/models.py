"""Small, explicit contracts shared by retrieval, providers and the API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    quote: str = Field(min_length=8, max_length=1800)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1800)
    evidence: list[Evidence] = Field(min_length=1, max_length=4)


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim] = Field(max_length=8)


class Source(BaseModel):
    id: str
    document_id: str
    name: str
    page: int
    section: str
    start: int
    end: int
    text: str
    score: float = 0


class Answer(BaseModel):
    status: Literal["answered", "abstained"]
    mode: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    reason: str = ""
    cached: bool = False
    elapsed_ms: float = 0
    retrieval: str = "bm25"
    reranker: str = "none"
