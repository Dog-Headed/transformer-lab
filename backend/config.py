"""Validate model, compute and request budgets before allocating tensors."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_PARAMETERS = 3_000_000
MAX_TEXT_CHARACTERS = 2_000_000
MAX_ATTENTION_ELEMENTS = 2_000_000
MAX_RUNTIME_SECONDS = 180
MAX_TASKS = 8
MAX_CHECKPOINTS = 20
MAX_STORAGE_BYTES = 200 * 1024 * 1024


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    tokenizer: Literal["character", "word"] = "character"
    dataset: Literal["shakespeare", "stories", "stories_identity"] = "shakespeare"
    layers: int = Field(2, ge=1, le=4)
    heads: int = Field(4, ge=1, le=8)
    hidden_size: int = Field(64, ge=16, le=128)
    sequence_length: int = Field(32, ge=8, le=128)
    learning_rate: float = Field(0.003, ge=0.00001, le=0.02, allow_inf_nan=False)
    batch_size: int = Field(16, ge=1, le=32)
    max_steps: int = Field(200, ge=1, le=2000)
    epochs: int | None = Field(None, ge=1, le=20)
    eval_interval: int = Field(20, ge=1, le=100)
    validation_ratio: float = Field(0.1, ge=0.1, le=0.3, allow_inf_nan=False)
    seed: int = Field(42, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def check_budget(self):
        if self.hidden_size % self.heads:
            raise ValueError("hidden_size 必须能被 heads 整除")
        attention = self.batch_size * self.layers * self.heads * self.sequence_length**2
        if attention > MAX_ATTENTION_ELEMENTS:
            raise ValueError("注意力矩阵预算超限，请降低 batch、layers、heads 或上下文长度")
        if self.batch_size * self.layers * self.sequence_length * self.hidden_size**2 > 100_000_000:
            raise ValueError("单步计算预算超限")
        return self


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: TrainConfig = Field(default_factory=TrainConfig)
    text: str | None = Field(None, min_length=256, max_length=MAX_TEXT_CHARACTERS)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=512)
    temperature: float = Field(0.8, ge=0.1, le=2.0, allow_inf_nan=False)
    top_k: int = Field(20, ge=0, le=50)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    layer: int = Field(0, ge=0, le=3)
    head: int = Field(0, ge=0, le=7)
    context_tokens: list[Annotated[str, Field(min_length=1, max_length=512)]] | None = Field(None, min_length=1, max_length=128)


class GenerateRequest(PredictRequest):
    max_new_tokens: int = Field(80, ge=1, le=128)
    stop_after_sentence: bool = False


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = Field(400, ge=1, le=2000)


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: Literal["auto", "cpu", "cuda"] = "auto"
