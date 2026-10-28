from dataclasses import dataclass, field
from .signal import AudioSignal

@dataclass
class Param:
    key: str
    label: str
    default: float | str
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.1
    decimals: int = 3
    unit: str = ""
    choices: list[str] | None = None


@dataclass
class EffectResult:
    output: AudioSignal
    intermediates: list[tuple[str, AudioSignal]] = field(default_factory=list)  # (caption, signal)


class Effect:
    name: str = ""
    params: list[Param] = []

    def apply(self, sig: AudioSignal, **values) -> EffectResult:
        raise NotImplementedError
