import numpy as np

from .effect_base import Effect, EffectResult, Param
from .signal import AudioSignal, from_float, to_float


class Reverse(Effect):
    name = "Reverse"

    def apply(self, sig: AudioSignal, **values) -> EffectResult:
        out = AudioSignal(sig.samples[::-1].copy(), sig.sample_rate, sig.bits, sig.fmt, sig.name)
        return EffectResult(out)

def fade_curve(g: float, curve: str) -> float:
    if curve == "linear":
        return g
    if curve == "quadratic":
        return g * g
    if curve == "logarithmic":
        return np.log10(1 + 9*g)
    raise Exception("Nezinoma funkcija")


class Fade(Effect):
    name = "Fade in/out"
    params = [
        Param("type", "Type", default="in", choices=["in", "out"]),
        Param("curve", "Curve", default="linear", choices=["linear", "quadratic", "logarithmic"]),
        Param("duration", "Ilgis", default=500.0, minimum=1, maximum=10000000, step=1.0, decimals=0,unit="ms")
    ]

    def apply(self, sig: AudioSignal, type: str, curve: str, duration: float, **values):
        x = to_float(sig)
        t = np.arange(sig.n_samples) / sig.sample_rate
        total = t[-1] * 1000 # ms
        k = 1 / duration
        mask = np.ones(len(t)) # kiekvienam t, t = 1

        if type == "in":
            i = 0
            for sample_t in t:
                sample_t *= 1000 # i ms
                if sample_t >= duration:
                    break
                mask[i] = fade_curve(k * sample_t, curve)
                i += 1

        elif type == "out":
            i = len(t) - 1
            for sample_t in t[::-1]:
                left = total - sample_t * 1000
                if left >= duration:
                    break
                mask[i] = fade_curve(k * left, curve)
                i -= 1

        else:
            raise Exception("Nezinomas tipas")

        out = x * mask[:, None]
        return EffectResult(from_float(out, sig), intermediates=[(f"Mask ({curve})", from_float(mask, sig))])

class Echo(Effect):
    name = "Echo"
    params = [
        Param("alpha", "α", default=0.5, minimum=0.0, maximum=1.0, step=0.05, decimals=2),
        Param("delta", "Δ", default=300.0, minimum=0.0, maximum=5000.0, step=10.0, decimals=0, unit="ms"),
    ]

    def apply(self, sig: AudioSignal, alpha: float, delta: float, **values):
        x = to_float(sig)
        n = sig.n_samples
        d = int(round(delta * sig.sample_rate / 1000))

        # bendras ilgis: n + d
        s_delta = np.zeros((n + d, sig.n_channels))
        out = np.zeros((n + d, sig.n_channels))

        for i in range(n + d):
            if i < n:
                out[i] = x[i] # s
            if i >= d:
                s_delta[i] = x[i - d] # sΔ
                out[i] += alpha * s_delta[i]
        return EffectResult(from_float(out, sig), intermediates=[(f"sΔ (Δ = {delta:g} ms)", from_float(s_delta, sig))])

EFFECTS: list[Effect] = [Reverse(), Fade(), Echo()]
