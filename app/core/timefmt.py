def choose_time_unit(duration_s: float) -> tuple[float, str]:
    """Pick a time unit matching the signal duration. Returns (multiplier from seconds, unit label)."""
    if duration_s < 1:
        return 1000.0, "ms"
    if duration_s < 120:
        return 1.0, "s"
    if duration_s < 7200:
        return 1 / 60, "min"
    return 1 / 3600, "h"
