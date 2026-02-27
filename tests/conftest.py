import os

from hypothesis import HealthCheck, settings

settings.register_profile("dev", max_examples=20, deadline=400)
settings.register_profile(
    "ci", max_examples=500, derandomize=True, deadline=None
)
settings.register_profile(
    "full",
    max_examples=10_000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
