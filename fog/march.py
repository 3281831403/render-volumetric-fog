"""Deterministic volumetric-fog ray-marching exercise.

All geometric quantities use world units.  Fog density and extinction use
inverse world units; one march step contributes an optical depth of
``density * step_len``.  Unless a function says otherwise, numeric contracts
use an absolute tolerance of ``1e-6``.

Every public function in this module is an intentional stub.  Implement the
contract in its docstring without changing the signatures; the tests are
fixed and must not be edited.
"""

from __future__ import annotations


class FogError(ValueError):
    """Raised when a fog configuration, ray, or sample is invalid."""


def ray_box(
    origin: tuple[float, float, float],
    dir: tuple[float, float, float],
    box: dict,
) -> tuple[float, float]:
    """Clip a camera ray against an axis-aligned fog box.

    ``origin`` is the point on the camera's near plane in world units and
    ``dir`` is a unit-length direction in world space.  ``box`` has the form
    ``{"min": (x, y, z), "max": (x, y, z)}``.

    Return ``(t_enter, t_exit)`` as world-unit distances from ``origin``
    along ``dir`` with ``0 <= t_enter < t_exit``.  The interval must be
    clipped at the near plane, so intersections with ``t < 0`` are rejected.
    A ray that misses the box, or a box entirely behind the near plane,
    returns ``(-1.0, -1.0)``.

    Absolute tolerance: ``1e-6``.
    """
    raise NotImplementedError


def step_count(t0: float, t1: float, step: float) -> int:
    """Return the number of uniform steps covering ``[t0, t1]``.

    ``t0``, ``t1``, and ``step`` are world-unit distances; ``step`` is the
    requested step length and must be positive.  Return
    ``max(1, ceil((t1 - t0) / step))`` so that a partial final segment is
    always retained.  A zero-length interval still needs one step.

    Raise ``FogError`` when ``step <= 0`` or ``t1 < t0``.
    Absolute tolerance: ``1e-6``.
    """
    raise NotImplementedError


def sample_positions(t0: float, t1: float, steps: int, jitter: float) -> list[float]:
    """Return deterministic jittered sample positions along a ray.

    ``t0``, ``t1``, and the returned positions are world-unit distances.
    The interval is split into ``steps`` equal segments of length
    ``step_len = (t1 - t0) / steps``.  Sample ``i`` is

    ``t0 + (i + jitter * u_i) * step_len``

    where ``u_i`` comes from the fixed golden-ratio sequence
    ``frac((i + 1) * 0.6180339887498949)``.  The jitter fraction is therefore
    always inside the segment; no random-number source is used.  With
    ``jitter == 0`` samples sit at segment starts, and with ``jitter == 1``
    they approach segment ends.

    Raise ``FogError`` when ``steps <= 0``, ``t1 < t0``, or ``jitter`` is
    outside ``[0, 1]``.  Absolute tolerance: ``1e-6``.
    """
    raise NotImplementedError


def density(pos: tuple[float, float, float], config: dict) -> float:
    """Return fog density (the extinction coefficient) at a world point.

    ``pos`` is ``(x, y, z)`` in world units.  ``config`` provides
    ``density`` (base extinction coefficient at ``y = 0``, inverse world
    units), ``height_falloff`` (inverse world units), ``noise_amplitude``
    (inverse world units), and ``noise_frequency`` (radians per world unit).
    The contract is::

        base  = max(0.0, config["density"])
                * exp(-config["height_falloff"] * max(0.0, y))
        noise = config["noise_amplitude"]
                * sin(config["noise_frequency"] * (x + 2*y + 3*z))
        return max(0.0, base + noise)

    The result is the non-negative extinction coefficient in inverse world
    units; a negative intermediate value is clamped to zero.  Absolute
    tolerance: ``1e-6``.
    """
    raise NotImplementedError


def extinction(density: float, step_len: float) -> float:
    """Return the Beer-Lambert optical depth for one march step.

    ``density`` is the extinction coefficient in inverse world units and
    ``step_len`` is the world-unit length of the step.  The result is the
    dimensionless optical depth ``max(0.0, density) * step_len``, so the
    contribution scales with the step length.  The corresponding
    per-step transmittance is ``exp(-result)``.

    Raise ``FogError`` when ``step_len < 0``.  Absolute tolerance: ``1e-6``.
    """
    raise NotImplementedError


def integrate(samples: list[dict], config: dict) -> dict:
    """Integrate transmittance and in-scattering over ordered samples.

    ``samples`` is a sequence of ``{"pos": (x, y, z), "step_len": length}``
    mappings in world units.  ``config`` supplies the density inputs plus
    ``phase_g`` and ``mu`` (the cosine between light and view directions),
    with ``-1 < phase_g < 1`` and ``-1 <= mu <= 1``.

    For each sample ``i``::

        tau_i   = extinction(density(pos_i, config), step_len_i)
        T_i     = exp(-tau_i)
        phase_i = clamp((1 - g*g) / (1 + g*g - 2*g*mu)**1.5, 0.0, 1.0)

    with ``g = phase_g``.  Accumulate
    ``transmittance = product(T_i)`` and
    ``inscatter = sum(T_before_i * (1 - T_i) * phase_i)``.
    Attenuation must be applied before adding each step's in-scattering;
    the loop must not be a plain sum of densities.

    Return ``{"transmittance": float, "inscatter": float}`` satisfying
    energy conservation: ``0 <= transmittance <= 1`` and
    ``0 <= inscatter <= 1 - transmittance``.  The isotropic phase
    (``phase_g == 0``) gives ``inscatter == 1 - transmittance``.

    Empty samples or invalid configuration raise ``FogError``.  World
    units; absolute tolerance: ``1e-6``.
    """
    raise NotImplementedError


def validate_config(config: dict, steps: int) -> list[str]:
    """Return human-readable configuration issues; empty means valid.

    ``steps`` is the proposed march count.  The returned strings, in this
    order when present, are:

    * ``"density must be non-negative"`` when ``config["density"] < 0``;
    * ``"step must be positive"`` when ``config["step"] <= 0``;
    * ``"steps must be positive"`` when ``steps <= 0``;
    * ``"too many steps"`` when ``steps > config.get("max_steps", 4096)``;
    * ``"step is too large"`` when
      ``config["step"] > config.get("max_step", 1.0)``;
    * ``"jitter must be in [0, 1]"`` when jitter is outside that range;
    * ``"phase_g must be in (-1, 1)"`` when ``phase_g`` is outside it;
    * ``"mu must be in [-1, 1]"`` when ``mu`` is outside that range.

    The function only reports problems; it does not mutate ``config`` and
    does not raise for the conditions above.  World units; absolute
    tolerance: ``1e-6``.
    """
    raise NotImplementedError


def fog_batch(rays: list[dict], config: dict) -> dict:
    """March a batch of cameras transactionally.

    Each ray is a camera mapping ``{"origin", "dir", "near", "far"}``.
    ``origin`` is in world units, ``dir`` is unit length, and ``near`` /
    ``far`` are distances along ``dir`` with ``0 <= near < far``.  ``config``
    must contain ``box`` (the fog volume) and ``step``; it may also contain
    the density, phase, jitter, ``max_steps``, and ``max_step`` keys.

    For each ray, clip ``[near, far]`` against ``box`` using
    ``ray_box(origin + dir*near, dir, box)``, count steps, build jittered
    samples, and integrate them.  A ray that misses the fog volume
    contributes transmittance ``1.0`` and inscatter ``0.0``.

    Return ``{"transmittance": [...], "inscatter": [...]}`` in exactly the
    input order.  Validate the whole batch before committing any result: if
    any ray or the config is invalid, raise ``FogError`` and return no
    partial result (all-or-nothing rollback).  Every ray must have its own
    accumulator; results must never leak from one ray to the next.  World
    units; absolute tolerance: ``1e-6``.
    """
    raise NotImplementedError
