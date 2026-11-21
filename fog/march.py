"""Deterministic volumetric-fog ray-marching exercise.

All geometric quantities use world units.  Fog density and extinction use
inverse world units; one march step contributes an optical depth of
``density * step_len``.  Unless a function says otherwise, numeric contracts
use an absolute tolerance of ``1e-6``.

The public functions implement the contracts in their docstrings; the tests
are fixed and must not be edited.
"""

from __future__ import annotations

import math

_GOLDEN = 0.6180339887498949
_EPS = 1e-12


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
    t_enter = 0.0
    t_exit = math.inf

    try:
        box_min = box["min"]
        box_max = box["max"]
    except (KeyError, TypeError):
        raise FogError("box must contain 'min' and 'max'") from None

    for axis in range(3):
        origin_axis = float(origin[axis])
        dir_axis = float(dir[axis])
        lo = float(box_min[axis])
        hi = float(box_max[axis])
        if hi < lo:
            lo, hi = hi, lo

        if abs(dir_axis) < _EPS:
            # Ray is parallel to this slab: it must already be inside it.
            if origin_axis < lo or origin_axis > hi:
                return (-1.0, -1.0)
            continue

        inv = 1.0 / dir_axis
        ta = (lo - origin_axis) * inv
        tb = (hi - origin_axis) * inv
        if ta > tb:
            ta, tb = tb, ta

        if ta > t_enter:
            t_enter = ta
        if tb < t_exit:
            t_exit = tb

        if t_enter >= t_exit:
            return (-1.0, -1.0)

    # Clip at the near plane: anything at t < 0 is behind the camera.
    if t_exit < 0.0:
        return (-1.0, -1.0)
    if t_enter < 0.0:
        t_enter = 0.0

    if not (0.0 <= t_enter < t_exit):
        return (-1.0, -1.0)

    return (t_enter, t_exit)


def step_count(t0: float, t1: float, step: float) -> int:
    """Return the number of uniform steps covering ``[t0, t1]``.

    ``t0``, ``t1``, and ``step`` are world-unit distances; ``step`` is the
    requested step length and must be positive.  Return
    ``max(1, ceil((t1 - t0) / step))`` so that a partial final segment is
    always retained.  A zero-length interval still needs one step.

    Raise ``FogError`` when ``step <= 0`` or ``t1 < t0``.
    Absolute tolerance: ``1e-6``.
    """
    if step <= 0.0:
        raise FogError("step must be positive")
    if t1 < t0:
        raise FogError("t1 must not be less than t0")
    span = t1 - t0
    return max(1, int(math.ceil(span / step)))


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
    if steps <= 0:
        raise FogError("steps must be positive")
    if t1 < t0:
        raise FogError("t1 must not be less than t0")
    if not (0.0 <= jitter <= 1.0):
        raise FogError("jitter must be in [0, 1]")

    step_len = (t1 - t0) / steps
    positions = []
    for i in range(steps):
        raw = (i + 1) * _GOLDEN
        u_i = raw - math.floor(raw)
        positions.append(t0 + (i + jitter * u_i) * step_len)
    return positions


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
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

    base = max(0.0, float(config["density"])) * math.exp(
        -float(config["height_falloff"]) * max(0.0, y)
    )
    noise = float(config["noise_amplitude"]) * math.sin(
        float(config["noise_frequency"]) * (x + 2.0 * y + 3.0 * z)
    )
    return max(0.0, base + noise)


def extinction(density: float, step_len: float) -> float:
    """Return the Beer-Lambert optical depth for one march step.

    ``density`` is the extinction coefficient in inverse world units and
    ``step_len`` is the world-unit length of the step.  The result is the
    dimensionless optical depth ``max(0.0, density) * step_len``, so the
    contribution scales with the step length.  The corresponding
    per-step transmittance is ``exp(-result)``.

    Raise ``FogError`` when ``step_len < 0``.  Absolute tolerance: ``1e-6``.
    """
    if step_len < 0.0:
        raise FogError("step_len must be non-negative")
    return max(0.0, float(density)) * float(step_len)


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
    if not samples:
        raise FogError("samples must not be empty")

    try:
        g = float(config["phase_g"])
        mu = float(config["mu"])
    except (KeyError, TypeError):
        raise FogError("config must provide phase_g and mu") from None

    if not (-1.0 < g < 1.0):
        raise FogError("phase_g must be in (-1, 1)")
    if not (-1.0 <= mu <= 1.0):
        raise FogError("mu must be in [-1, 1]")

    denominator = 1.0 + g * g - 2.0 * g * mu
    phase = (1.0 - g * g) / (denominator ** 1.5)
    phase = min(1.0, max(0.0, phase))

    transmittance = 1.0
    inscatter = 0.0
    for sample in samples:
        try:
            pos = sample["pos"]
            step_len = float(sample["step_len"])
        except (KeyError, TypeError):
            raise FogError("each sample needs 'pos' and 'step_len'") from None

        tau = extinction(density(pos, config), step_len)
        step_transmittance = math.exp(-tau)

        # Apply the accumulated attenuation before adding in-scattering.
        inscatter += transmittance * (1.0 - step_transmittance) * phase
        transmittance *= step_transmittance

    transmittance = min(1.0, max(0.0, transmittance))
    inscatter = min(1.0 - transmittance, max(0.0, inscatter))
    return {"transmittance": transmittance, "inscatter": inscatter}


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
    issues = []

    if float(config.get("density", 0.0)) < 0.0:
        issues.append("density must be non-negative")
    if float(config.get("step", 0.0)) <= 0.0:
        issues.append("step must be positive")
    if steps <= 0:
        issues.append("steps must be positive")
    if steps > config.get("max_steps", 4096):
        issues.append("too many steps")
    if float(config.get("step", 0.0)) > config.get("max_step", 1.0):
        issues.append("step is too large")

    jitter = config.get("jitter")
    if jitter is not None and not (0.0 <= float(jitter) <= 1.0):
        issues.append("jitter must be in [0, 1]")

    phase_g = config.get("phase_g")
    if phase_g is not None and not (-1.0 < float(phase_g) < 1.0):
        issues.append("phase_g must be in (-1, 1)")

    mu = config.get("mu")
    if mu is not None and not (-1.0 <= float(mu) <= 1.0):
        issues.append("mu must be in [-1, 1]")

    return issues


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
    try:
        box = config["box"]
        step = float(config["step"])
    except (KeyError, TypeError):
        raise FogError("config must contain 'box' and 'step'") from None

    jitter = float(config.get("jitter", 0.5))

    # First pass: validate every ray and the configuration.  Nothing is
    # committed until the whole batch has been proven valid.
    plans: list[dict] = []
    for ray in rays:
        try:
            origin = tuple(float(v) for v in ray["origin"])
            direction = tuple(float(v) for v in ray["dir"])
            near = float(ray["near"])
            far = float(ray["far"])
        except (KeyError, TypeError):
            raise FogError("ray must provide origin, dir, near and far") from None

        if len(origin) != 3 or len(direction) != 3:
            raise FogError("origin and dir must be 3-dimensional")
        if not (near >= 0.0 and far > near):
            raise FogError("ray must satisfy 0 <= near < far")

        near_origin = tuple(
            origin[axis] + direction[axis] * near for axis in range(3)
        )
        enter, leave = ray_box(near_origin, direction, box)

        if enter < 0.0:
            # The ray misses the fog volume; still validate the config.
            issues = validate_config(config, 1)
            if issues:
                raise FogError("; ".join(issues))
            plans.append({"hit": False})
            continue

        t0 = enter
        t1 = min(leave, far - near)
        if t1 <= t0:
            issues = validate_config(config, 1)
            if issues:
                raise FogError("; ".join(issues))
            plans.append({"hit": False})
            continue

        steps = step_count(t0, t1, step)
        issues = validate_config(config, steps)
        if issues:
            raise FogError("; ".join(issues))

        positions = sample_positions(t0, t1, steps, jitter)
        step_len = (t1 - t0) / steps
        plans.append(
            {
                "hit": True,
                "near_origin": near_origin,
                "direction": direction,
                "positions": positions,
                "step_len": step_len,
            }
        )

    # Second pass: integrate each ray with its own accumulators.
    transmittance: list[float] = []
    inscatter: list[float] = []
    for plan in plans:
        if not plan["hit"]:
            transmittance.append(1.0)
            inscatter.append(0.0)
            continue

        near_origin = plan["near_origin"]
        direction = plan["direction"]
        samples = [
            {
                "pos": (
                    near_origin[0] + direction[0] * t,
                    near_origin[1] + direction[1] * t,
                    near_origin[2] + direction[2] * t,
                ),
                "step_len": plan["step_len"],
            }
            for t in plan["positions"]
        ]
        result = integrate(samples, config)
        transmittance.append(result["transmittance"])
        inscatter.append(result["inscatter"])

    return {"transmittance": transmittance, "inscatter": inscatter}
