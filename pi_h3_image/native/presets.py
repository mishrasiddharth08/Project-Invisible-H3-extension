"""The "ph3" and "h3image" UI presets, added at runtime: PresetArch gains a member per profile and the preset's options
are registered through Forge Neo's own presets.register, so they match every other preset (as in the Qwen-Image 2.1
extension). The P-H3-Image and H3-Image schedules join Forge's scheduler list the same way.
"""

from enum import Enum

from modules_forge import presets

from ..profiles import ALL, SCHEDULERS

PROFILES = ALL
PRESETS = tuple(profile.name for profile in PROFILES)


def _add_enum_member(enum_cls: type[Enum], name: str) -> Enum:
    if name in enum_cls.__members__:
        return enum_cls[name]

    value = max(m.value for m in enum_cls) + 1
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    member.__objclass__ = enum_cls
    member._sort_order_ = len(enum_cls._member_names_)

    # EnumType.__setattr__ refuses new members, so go through type
    type.__setattr__(enum_cls, name, member)
    enum_cls._member_map_[name] = member
    enum_cls._member_names_.append(name)
    enum_cls._value2member_map_[value] = member
    if hasattr(enum_cls, "_hashable_values_"):
        enum_cls._hashable_values_.append(value)
    return member


def _is_preset_option(key: str) -> bool:
    return any(key.startswith(f"{name}_") or key.endswith(f"_{name}") for name in PRESETS)


def register() -> None:
    for profile in PROFILES:
        arch = _add_enum_member(presets.PresetArch, profile.name)
        presets.SAMPLERS[arch] = profile.sampler
        presets.SCHEDULERS[arch] = profile.scheduler
        presets.STEPS[arch] = profile.steps
        presets.CFG[arch] = profile.cfg
        # a negative shift hides the Shift slider by default (Forge's own convention): a model on its own sigma
        # grid ignores it
        presets.SHIFT[arch] = profile.shift if profile.shift_used else -abs(profile.shift)

    from modules import shared

    templates: dict = {}
    presets.register(templates)
    for key, info in templates.items():
        if _is_preset_option(key) and key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)


def register_sampler() -> None:
    """The H3-Image LCM sampler as a Forge k-diffusion sampler, before the UI lists the samplers."""
    from modules import sd_samplers, sd_samplers_common, sd_samplers_kdiffusion

    from ..sampler import SAMPLER, sample_h3_image_lcm

    data = sd_samplers_common.SamplerData(
        SAMPLER, lambda model: sd_samplers_kdiffusion.KDiffusionSampler(sample_h3_image_lcm, model), ["h3_image_lcm"], {})
    sd_samplers.add_sampler(data)


def register_scheduler() -> None:
    """The models' own schedules as Forge schedules, before the UI lists the schedules."""
    from modules import sd_schedulers

    hidden = getattr(sd_schedulers.shared.opts, "hide_schedulers", ())
    for name, label, function in SCHEDULERS:
        if any(s.label == label for s in sd_schedulers.all_schedulers):
            continue
        # the model's schedule linker, for the [H3] Resolution shift (profiles.follow_grid)
        scheduler = sd_schedulers.Scheduler(name, label, function, need_inner_model=True)
        sd_schedulers.all_schedulers.append(scheduler)
        if scheduler.label not in hidden:
            sd_schedulers.schedulers.append(scheduler)
            sd_schedulers.schedulers_map[scheduler.name] = scheduler
            sd_schedulers.schedulers_map[scheduler.label] = scheduler
