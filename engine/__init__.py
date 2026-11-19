"""Fast C Catan engine, rule-for-rule identical to the patched catanatron. See engine/README.md.

The ctypes bindings (engine.cengine) load lazily, so `engine.obs_np` (pure numpy, shippable in a
submission) can be imported without building or loading the shared library.
"""
_LAZY = {"Engine", "VecEnv", "N_ACTIONS", "OBS_DIM", "STATE_SIZE", "encode_states", "lib", "build"}


def __getattr__(name):
    if name in _LAZY:
        from engine import cengine
        return getattr(cengine, name)
    raise AttributeError(name)
