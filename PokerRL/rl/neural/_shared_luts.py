import torch

_CACHE_ATTR = "_f32_lut_cache"


def get_shared_lut(env_bldr, device, lut_name):
    """
    Returns ``env_bldr.lut_holder.<lut_name>`` as a float32 tensor on ``device``,
    cached on the shared lut_holder so every net in the process aliases one copy.

    Why this exists: the source LUTs are int8 of shape
    ``(RANGE_SIZE, (N_SUITS + N_RANKS) * N_HOLE_CARDS)``. For Omaha,
    RANGE_SIZE = C(52, 4) = 270,725 and the row width is 68, so each float32
    conversion is ~70MB -- ~140MB per network once both LUTs are counted.

    Every net used to build its own copy. Deep CFR/SD-CFR retains one network per
    player per CFR iteration in the Chief's StrategyBuffer, so that cost compounded
    at ~295MB per iteration: 12.5GB by iteration 44, which crossed the training
    service's MemoryHigh and throttled the process into a stall that produced zero
    completed iterations overnight. Exported eval agents hit 20.2GB for the same
    reason. The LUTs are plain attributes rather than registered buffers, so they
    never appear in state_dict() -- which is why a net is 409KB on disk but ~140MB
    live, and why the growth was so hard to attribute.

    Upstream (EricSteinberger/PokerRL) this was harmless: Hold'em's RANGE_SIZE is
    C(52, 2) = 1,326 with row width 34, making the same per-net copy ~360KB. The
    Omaha fork scaled RANGE_SIZE by 204x without revisiting it.

    Sharing is safe because both tables are read-only in the forward pass. Callers
    index them (``lut[range_idxs]``), and advanced indexing allocates a fresh
    tensor, so the in-place writes that follow land on that copy, never on the LUT
    itself.
    """
    lut_holder = env_bldr.lut_holder

    cache = getattr(lut_holder, _CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(lut_holder, _CACHE_ATTR, cache)

    key = (lut_name, str(device))
    if key not in cache:
        cache[key] = torch.from_numpy(getattr(lut_holder, lut_name)).to(device=device, dtype=torch.float32)

    return cache[key]
