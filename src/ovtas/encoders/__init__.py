"""
Vision-Language Model (VLM) dual-encoder backends.

    ENCODERS registry:
        "mock"   -> MockEncoder     (dependency-free, for tests/CI)
        "clip"   -> ClipEncoder     (needs: pip install -e ".[vlm]")
        "siglip" -> SiglipEncoder   (needs: pip install -e ".[vlm]")

Add a new backbone by subclassing BaseVLMEncoder and decorating it
with @ENCODERS.register("your_name") -- no other file needs to change.
"""

from ovtas.encoders.base import BaseVLMEncoder
from ovtas.encoders.mock_encoder import MockEncoder
from ovtas.encoders.registry import ENCODERS

__all__ = ["BaseVLMEncoder", "MockEncoder", "ENCODERS"]

# ClipEncoder / SiglipEncoder are only importable (and only register
# themselves) when the optional 'vlm' extra (torch, transformers /
# open_clip) is installed. We attempt the import so that, when the
# extra *is* present, `ovtas.encoders.ENCODERS` already contains
# "clip" and "siglip" without any extra user action. If the extra is
# missing, we silently skip -- `ENCODERS.get("clip")` will then raise
# a clear RegistryError instead of an opaque ImportError at import
# time.
try:
    from ovtas.encoders.clip_encoder import ClipEncoder  # noqa: F401

    __all__.append("ClipEncoder")
except ImportError:
    pass

try:
    from ovtas.encoders.siglip_encoder import SiglipEncoder  # noqa: F401

    __all__.append("SiglipEncoder")
except ImportError:
    pass

try:
    from ovtas.encoders.openclip_encoder import OpenClipEncoder  # noqa: F401

    __all__.append("OpenClipEncoder")
except ImportError:
    pass
