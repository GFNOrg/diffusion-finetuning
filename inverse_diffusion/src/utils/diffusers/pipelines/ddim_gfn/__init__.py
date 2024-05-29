from typing import TYPE_CHECKING

from diffusers.utils import DIFFUSERS_SLOW_IMPORT, _LazyModule


_import_structure = {"pipeline_ddim_gfn": ["DDIMGFNPipeline"]}

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_ddim_gfn import DDIMPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
