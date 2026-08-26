"""Learned wound-segmentation backends.

These backends implement the same :class:`~sorbed.segmentation.base.WoundSegmenter`
protocol as the weight-free classical path, so they are swapped in purely by
configuration (``SORBED_SEGMENTATION_BACKEND=onnx``). They are kept in a
sub-package because they carry heavier runtime dependencies (ONNX Runtime) and
require downloaded weights, whereas the classical backend always runs offline.
"""

from __future__ import annotations

from sorbed.segmentation.backends.onnx_backend import OnnxSegmenter

__all__ = ["OnnxSegmenter"]
