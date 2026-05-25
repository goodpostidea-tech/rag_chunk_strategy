"""Vision-language model PDF parsing."""

from parse_lab.vlm.client import parse_pdf_with_vlm
from parse_lab.vlm.context import get_vlm_config, set_vlm_config, vlm_config_scope

__all__ = [
    "parse_pdf_with_vlm",
    "get_vlm_config",
    "set_vlm_config",
    "vlm_config_scope",
]
