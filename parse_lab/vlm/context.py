"""Request-scoped VLM parse configuration."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from parse_lab.types import VlmParseConfig

_vlm_config_var: ContextVar[VlmParseConfig | None] = ContextVar("vlm_parse_config", default=None)


def get_vlm_config() -> VlmParseConfig | None:
    return _vlm_config_var.get()


def set_vlm_config(cfg: VlmParseConfig | None) -> None:
    _vlm_config_var.set(cfg)


@contextmanager
def vlm_config_scope(cfg: VlmParseConfig | None) -> Iterator[None]:
    token = _vlm_config_var.set(cfg)
    try:
        yield
    finally:
        _vlm_config_var.reset(token)
