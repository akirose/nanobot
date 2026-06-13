"""Compatibility shim for the external Mattermost channel plugin.

Mattermost is distributed as the ``nanobot-channel-mattermost`` package.
This module intentionally does not define a BaseChannel subclass, so built-in
channel discovery cannot shadow the external ``mattermost`` entry point.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {"MattermostChannel", "MattermostConfig", "MattermostDMConfig"}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    try:
        plugin = import_module("nanobot_channel_mattermost")
    except ModuleNotFoundError as exc:
        if exc.name != "nanobot_channel_mattermost":
            raise
        raise ImportError(
            "Mattermost channel has moved to the external "
            "'nanobot-channel-mattermost' package. Install it to use "
            "channels.mattermost."
        ) from exc
    return getattr(plugin, name)


__all__ = sorted(_EXPORTS)
