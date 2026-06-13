"""Channel test configuration."""

from __future__ import annotations

import sys
from pathlib import Path

_MATTERMOST_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "nanobot-channel-mattermost"
if str(_MATTERMOST_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_MATTERMOST_PLUGIN_DIR))
