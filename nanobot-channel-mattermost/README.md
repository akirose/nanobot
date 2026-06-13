# nanobot-channel-mattermost

Mattermost channel plugin for nanobot.

Install from this source checkout:

```bash
python -m pip install -e ./nanobot-channel-mattermost
```

Configure the existing `channels.mattermost` section in `~/.nanobot/config.json`:

```json
{
  "channels": {
    "mattermost": {
      "enabled": true,
      "serverUrl": "https://your-mattermost.example.com",
      "token": "YOUR_MATTERMOST_BOT_TOKEN",
      "allowFrom": ["YOUR_MATTERMOST_USER_ID"]
    }
  }
}
```

The plugin registers the `mattermost` entry point in the `nanobot.channels` group.
