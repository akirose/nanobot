from __future__ import annotations

import json
from pathlib import Path
import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.mattermost import MattermostChannel, MattermostConfig


class _FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.get_calls: list[dict[str, object]] = []
        self.post_calls: list[dict[str, object]] = []
        self.get_responses: list[_FakeResponse] = []
        self.post_responses: list[_FakeResponse] = []
        self.closed = False

    async def get(self, url: str, **kwargs):
        self.get_calls.append({"url": url, **kwargs})
        if not self.get_responses:
            raise AssertionError("No fake GET response queued")
        return self.get_responses.pop(0)

    async def post(self, url: str, **kwargs):
        self.post_calls.append({"url": url, **kwargs})
        if not self.post_responses:
            raise AssertionError("No fake POST response queued")
        return self.post_responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def _make_channel(**kwargs) -> MattermostChannel:
    config = MattermostConfig(
        enabled=True,
        server_url="https://mm.example.com",
        token="tok",
        allow_from=["*"],
        **kwargs,
    )
    return MattermostChannel(config, MessageBus())


def test_default_config_returns_disabled_dict() -> None:
    cfg = MattermostChannel.default_config()
    assert isinstance(cfg, dict)
    assert cfg["enabled"] is False
    assert "serverUrl" in cfg
    assert "token" in cfg


def test_init_from_dict_validates_aliases() -> None:
    channel = MattermostChannel(
        {
            "enabled": True,
            "serverUrl": "https://mm.example.com",
            "token": "tok",
            "allowFrom": ["*"],
        },
        MessageBus(),
    )
    assert channel.config.server_url == "https://mm.example.com"
    assert channel.config.allow_from == ["*"]


@pytest.mark.asyncio
async def test_send_posts_basic_message() -> None:
    channel = _make_channel()
    fake = _FakeAsyncClient()
    fake.post_responses.append(_FakeResponse({"id": "post1"}, status_code=201))
    channel._client = fake

    await channel.send(OutboundMessage(channel="mattermost", chat_id="chan1", content="hello"))

    assert len(fake.post_calls) == 1
    call = fake.post_calls[0]
    assert call["url"].endswith("/api/v4/posts")
    assert call["json"] == {"channel_id": "chan1", "message": "hello"}


@pytest.mark.asyncio
async def test_send_includes_root_id_for_thread_reply() -> None:
    channel = _make_channel()
    fake = _FakeAsyncClient()
    fake.post_responses.append(_FakeResponse({"id": "post1"}, status_code=201))
    channel._client = fake

    await channel.send(
        OutboundMessage(
            channel="mattermost",
            chat_id="chan1",
            content="reply",
            metadata={"mattermost": {"root_id": "root123", "channel_type": "O"}},
        )
    )

    assert fake.post_calls[0]["json"]["root_id"] == "root123"


@pytest.mark.asyncio
async def test_send_uploads_files_before_creating_post(tmp_path: Path) -> None:
    media_file = tmp_path / "demo.txt"
    media_file.write_text("demo")

    channel = _make_channel()
    fake = _FakeAsyncClient()
    fake.post_responses.extend(
        [
            _FakeResponse({"file_infos": [{"id": "file1"}]}, status_code=201),
            _FakeResponse({"id": "post1"}, status_code=201),
        ]
    )
    channel._client = fake

    await channel.send(
        OutboundMessage(
            channel="mattermost",
            chat_id="chan1",
            content="hello",
            media=[str(media_file)],
        )
    )

    assert len(fake.post_calls) == 2
    upload_call = fake.post_calls[0]
    post_call = fake.post_calls[1]
    assert upload_call["url"].endswith("/api/v4/files")
    assert upload_call["data"] == {"channel_id": "chan1"}
    assert post_call["json"]["file_ids"] == ["file1"]


@pytest.mark.asyncio
async def test_posted_event_publishes_inbound_message_for_mention() -> None:
    channel = _make_channel(group_policy="mention")
    channel._bot_user_id = "bot1"
    channel._bot_username = "nanobot"
    channel._channel_types["chan1"] = "O"

    payload = {
        "event": "posted",
        "data": {
            "post": json.dumps(
                {
                    "id": "post1",
                    "user_id": "user1",
                    "channel_id": "chan1",
                    "message": "@nanobot hello",
                    "root_id": "",
                }
            )
        },
        "broadcast": {"channel_id": "chan1", "team_id": "team1"},
    }

    await channel._handle_websocket_message(json.dumps(payload))
    msg = await channel.bus.consume_inbound()
    assert msg.sender_id == "user1"
    assert msg.chat_id == "chan1"
    assert msg.content == "hello"
    assert msg.metadata["mattermost"]["post_id"] == "post1"
    assert msg.metadata["mattermost"]["team_id"] == "team1"
    assert msg.session_key == "mattermost:chan1:post1"


@pytest.mark.asyncio
async def test_posted_event_ignores_channel_message_without_mention() -> None:
    channel = _make_channel(group_policy="mention")
    channel._bot_user_id = "bot1"
    channel._bot_username = "nanobot"
    channel._channel_types["chan1"] = "O"

    payload = {
        "event": "posted",
        "data": {
            "post": json.dumps(
                {
                    "id": "post1",
                    "user_id": "user1",
                    "channel_id": "chan1",
                    "message": "hello",
                }
            )
        },
        "broadcast": {"channel_id": "chan1", "team_id": "team1"},
    }

    await channel._handle_websocket_message(json.dumps(payload))
    assert channel.bus.inbound_size == 0


@pytest.mark.asyncio
async def test_posted_event_respects_dm_policy_allowlist() -> None:
    channel = _make_channel(dm={"enabled": True, "policy": "allowlist", "allow_from": ["user1"]})
    channel._bot_user_id = "bot1"
    channel._bot_username = "nanobot"
    channel._channel_types["dm1"] = "D"

    allowed = {
        "event": "posted",
        "data": {
            "post": json.dumps(
                {
                    "id": "post1",
                    "user_id": "user1",
                    "channel_id": "dm1",
                    "message": "hello",
                }
            )
        },
        "broadcast": {"channel_id": "dm1"},
    }
    denied = {
        "event": "posted",
        "data": {
            "post": json.dumps(
                {
                    "id": "post2",
                    "user_id": "user2",
                    "channel_id": "dm1",
                    "message": "hello",
                }
            )
        },
        "broadcast": {"channel_id": "dm1"},
    }

    await channel._handle_websocket_message(json.dumps(allowed))
    assert channel.bus.inbound_size == 1
    await channel.bus.consume_inbound()

    await channel._handle_websocket_message(json.dumps(denied))
    assert channel.bus.inbound_size == 0


@pytest.mark.asyncio
async def test_posted_event_ignores_self_message() -> None:
    channel = _make_channel(group_policy="open")
    channel._bot_user_id = "bot1"
    channel._bot_username = "nanobot"
    channel._channel_types["chan1"] = "O"

    payload = {
        "event": "posted",
        "data": {
            "post": json.dumps(
                {
                    "id": "post1",
                    "user_id": "bot1",
                    "channel_id": "chan1",
                    "message": "hello",
                }
            )
        },
        "broadcast": {"channel_id": "chan1"},
    }

    await channel._handle_websocket_message(json.dumps(payload))
    assert channel.bus.inbound_size == 0


@pytest.mark.asyncio
async def test_load_bot_identity_reads_users_me() -> None:
    channel = _make_channel()
    fake = _FakeAsyncClient()
    fake.get_responses.append(_FakeResponse({"id": "bot1", "username": "nanobot"}))
    channel._client = fake

    await channel._load_bot_identity()

    assert fake.get_calls[0]["url"].endswith("/api/v4/users/me")
    assert channel._bot_user_id == "bot1"
    assert channel._bot_username == "nanobot"


@pytest.mark.asyncio
async def test_get_channel_type_fetches_and_caches_when_missing() -> None:
    channel = _make_channel()
    fake = _FakeAsyncClient()
    fake.get_responses.append(_FakeResponse({"id": "chan1", "type": "D"}))
    channel._client = fake

    first = await channel._get_channel_type("chan1", {})
    second = await channel._get_channel_type("chan1", {})

    assert first == "D"
    assert second == "D"
    assert len(fake.get_calls) == 1


@pytest.mark.asyncio
async def test_posted_event_ignores_message_when_channel_type_lookup_fails() -> None:
    channel = _make_channel(group_policy="open")
    channel._bot_user_id = "bot1"
    channel._bot_username = "nanobot"

    async def fail_lookup(_channel_id: str, _data: dict[str, object]) -> str:
        raise RuntimeError("lookup failed")

    channel._get_channel_type = fail_lookup  # type: ignore[method-assign]

    payload = {
        "event": "posted",
        "data": {
            "post": json.dumps(
                {
                    "id": "post1",
                    "user_id": "user1",
                    "channel_id": "chan1",
                    "message": "hello",
                }
            )
        },
        "broadcast": {"channel_id": "chan1"},
    }

    await channel._handle_websocket_message(json.dumps(payload))
    assert channel.bus.inbound_size == 0
