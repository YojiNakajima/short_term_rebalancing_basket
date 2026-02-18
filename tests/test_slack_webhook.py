import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from infrastructure.slack.webhook import (
    load_slack_webhook_url,
    main,
    send_slack_message,
    send_slack_payload,
)


class _FakeResponse:
    def __init__(self, *, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestSlackWebhook(unittest.TestCase):
    def test_load_slack_webhook_url_reads_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            env_path = Path(d) / ".env"
            env_path.write_text(
                "\n".join([
                    "SLACK_WEBHOOK_URL=https://example.com/hook",
                ]),
                encoding="utf-8",
            )

            with unittest.mock.patch.dict(os.environ, {}, clear=True):
                url = load_slack_webhook_url(env_path)

        self.assertEqual(url, "https://example.com/hook")

    def test_load_slack_webhook_url_raises_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            env_path = Path(d) / ".env"
            env_path.write_text("\n", encoding="utf-8")

            with unittest.mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_slack_webhook_url(env_path)

    def test_send_slack_message_posts_expected_payload(self) -> None:
        called = {}

        def fake_opener(req, timeout):
            called["timeout"] = timeout
            called["url"] = req.full_url
            called["method"] = req.get_method()
            called["headers"] = dict(req.header_items())
            called["data"] = req.data
            return _FakeResponse(status=200, body=b"ok")

        body = send_slack_message(
            "hello",
            webhook_url="https://example.com/hook",
            timeout=3.0,
            opener=fake_opener,
        )

        self.assertEqual(body, "ok")
        self.assertEqual(called["timeout"], 3.0)
        self.assertEqual(called["url"], "https://example.com/hook")
        self.assertEqual(called["method"], "POST")
        self.assertEqual(called["headers"].get("Content-type"), "application/json")
        self.assertEqual(json.loads(called["data"].decode("utf-8")), {"text": "hello"})

    def test_send_slack_payload_posts_expected_payload(self) -> None:
        called = {}

        def fake_opener(req, timeout):
            called["timeout"] = timeout
            called["url"] = req.full_url
            called["method"] = req.get_method()
            called["headers"] = dict(req.header_items())
            called["data"] = req.data
            return _FakeResponse(status=200, body=b"ok")

        payload = {
            "text": "fallback",
            "attachments": [
                {
                    "color": "#2eb886",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": "*hello*"},
                        }
                    ],
                }
            ],
        }

        body = send_slack_payload(
            payload,
            webhook_url="https://example.com/hook",
            timeout=3.0,
            opener=fake_opener,
        )

        self.assertEqual(body, "ok")
        self.assertEqual(called["timeout"], 3.0)
        self.assertEqual(called["url"], "https://example.com/hook")
        self.assertEqual(called["method"], "POST")
        self.assertEqual(called["headers"].get("Content-type"), "application/json")
        self.assertEqual(json.loads(called["data"].decode("utf-8")), payload)

    def test_send_slack_message_raises_on_non_2xx(self) -> None:
        def fake_opener(req, timeout):
            return _FakeResponse(status=500, body=b"ng")

        with self.assertRaises(RuntimeError) as ctx:
            send_slack_message(
                "hello",
                webhook_url="https://example.com/hook",
                opener=fake_opener,
            )

        self.assertIn("status=500", str(ctx.exception))

    def test_main_sends_test_message_when_text_is_omitted(self) -> None:
        called = {}

        def fake_opener(req, timeout):
            called["url"] = req.full_url
            called["method"] = req.get_method()
            called["headers"] = dict(req.header_items())
            called["data"] = req.data
            return _FakeResponse(status=200, body=b"ok")

        buf = StringIO()
        with redirect_stdout(buf):
            rc = main(["--url", "https://example.com/hook"], opener=fake_opener)

        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue().strip(), "ok")
        self.assertEqual(called["url"], "https://example.com/hook")
        self.assertEqual(called["method"], "POST")
        self.assertEqual(called["headers"].get("Content-type"), "application/json")
        self.assertEqual(
            json.loads(called["data"].decode("utf-8")), {"text": "テスト通知です"}
        )

    def test_main_sends_message_when_text_is_provided(self) -> None:
        called = {}

        def fake_opener(req, timeout):
            called["url"] = req.full_url
            called["method"] = req.get_method()
            called["headers"] = dict(req.header_items())
            called["data"] = req.data
            return _FakeResponse(status=200, body=b"ok")

        buf = StringIO()
        with redirect_stdout(buf):
            rc = main(["hello", "--url", "https://example.com/hook"], opener=fake_opener)

        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue().strip(), "ok")
        self.assertEqual(called["url"], "https://example.com/hook")
        self.assertEqual(called["method"], "POST")
        self.assertEqual(called["headers"].get("Content-type"), "application/json")
        self.assertEqual(json.loads(called["data"].decode("utf-8")), {"text": "hello"})


if __name__ == "__main__":
    unittest.main()
