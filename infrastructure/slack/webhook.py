from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import dotenv_values


def _default_env_path() -> Path:
    # project_root / .env
    return Path(__file__).resolve().parents[2] / ".env"


def load_slack_webhook_url(env_file: Optional[os.PathLike[str] | str] = None) -> str:
    env_path = Path(env_file) if env_file is not None else _default_env_path()

    env_values: dict[str, Optional[str]] = {}
    if env_path.exists():
        env_values = dotenv_values(str(env_path), interpolate=False)

    def get_value(key: str) -> Optional[str]:
        v = env_values.get(key)
        if v is not None:
            return v
        return os.getenv(key)

    url = get_value("SLACK_WEBHOOK_URL")
    if not url:
        raise ValueError("Missing required Slack config value: SLACK_WEBHOOK_URL")

    return str(url)


def send_slack_payload(
    payload: dict[str, Any],
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    webhook_url: Optional[str] = None,
    timeout: float = 10.0,
    opener: Any = urlopen,
) -> str:
    """Send an arbitrary Slack Incoming Webhook payload.

    This enables richer formatting via `blocks` and `attachments` (including attachment colors).
    """

    url = webhook_url if webhook_url is not None else load_slack_webhook_url(env_file)

    body_bytes = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with opener(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            if status is None:
                try:
                    status = resp.getcode()
                except Exception:
                    status = None
            body = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Slack webhook request failed. status={e.code} body={body}"
        ) from e
    except URLError as e:
        raise RuntimeError("Slack webhook request failed (network error)") from e

    if status is None or int(status) < 200 or int(status) >= 300:
        raise RuntimeError(f"Slack webhook request failed. status={status} body={body}")

    return body


def send_slack_message(
    text: str,
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    webhook_url: Optional[str] = None,
    timeout: float = 10.0,
    opener: Any = urlopen,
) -> str:
    return send_slack_payload(
        {"text": text},
        env_file=env_file,
        webhook_url=webhook_url,
        timeout=timeout,
        opener=opener,
    )


def send_test_message(
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    webhook_url: Optional[str] = None,
    timeout: float = 10.0,
    opener: Any = urlopen,
) -> str:
    return send_slack_message(
        "テスト通知です",
        env_file=env_file,
        webhook_url=webhook_url,
        timeout=timeout,
        opener=opener,
    )


def main(argv: Optional[list[str]] = None, *, opener: Any = urlopen) -> int:
    parser = argparse.ArgumentParser(
        description="Send a Slack message via Incoming Webhook (default: test message)"
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Message text to send (if omitted, sends a test message)",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Path to .env file (default: project root .env)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Slack webhook URL (overrides SLACK_WEBHOOK_URL)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP request timeout seconds (default: 10)",
    )
    args = parser.parse_args(argv)

    if args.text is None:
        body = send_test_message(
            env_file=args.env,
            webhook_url=args.url,
            timeout=args.timeout,
            opener=opener,
        )
    else:
        body = send_slack_message(
            args.text,
            env_file=args.env,
            webhook_url=args.url,
            timeout=args.timeout,
            opener=opener,
        )

    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
