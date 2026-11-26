"""Local Hugging Face and W&B credentials, attached to Modal runs as ephemeral secrets.

Nothing is stored in Modal or baked into an image: each app run reads the
credentials from this machine and passes them for that run only. A missing
credential gives an empty secret, which is fine for ungated models and for runs
with W&B turned off (the default here).
"""

import netrc
import os
from functools import cache
from pathlib import Path

import modal

WANDB_NETRC_HOST = "api.wandb.ai"


@cache
def _wandb_api_key() -> str | None:
    """``WANDB_API_KEY`` if set, else the entry ``wandb login`` wrote to ``~/.netrc``."""
    if key := os.environ.get("WANDB_API_KEY"):
        return key
    try:
        authenticators = netrc.netrc().authenticators(WANDB_NETRC_HOST)
    except (OSError, netrc.NetrcParseError):
        return None
    return authenticators[2] if authenticators else None


def wandb_secret() -> modal.Secret:
    """The local W&B key as an ephemeral secret (empty if there is none)."""
    return modal.Secret.from_dict({"WANDB_API_KEY": _wandb_api_key()})


@cache
def _hf_token() -> str | None:
    """``HF_TOKEN`` if set, then a ``.env`` above the working directory, then ``hf auth login``'s file."""
    if token := os.environ.get("HF_TOKEN"):
        return token
    from dotenv import dotenv_values, find_dotenv

    if (dotenv := find_dotenv(usecwd=True)) and (token := dotenv_values(dotenv).get("HF_TOKEN")):
        return token
    token_file = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    if token_file.is_file():
        return token_file.read_text().strip() or None
    return None


def hf_secret() -> modal.Secret:
    """The local Hugging Face token as an ephemeral secret (empty if there is none)."""
    return modal.Secret.from_dict({"HF_TOKEN": _hf_token()})
