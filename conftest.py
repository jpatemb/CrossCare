"""Offline guards for the test suite.

Every test in this repo must pass with no ANTHROPIC_API_KEY and no network
access — the diagnostic/triage logic in tools.py is plain Python specifically
so it's unit-testable without a live model call. Live API usage belongs in
manual_loop.py, never in tests/.
"""

import socket

import pytest

# All three are read by the Anthropic SDK when a client is constructed with no
# explicit arguments, so stripping only ANTHROPIC_API_KEY would be incomplete.
_ANTHROPIC_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
)

_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", ""})


class OfflineTestError(RuntimeError):
    """A test tried to open an outbound network connection."""


@pytest.fixture(autouse=True)
def no_anthropic_api_key(monkeypatch):
    """Tests must behave identically whether or not a key is exported.

    Function-scoped so a test that deliberately sets a key cannot leak it
    into the next test.
    """
    for var in _ANTHROPIC_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="session", autouse=True)
def block_outbound_network():
    """Fail loudly instead of hanging if a test reaches for the network.

    Patches `socket.socket.connect`/`connect_ex` — the choke point every
    stdlib, httpx and requests client funnels through — so no pytest-socket
    dependency is needed.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _is_outbound(sock, address):
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return False
        host = address[0] if isinstance(address, tuple) else address
        return host not in _LOCAL_HOSTS

    def _explain(address):
        return OfflineTestError(
            f"Outbound network call to {address!r} blocked. Tests in this repo "
            "run offline. Live API calls belong in manual_loop.py, not in tests/."
        )

    def guarded_connect(self, address, *args, **kwargs):
        if _is_outbound(self, address):
            raise _explain(address)
        return real_connect(self, address, *args, **kwargs)

    def guarded_connect_ex(self, address, *args, **kwargs):
        if _is_outbound(self, address):
            raise _explain(address)
        return real_connect_ex(self, address, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", guarded_connect)
        mp.setattr(socket.socket, "connect_ex", guarded_connect_ex)
        yield
