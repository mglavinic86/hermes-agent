import sys
from types import ModuleType, SimpleNamespace

import pytest

import hermes_cli.main as cli_main


class _HardExitCalled(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


def test_cmd_gateway_run_routes_clean_return_through_hard_exit(monkeypatch):
    """The actual ``hermes gateway run`` entrypoint must not return into
    Py_FinalizeEx, where in-flight cron worker threads can wedge launchd
    restarts after graceful gateway teardown.
    """
    fake_gateway_cli = ModuleType("hermes_cli.gateway")
    setattr(fake_gateway_cli, "gateway_command", lambda args: None)
    fake_gateway_run = ModuleType("gateway.run")
    setattr(
        fake_gateway_run,
        "_exit_after_graceful_shutdown",
        lambda code: (_ for _ in ()).throw(_HardExitCalled(code)),
    )

    monkeypatch.setitem(sys.modules, "hermes_cli.gateway", fake_gateway_cli)
    monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)
    monkeypatch.setattr(cli_main, "_sync_bundled_skills_quietly", lambda: None)

    with pytest.raises(_HardExitCalled) as exc_info:
        cli_main.cmd_gateway(SimpleNamespace(gateway_command="run"))

    assert exc_info.value.code == 0


def test_cmd_gateway_run_routes_service_restart_code_through_hard_exit(monkeypatch):
    """Exit 75 from a planned service restart must bypass interpreter
    finalization too; re-raising SystemExit recreates the cron-thread hang.
    """
    def _raise_restart(_args):
        raise SystemExit(75)

    fake_gateway_cli = ModuleType("hermes_cli.gateway")
    setattr(fake_gateway_cli, "gateway_command", _raise_restart)
    fake_gateway_run = ModuleType("gateway.run")
    setattr(
        fake_gateway_run,
        "_exit_after_graceful_shutdown",
        lambda code: (_ for _ in ()).throw(_HardExitCalled(code)),
    )

    monkeypatch.setitem(sys.modules, "hermes_cli.gateway", fake_gateway_cli)
    monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)
    monkeypatch.setattr(cli_main, "_sync_bundled_skills_quietly", lambda: None)

    with pytest.raises(_HardExitCalled) as exc_info:
        cli_main.cmd_gateway(SimpleNamespace(gateway_command="run"))

    assert exc_info.value.code == 75
