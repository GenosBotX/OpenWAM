"""验证凭证启动入口，不访问 TOS，也不使用真实密钥。"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_with_tos_credentials.py"
KEYS = ("TOS_ACCESS_KEY", "TOS_SECRET_KEY", "TOS_SECURITY_TOKEN", "TOS_ENDPOINT", "TOS_REGION")
VALUES = {"TOS_ACCESS_KEY": "fake-access", "TOS_SECRET_KEY": "fake-secret"}
PROBE = "import json,os; print(json.dumps({k:os.getenv(k) for k in " + repr(KEYS) + "}))"


def run_wrapper(path, overrides=None, probe=PROBE):
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env.update(overrides or {})
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path), sys.executable, "-c", probe],
        env=env, capture_output=True, text=True,
    )


def credential_file(tmp_path, values=None, mode=0o600):
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps(VALUES if values is None else values))
    path.chmod(mode)
    return path


def test_file_credentials_reach_child(tmp_path):
    result = run_wrapper(credential_file(tmp_path, {**VALUES, "TOS_SECURITY_TOKEN": "fake-token"}))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {**dict.fromkeys(KEYS), **VALUES, "TOS_SECURITY_TOKEN": "fake-token"}


def test_full_environment_does_not_require_file(tmp_path):
    env = {**VALUES, "TOS_SECURITY_TOKEN": "fake-env-token"}
    result = run_wrapper(tmp_path / "missing.json", env)
    assert result.returncode == 0, result.stderr
    assert all(json.loads(result.stdout)[k] == v for k, v in env.items())


def test_environment_credentials_take_precedence(tmp_path):
    result = run_wrapper(credential_file(tmp_path), {"TOS_ACCESS_KEY": "env-ak", "TOS_SECRET_KEY": "env-sk"})
    assert result.returncode == 0
    assert json.loads(result.stdout)["TOS_ACCESS_KEY"] == "env-ak"


@pytest.mark.parametrize("env", [
    {"TOS_ACCESS_KEY": "fake-partial"},
    {"TOS_SECRET_KEY": "fake-partial"},
    {"TOS_ACCESS_KEY": "", "TOS_SECRET_KEY": "fake-partial"},
    {"TOS_ACCESS_KEY": "fake-partial", "TOS_SECRET_KEY": "   "},
])
def test_partial_environment_is_rejected_without_file_fallback(tmp_path, env):
    result = run_wrapper(credential_file(tmp_path), env)
    assert result.returncode != 0
    assert "同时设置" in result.stderr
    assert "fake-partial" not in result.stderr
    assert not result.stdout


def test_missing_file_reports_worker_configuration(tmp_path):
    result = run_wrapper(tmp_path / "missing.json")
    assert result.returncode != 0
    assert "OPENWAM_TOS_CREDENTIALS" in result.stderr
    assert "worker" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604])
def test_file_permissions_are_checked(tmp_path, mode):
    result = run_wrapper(credential_file(tmp_path, mode=mode))
    assert result.returncode != 0
    assert "0600" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("values", [
    [], {"TOS_ACCESS_KEY": "fake-access"}, {**VALUES, "unexpected": "private-value"},
    {**VALUES, "TOS_ACCESS_KEY": 123}, {**VALUES, "TOS_SECRET_KEY": ""},
    {**VALUES, "TOS_SECURITY_TOKEN": None},
])
def test_invalid_json_values_do_not_leak(tmp_path, values):
    result = run_wrapper(credential_file(tmp_path, values))
    assert result.returncode != 0
    assert not result.stdout
    assert "Traceback" not in result.stderr
    assert all(value not in result.stderr for value in ("fake-access", "fake-secret", "private-value"))


def test_invalid_json_does_not_echo_contents(tmp_path):
    path = credential_file(tmp_path)
    path.write_text("private-invalid-json")
    result = run_wrapper(path)
    assert result.returncode != 0
    assert "private-invalid-json" not in result.stderr
    assert "Traceback" not in result.stderr


def test_file_credentials_clear_stale_environment_token(tmp_path):
    result = run_wrapper(credential_file(tmp_path), {"TOS_SECURITY_TOKEN": "stale-token"})
    assert result.returncode == 0
    assert json.loads(result.stdout)["TOS_SECURITY_TOKEN"] is None


def test_environment_connection_settings_override_file(tmp_path):
    values = {**VALUES, "TOS_ENDPOINT": "file-endpoint", "TOS_REGION": "file-region"}
    result = run_wrapper(credential_file(tmp_path, values), {"TOS_ENDPOINT": "env-endpoint", "TOS_REGION": "env-region"})
    assert result.returncode == 0
    assert json.loads(result.stdout)["TOS_ENDPOINT"] == "env-endpoint"
    assert json.loads(result.stdout)["TOS_REGION"] == "env-region"


def test_file_connection_defaults_are_loaded(tmp_path):
    values = {**VALUES, "TOS_ENDPOINT": "file-endpoint", "TOS_REGION": "file-region"}
    result = run_wrapper(credential_file(tmp_path, values))
    assert result.returncode == 0
    assert all(json.loads(result.stdout)[k] == v for k, v in values.items())


def test_wrong_owner_is_rejected(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("tos_credential_wrapper", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = credential_file(tmp_path)
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(module.os, "getuid", lambda: path.stat().st_uid + 1)
    with pytest.raises(module.CredentialError, match="当前用户"):
        module.load_credentials(path)


def test_child_exit_status_is_preserved(tmp_path):
    result = run_wrapper(credential_file(tmp_path), probe="raise SystemExit(7)")
    assert result.returncode == 7


def test_launcher_defaults_to_shared_credentials():
    launcher = ROOT / "scripts_piper/launch_volcano_piper_x_20261022_20261023_tos_eef_8gpu_e10.sh"
    env = dict(os.environ, OPENWAM_LAUNCH_DRYRUN="1")
    for key in ("OPENWAM_TOS_CREDENTIALS", "TOS_CREDENTIAL_FILE"):
        env.pop(key, None)
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert "/vePFS/tangyuting/.config/openwam/tos-linglong.json" in result.stdout
    assert "/root/.config/openwam" not in result.stdout


@pytest.mark.parametrize("overrides,expected", [
    ({"TOS_CREDENTIAL_FILE": "/shared/credentials.json"}, "/shared/credentials.json"),
    ({"TOS_CREDENTIAL_FILE": "/shared/credentials.json", "OPENWAM_TOS_CREDENTIALS": "/job/credentials.json"}, "/job/credentials.json"),
])
def test_launcher_credential_path_overrides(overrides, expected):
    launcher = ROOT / "scripts_piper/launch_volcano_piper_x_20261022_20261023_tos_eef_8gpu_e10.sh"
    env = dict(os.environ, OPENWAM_LAUNCH_DRYRUN="1")
    for key in ("OPENWAM_TOS_CREDENTIALS", "TOS_CREDENTIAL_FILE"):
        env.pop(key, None)
    env.update(overrides)
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert expected in result.stdout
