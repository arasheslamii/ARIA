"""The commerce browser engine installs itself: readiness check, idempotent
command sequence, privilege escalation only when the venv isn't writable, and a
clear fallback on failure. NOTHING here runs a real subprocess or touches the
network — subprocess.run and every probe are mocked."""

from __future__ import annotations

import subprocess

import pytest

import aria.agents.browser_setup as bs
from aria.agents.browser_setup import (
    BrowserSetupError,
    commerce_engine_ready,
    install_commerce_engine,
)


def _fake_venv(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").touch()
    return tmp_path


def _capture(monkeypatch):
    """Record every subprocess.run argv instead of executing it."""
    calls: list[list[str]] = []

    def fake_run(argv, **kw):
        calls.append(list(argv))

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(bs.subprocess, "run", fake_run)
    return calls


def _stub_probes(monkeypatch, *, pip, pkgs, chromium, writable):
    monkeypatch.setattr(bs, "_pip_present", lambda: pip)
    monkeypatch.setattr(bs, "_browser_use_importable", lambda: pkgs)
    monkeypatch.setattr(bs, "chromium_installed", lambda: chromium)
    monkeypatch.setattr(bs.os, "access", lambda _p, _m: writable)


# --- readiness reflects imports AND chromium ------------------------------
def test_ready_false_when_imports_fail(monkeypatch):
    monkeypatch.setattr(bs, "_browser_use_importable", lambda: False)
    monkeypatch.setattr(bs, "chromium_installed", lambda: True)
    assert commerce_engine_ready() is False


def test_ready_false_when_chromium_absent(monkeypatch):
    monkeypatch.setattr(bs, "_browser_use_importable", lambda: True)
    monkeypatch.setattr(bs, "chromium_installed", lambda: False)
    assert commerce_engine_ready() is False


def test_ready_true_when_both_present(monkeypatch):
    monkeypatch.setattr(bs, "_browser_use_importable", lambda: True)
    monkeypatch.setattr(bs, "chromium_installed", lambda: True)
    assert commerce_engine_ready() is True


# --- the full, correct command sequence -----------------------------------
def test_install_builds_full_sequence_when_nothing_present(monkeypatch, tmp_path):
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(bs, "runtime_venv", lambda: venv)
    _stub_probes(monkeypatch, pip=False, pkgs=False, chromium=False, writable=True)
    monkeypatch.setattr(bs, "commerce_engine_ready", lambda: True)  # verify passes
    calls = _capture(monkeypatch)

    install_commerce_engine()

    joined = [" ".join(c) for c in calls]
    assert any("ensurepip --upgrade" in c for c in joined)
    assert any("pip install" in c and "browser-use>=0.1" in c
               and "playwright>=1.40" in c and "langchain-openai>=0.1" in c for c in joined)
    assert any("playwright install chromium" in c for c in joined)
    assert any("playwright install-deps" in c for c in joined)
    # venv is writable -> no privilege escalation
    assert all(c[0] != "pkexec" and c[0] != "sudo" for c in calls)


# --- idempotent: satisfied steps are skipped ------------------------------
def test_install_skips_satisfied_steps(monkeypatch, tmp_path):
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(bs, "runtime_venv", lambda: venv)
    # pip + packages already there; only Chromium is missing.
    _stub_probes(monkeypatch, pip=True, pkgs=True, chromium=False, writable=True)
    monkeypatch.setattr(bs, "commerce_engine_ready", lambda: True)
    calls = _capture(monkeypatch)

    install_commerce_engine()

    joined = [" ".join(c) for c in calls]
    assert not any("ensurepip" in c for c in joined)          # skipped
    assert not any("pip install" in c and "browser-use" in c for c in joined)  # skipped
    assert any("playwright install chromium" in c for c in joined)  # still run


def test_install_noops_when_everything_present(monkeypatch, tmp_path):
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(bs, "runtime_venv", lambda: venv)
    _stub_probes(monkeypatch, pip=True, pkgs=True, chromium=True, writable=True)
    monkeypatch.setattr(bs, "commerce_engine_ready", lambda: True)
    calls = _capture(monkeypatch)

    install_commerce_engine()
    assert calls == []  # nothing to do


# --- privilege escalation ONLY when the venv isn't writable ---------------
def test_pkexec_prefix_when_venv_not_writable(monkeypatch, tmp_path):
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(bs, "runtime_venv", lambda: venv)
    _stub_probes(monkeypatch, pip=False, pkgs=False, chromium=False, writable=False)
    monkeypatch.setattr(bs.shutil, "which",
                        lambda n: "/usr/bin/pkexec" if n == "pkexec" else None)
    monkeypatch.setattr(bs, "commerce_engine_ready", lambda: True)
    calls = _capture(monkeypatch)

    install_commerce_engine()
    assert calls and all(c[0] == "pkexec" for c in calls)  # graphical polkit prompt


def test_sudo_fallback_when_no_pkexec(monkeypatch, tmp_path):
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(bs, "runtime_venv", lambda: venv)
    _stub_probes(monkeypatch, pip=False, pkgs=False, chromium=False, writable=False)
    monkeypatch.setattr(bs.shutil, "which",
                        lambda n: "/usr/bin/sudo" if n == "sudo" else None)
    monkeypatch.setattr(bs, "commerce_engine_ready", lambda: True)
    calls = _capture(monkeypatch)

    install_commerce_engine()
    assert calls and all(c[0] == "sudo" for c in calls)


def test_no_prefix_when_writable_even_without_escalators(monkeypatch, tmp_path):
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(bs, "runtime_venv", lambda: venv)
    _stub_probes(monkeypatch, pip=False, pkgs=False, chromium=False, writable=True)
    monkeypatch.setattr(bs.shutil, "which", lambda n: None)
    monkeypatch.setattr(bs, "commerce_engine_ready", lambda: True)
    calls = _capture(monkeypatch)

    install_commerce_engine()
    assert calls and all(c[0] != "pkexec" and c[0] != "sudo" for c in calls)


# --- failures name the exact manual fallback ------------------------------
def test_subprocess_failure_raises_with_fallback_command(monkeypatch, tmp_path):
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(bs, "runtime_venv", lambda: venv)
    _stub_probes(monkeypatch, pip=False, pkgs=False, chromium=False, writable=True)

    def boom(argv, **kw):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(bs.subprocess, "run", boom)
    with pytest.raises(BrowserSetupError, match=r"sudo bash /opt/aria/scripts/install_commerce.sh"):
        install_commerce_engine()


def test_verify_failure_raises_with_fallback_command(monkeypatch, tmp_path):
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(bs, "runtime_venv", lambda: venv)
    _stub_probes(monkeypatch, pip=False, pkgs=False, chromium=False, writable=True)
    _capture(monkeypatch)  # steps "succeed"...
    monkeypatch.setattr(bs, "commerce_engine_ready", lambda: False)  # ...but engine absent
    with pytest.raises(BrowserSetupError, match="install_commerce.sh"):
        install_commerce_engine()


def test_missing_venv_python_raises_with_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(bs, "runtime_venv", lambda: tmp_path)  # no bin/python
    with pytest.raises(BrowserSetupError, match="install_commerce.sh"):
        install_commerce_engine()


# --- the CLI subcommand is registered and callable ------------------------
def test_install_commerce_subcommand_registered_and_callable(monkeypatch):
    import aria.__main__ as m

    called = {"v": False}

    def fake_install(progress=None):
        called["v"] = True
        if progress:
            progress("→ installing")

    monkeypatch.setattr("aria.agents.browser_setup.install_commerce_engine", fake_install)
    monkeypatch.setattr("aria.agents.browser_setup.commerce_engine_ready", lambda: False)

    assert m.main(["install-commerce"]) == 0
    assert called["v"] is True


def test_setup_browser_alias_calls_the_same_path(monkeypatch):
    import aria.__main__ as m

    called = {"v": False}
    monkeypatch.setattr("aria.agents.browser_setup.install_commerce_engine",
                        lambda progress=None: called.__setitem__("v", True))
    monkeypatch.setattr("aria.agents.browser_setup.commerce_engine_ready", lambda: False)

    assert m.main(["setup-browser"]) == 0
    assert called["v"] is True


def test_install_commerce_noops_when_already_ready(monkeypatch):
    import aria.__main__ as m

    monkeypatch.setattr("aria.agents.browser_setup.commerce_engine_ready", lambda: True)

    def fail(progress=None):
        raise AssertionError("must not install when already ready")

    monkeypatch.setattr("aria.agents.browser_setup.install_commerce_engine", fail)
    assert m.main(["install-commerce"]) == 0


# --- the wizard offers the install ONLY when the engine isn't ready --------
def test_wizard_install_prompt_only_when_not_ready():
    import aria.tui.wizard as wiz

    assert wiz.commerce_install_prompt(True) is None
    assert "one-time browser setup" in wiz.commerce_install_prompt(False)
