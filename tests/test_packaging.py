"""Packaging assets: maintainer scripts, the launcher, model resolution, and (if
a built artifact is present) the .deb control metadata."""

from __future__ import annotations

import glob
import shutil
import subprocess
from pathlib import Path

import pytest

PKG = Path(__file__).parents[1] / "aria" / "packaging"
REPO = Path(__file__).parents[1]


def _syntax_ok(interpreter: str, script: Path) -> None:
    assert script.exists(), f"missing {script}"
    proc = subprocess.run(
        [interpreter, "-n", str(script)], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"{script.name} syntax error: {proc.stderr}"


@pytest.mark.parametrize(
    "name", ["postinst", "prerm", "postrm", "fetch_models.sh", "aria-launcher.sh"]
)
def test_shell_scripts_are_valid_sh(name):
    _syntax_ok("sh", PKG / name)


def test_build_script_is_valid_bash():
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    _syntax_ok("bash", PKG / "build_deb.sh")


def test_launcher_runs_bundled_venv_python():
    text = (PKG / "aria-launcher.sh").read_text()
    # Must run the bundled venv's interpreter directly (no system python/PATH).
    assert "/opt/aria/venv/bin/python -m aria" in text
    assert 'exec ' in text
    # And point the model resolver at the bundled voice.
    assert "ARIA_MODELS_DIR" in text and "/opt/aria/models" in text


def test_maintainer_scripts_have_shebang():
    for name in ("postinst", "prerm", "postrm"):
        first = (PKG / name).read_text().splitlines()[0]
        assert first.startswith("#!"), f"{name} missing shebang"


def test_postinst_does_not_enable_service_from_root():
    # The user runs `aria enable`; root postinst must NOT invoke systemctl.
    text = (PKG / "postinst").read_text()
    assert "systemctl" not in text


def test_resolve_piper_model_honors_env(tmp_path, monkeypatch):
    from aria.app import resolve_piper_model
    from aria.config.schema import AriaConfig

    cfg = AriaConfig()
    models = tmp_path / "models"
    models.mkdir()
    voice_file = models / f"{cfg.tts.voice}.onnx"
    voice_file.write_bytes(b"fake-onnx")
    monkeypatch.setenv("ARIA_MODELS_DIR", str(models))
    assert resolve_piper_model(cfg) == voice_file


def test_service_execstart_is_daemon():
    text = (PKG / "aria.service").read_text()
    assert "ExecStart=/usr/bin/aria daemon" in text


# --- only runs when a .deb has actually been built in the repo ------------
def _built_deb() -> Path | None:
    hits = sorted(glob.glob(str(REPO / "aria_*.deb")))
    return Path(hits[-1]) if hits else None


@pytest.mark.skipif(shutil.which("dpkg-deb") is None, reason="dpkg-deb not available")
def test_built_deb_control_metadata():
    deb = _built_deb()
    if deb is None:
        pytest.skip("no aria_*.deb built yet (run packaging/build_deb.sh)")
    info = subprocess.run(
        ["dpkg-deb", "--info", str(deb)], capture_output=True, text=True
    ).stdout
    assert "Package: aria" in info
    assert "Architecture:" in info
    assert "libportaudio2" in info and "libsecret-1-0" in info
    contents = subprocess.run(
        ["dpkg-deb", "--contents", str(deb)], capture_output=True, text=True
    ).stdout
    assert "/usr/bin/aria" in contents
    assert "/opt/aria/venv/bin/python" in contents
    assert "/usr/lib/systemd/user/aria.service" in contents
    assert "/opt/aria/models/" in contents
    # Branded app entry + icon (FIX 3) so notifications/menus show "Aria".
    assert "/usr/share/applications/aria.desktop" in contents
    assert "/usr/share/icons/hicolor/scalable/apps/aria.svg" in contents
