import pytest


@pytest.fixture(autouse=True)
def _degenerate_capture_dir(tmp_path_factory, monkeypatch):
    """Tests that hit the degenerate path must not write captures into the working tree's runs/."""
    monkeypatch.setenv("ELH_DEGENERATE_DIR", str(tmp_path_factory.mktemp("degenerate")))
