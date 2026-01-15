import pytest

from equilibrium.settings import get_settings


@pytest.fixture(autouse=True)
def _isolate_equilibrium_paths(tmp_path_factory, monkeypatch):
    base_dir = tmp_path_factory.mktemp("equilibrium_data")
    monkeypatch.setenv("EQUILIBRIUM_PATHS__DATA_DIR", str(base_dir))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
