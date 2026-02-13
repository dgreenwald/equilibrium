from equilibrium.settings import get_settings


def test_env_loaded_from_parent_directory(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    work_dir = project_root / "analysis" / "notebooks"
    work_dir.mkdir(parents=True)
    (project_root / ".env").write_text("EQUILIBRIUM_VERBOSE=true\n")

    monkeypatch.chdir(work_dir)
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.verbose is True


def test_nearest_env_file_wins(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    work_dir = project_root / "analysis" / "notebooks"
    work_dir.mkdir(parents=True)
    (project_root / ".env").write_text("EQUILIBRIUM_VERBOSE=true\n")
    (work_dir / ".env").write_text("EQUILIBRIUM_VERBOSE=false\n")

    monkeypatch.chdir(work_dir)
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.verbose is False


def test_os_env_overrides_parent_env(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    work_dir = project_root / "analysis" / "notebooks"
    work_dir.mkdir(parents=True)
    (project_root / ".env").write_text("EQUILIBRIUM_VERBOSE=false\n")

    monkeypatch.chdir(work_dir)
    monkeypatch.setenv("EQUILIBRIUM_VERBOSE", "true")
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.verbose is True
