import src.data as data_module
from src.data import (
    BARANGAYS,
    DEFAULT_BARANGAY_COORDS,
    load_barangay_coords,
    save_barangay_coords,
)


def test_load_barangay_coords_falls_back_to_defaults_for_missing_file(tmp_path):
    """Missing coordinate files should not break the app."""
    missing_path = tmp_path / "missing_coords.json"

    coords = load_barangay_coords(str(missing_path))

    assert coords == DEFAULT_BARANGAY_COORDS


def test_save_barangay_coords_round_trips_to_disk(tmp_path):
    """Persisted coordinates should be readable and remain complete."""
    path = tmp_path / "coords.json"
    updated = dict(DEFAULT_BARANGAY_COORDS)
    updated["Amsic"] = (15.15, 120.60)

    save_barangay_coords(updated, str(path))
    loaded = load_barangay_coords(str(path))

    assert loaded["Amsic"] == (15.15, 120.60)
    assert set(loaded) == set(BARANGAYS)


def test_load_barangay_coords_handles_corrupted_json(tmp_path):
    """Corrupted JSON should fall back to defaults instead of crashing."""
    path = tmp_path / "coords.json"
    path.write_text("{not-valid-json", encoding="utf-8")

    coords = load_barangay_coords(str(path))

    assert coords == DEFAULT_BARANGAY_COORDS


def test_save_barangay_coords_leaves_cached_distance_matrix_on_disk(tmp_path, monkeypatch):
    """Stale matrices are rejected by signature, not deleted from disk."""
    coords_path = tmp_path / "coords.json"
    matrix_path = tmp_path / "distance_matrix.npy"
    matrix_path.write_bytes(b"cached-matrix")
    monkeypatch.setattr(data_module, "DISTANCE_MATRIX_PATH", str(matrix_path), raising=False)

    save_barangay_coords(dict(DEFAULT_BARANGAY_COORDS), str(coords_path))

    assert matrix_path.exists()
