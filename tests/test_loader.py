import numpy as np
import polars as pl
import pytest

import FragmentsDataLoader as loader
from constants import ATOMIC_UNIT_TO_KEV, ATOMIC_MASS_UNIT_KG, ELEMENTARY_CHARGE


def _write_dataset(path, fragments: pl.DataFrame, total_events: int, isotopes: pl.DataFrame):
    pl.DataFrame({"targetA": [40], "targetZ": [18], "total_events": [total_events]},
                 schema={"targetA": pl.UInt16, "targetZ": pl.UInt8, "total_events": pl.UInt32}
                 ).write_parquet(path / "events_counts.parquet")
    fragments.write_parquet(path / "Test_fragments.parquet")
    isotopes.write_parquet(path / "isotopes_info.parquet")


def _fragments(events, A, Z, energies):
    n = len(events)
    return pl.DataFrame({"event": events, "targetA": [40] * n, "targetZ": [18] * n,
                         "fragmentA": A, "fragmentZ": Z, "energy_keV": energies},
                        schema={"event": pl.UInt32, "targetA": pl.UInt16, "targetZ": pl.UInt8,
                                "fragmentA": pl.UInt16, "fragmentZ": pl.UInt8, "energy_keV": pl.Float64})


ISOTOPES = pl.DataFrame({"Z": [2, 6, 10], "N": [2, 6, 12], "symbol": ["He", "C", "Ne"],
                         "half_life_s": [None, None, None], "atomic_mass": [4.002603, 12.0, None],
                         "neutron_separation_energy_keV": [None, None, None]},
                        schema={"Z": pl.Int64, "N": pl.Int64, "symbol": pl.String, "half_life_s": pl.Float64,
                                "atomic_mass": pl.Float64, "neutron_separation_energy_keV": pl.Float64})


def _monte_carlo_fraction(E_keV, Z, mass_u, V_kV, R, B, r0, n=2_000_000, seed=1):
    '''Fraction of isotropic emission directions which are trapped, calculated in SI units.'''
    rng = np.random.default_rng(seed)
    cos_theta = rng.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    E = E_keV * 1e3 * ELEMENTARY_CHARGE
    p = np.sqrt(2 * mass_u * ATOMIC_MASS_UNIT_KG * E)
    axial = E * cos_theta**2 <= Z * ELEMENTARY_CHARGE * V_kV * 1e3
    r_c = p * sin_theta / (Z * ELEMENTARY_CHARGE * B)
    radial = r0 + 2 * r_c <= R
    return np.mean(axial & radial)


@pytest.mark.parametrize("E_keV,Z,A", [(5, 2, 4), (25, 2, 4), (60, 2, 4), (20, 6, 12), (100, 6, 12), (130, 6, 12), (300, 6, 12)])
@pytest.mark.parametrize("r0", [0.0, 0.005])
def test_trappable_fraction_matches_isotropic_monte_carlo(tmp_path, monkeypatch, E_keV, Z, A, r0):
    V, R, B = 10, 0.015, 5
    _write_dataset(tmp_path, _fragments([0], [A], [Z], [E_keV]), 1, ISOTOPES)
    monkeypatch.setattr(loader, "DATA_PATH", str(tmp_path))
    data = loader.create_lf_with_weights("Test", half_life=0, V=V, R=R, B=B, r0=r0).collect()
    mass_u = data["atomic_mass"].item()
    expected = _monte_carlo_fraction(E_keV, Z, mass_u, V, R, B, r0)
    assert data["trappable_fraction"].item() == pytest.approx(expected, abs=2e-3)


def test_missing_mass_uses_mass_number(tmp_path, monkeypatch):
    _write_dataset(tmp_path, _fragments([0], [22], [10], [1.0]), 1, ISOTOPES)
    monkeypatch.setattr(loader, "DATA_PATH", str(tmp_path))
    data = loader.create_lf_with_weights("Test", half_life=0).collect()
    assert data["atomic_mass"].item() == 22.0
    assert data["mass_keV"].item() == pytest.approx(22 * ATOMIC_UNIT_TO_KEV)


def test_max_events_limits_events_not_rows(tmp_path, monkeypatch):
    # every event has 3 fragments, so 10 events give 30 rows
    events = [e for e in range(10) for _ in range(3)]
    _write_dataset(tmp_path, _fragments(events, [4] * 30, [2] * 30, [1.0] * 30), 10, ISOTOPES)
    monkeypatch.setattr(loader, "DATA_PATH", str(tmp_path))

    limited = loader.create_lf_with_weights("Test", half_life=0, max_events=4).collect()
    assert limited["event"].unique().sort().to_list() == [0, 1, 2, 3]
    assert limited.height == 12
    assert limited["total_events"].unique().to_list() == [4]

    everything = loader.create_lf_with_weights("Test", half_life=0, max_events=None).collect()
    assert everything.height == 30
    assert everything["total_events"].unique().to_list() == [10]

    # limit larger than the simulation doesn't change the normalisation
    large = loader.create_lf_with_weights("Test", half_life=0, max_events=1_000_000).collect()
    assert large["total_events"].unique().to_list() == [10]


def test_annihilation_radius_must_be_inside_trap():
    with pytest.raises(ValueError):
        loader.create_lf_with_weights("Test", R=0.015, r0=0.015)
