import os

import numpy as np

from trap import TTrap


def test_matrix_rows_are_matched_by_name():
    trap = TTrap()
    path = os.path.join(os.path.dirname(__import__("trap").__file__), "ElectrodesPotentialMap.txt")
    rows = {line.split("\t")[0]: np.array(line.strip().split("\t")[1:trap.segments + 1], dtype=float)
            for line in open(path) if line.strip()}
    for name in ["P3", "P11", "HV3", "T6"]:
        trap.SetEverythingToZero()
        trap.SetElectrodeV(name, 1.0)
        np.testing.assert_allclose(trap.get_final_V(), rows[name])
    # grounded electrode doesn't contribute
    trap.SetEverythingToZero()
    trap.SetElectrodeV("C1", 1000.0)
    assert np.all(trap.get_final_V() == 0)


def test_set_everything_to_zero_resets_all_electrodes():
    trap = TTrap()
    for electrode in trap.GetElectrodeNames():
        trap.SetElectrodeV(electrode, 5.0)
    trap.SetEverythingToZero()
    assert all(electrode.GetPotential() == 0 for electrode in trap.electrodes)


def test_instances_are_independent():
    first, second = TTrap(), TTrap(position=-100)
    first.SetElectrodeV("P11", 10.0)
    assert second.GetElectrodeV("P11") == 0
    assert first.electrodes is not second.electrodes
    assert second.GetElectrodePosition("P11") == first.GetElectrodePosition("P11") - 100
