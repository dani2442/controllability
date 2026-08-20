from ddinf.systems.delay import hautus_delay_defect, lambert_roots, uncontrollable_pair


def test_lambert_roots_are_hautus_obstructions():
    data = uncontrollable_pair()
    roots = lambert_roots(data["A0"][1, 1], data["A1"][1, 1], data["h"], branches=2)
    defects = [hautus_delay_defect(z, data["A0"], data["A1"], data["B0"], data["h"])
               for z in roots]
    assert max(defects) < 1e-10
