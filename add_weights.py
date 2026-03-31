import numpy as np
from ase.io import read, write

ADSORBATE_ELEMENTS = {"C", "H", "O"}
ADSORBATE_WEIGHT = 5.0

for fname in ["train1.extxyz", "test1.extxyz"]:
    atoms_list = read(fname, index=":")
    n_ads = 0
    for atoms in atoms_list:
        symbols = atoms.get_chemical_symbols()
        # Per-atom weight: adsorbate atoms get ADSORBATE_WEIGHT, surface atoms get 1.0
        atom_weights = np.array(
            [ADSORBATE_WEIGHT if s in ADSORBATE_ELEMENTS else 1.0 for s in symbols]
        )
        atoms.arrays["atom_forces_weight"] = atom_weights
        if any(s in ADSORBATE_ELEMENTS for s in symbols):
            n_ads += 1

    out = fname.replace(".extxyz", "_weighted.extxyz")
    write(out, atoms_list)
    print(f"{fname}: {n_ads}/{len(atoms_list)} structures with adsorbate atoms -> {out}")
