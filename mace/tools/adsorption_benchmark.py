###########################################################################################
# Adsorption energy benchmark for MACE fine-tuning
# Adapted from 04_benchmark_finetuned.py (FTuMLP1 project)
# Runs after each training epoch to track adsorption energy prediction quality
###########################################################################################

import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HA_TO_EV = 27.2114  # Hartree → eV (CP2K PDB energies are in Hartree)


# ─── Fixed-atom detection ─────────────────────────────────────────────────────

def _parse_cp2k_fixed_atoms(inp_path: Path) -> List[int]:
    """
    Parse fixed atom indices (0-indexed) from a CP2K input file.

    Reads all ``&FIXED_ATOMS … LIST … &END FIXED_ATOMS`` blocks in the file.
    CP2K uses 1-based indices; they are converted to 0-based for ASE.

    Supports:
    - Space-separated integers:   ``LIST 1 2 3 4``
    - Ranges with ``..``:         ``LIST 1..10 15 20..25``
    - Multi-line LIST (continuation lines between &FIXED_ATOMS and &END)
    """
    text = inp_path.read_text(errors="replace")

    # Collect all LIST values from every &FIXED_ATOMS block
    all_indices: List[int] = []
    for block in re.finditer(
        r"&FIXED_ATOMS(.*?)&END\s+FIXED_ATOMS", text, flags=re.IGNORECASE | re.DOTALL
    ):
        block_text = block.group(1)
        # Find every LIST line (may span several lines with continuations)
        for list_match in re.finditer(r"LIST\s+([\d\s..]+)", block_text, flags=re.IGNORECASE):
            raw = list_match.group(1)
            # parse tokens: either "N..M" ranges or plain integers
            for token in raw.split():
                if ".." in token:
                    parts = token.split("..")
                    start, end = int(parts[0]), int(parts[1])
                    all_indices.extend(range(start - 1, end))  # 1→0 indexed
                else:
                    try:
                        all_indices.append(int(token) - 1)  # 1→0 indexed
                    except ValueError:
                        pass

    return sorted(set(all_indices))


def get_fixed_atoms(atoms, structure_dir: Optional[Path] = None) -> List[int]:
    """
    Determine which atom indices should be fixed, in priority order:

    1. FixAtoms constraints already present in the ``atoms`` object
       (e.g. read from extxyz with ``constraint`` info key).
    2. A CP2K input file (``*.inp``) found in *structure_dir*.
    3. No fixed atoms — return an empty list.

    Parameters
    ----------
    atoms : ase.Atoms
    structure_dir : Path, optional
        Directory to search for a CP2K input file (``*.inp``).

    Returns
    -------
    List[int]
        0-indexed atom indices to fix (may be empty).
    """
    from ase.constraints import FixAtoms

    # Priority 1: constraints already in the Atoms object
    existing: List[int] = []
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            existing.extend(c.get_indices().tolist())
    if existing:
        logger.info(f"  Fixed atoms: {len(existing)} indices read from structure file")
        return sorted(set(existing))

    # Priority 2: parse CP2K input file
    if structure_dir is not None:
        inp_files = list(Path(structure_dir).glob("*.inp"))
        if inp_files:
            inp_path = inp_files[0]
            if len(inp_files) > 1:
                logger.debug(
                    f"  Multiple *.inp files found in {structure_dir}; "
                    f"using {inp_path.name}"
                )
            indices = _parse_cp2k_fixed_atoms(inp_path)
            if indices:
                logger.info(
                    f"  Fixed atoms: {len(indices)} indices parsed from {inp_path.name}"
                )
                return indices
            else:
                logger.info(
                    f"  CP2K input {inp_path.name} found but no FIXED_ATOMS LIST — "
                    "no atoms will be fixed"
                )
        else:
            logger.info(f"  No *.inp file in {structure_dir} — no atoms will be fixed")
    else:
        logger.info("  No structure_dir provided — no atoms will be fixed")

    # Priority 3: nothing found
    return []


# ─── DFT data loading ─────────────────────────────────────────────────────────

def extract_cp2k_pdb_energy(pdb_path: Path) -> float:
    """Extract the final optimized energy from a CP2K PDB trajectory REMARK line."""
    last_energy = None
    with open(pdb_path) as f:
        for line in f:
            m = re.match(r"REMARK\s+Step\s+\d+,\s+E\s*=\s*([-\d.]+)", line)
            if m:
                last_energy = float(m.group(1))
    if last_energy is None:
        raise ValueError(f"No energy REMARK found in {pdb_path}")
    return last_energy * HA_TO_EV


def _find_pdb(directory: Path) -> Optional[Path]:
    """Return the first *-pos-1.pdb file in directory, or None."""
    files = list(directory.glob("*-pos-1.pdb"))
    return files[0] if files else None


def _resolve_structure_dir(top_dir: Path) -> Path:
    """
    Return the directory that directly contains a *-pos-1.pdb.
    If *top_dir* itself has one, return it.
    Otherwise descend one level into opt_* sub-directories.
    """
    if _find_pdb(top_dir) is not None:
        return top_dir
    for d in sorted(top_dir.glob("opt_*")):
        if d.is_dir() and _find_pdb(d) is not None:
            return d
    raise RuntimeError(f"No *-pos-1.pdb found under {top_dir}")


def load_adsorption_system(
    surface_dir: Path,
    gas_dir: Path,
    ads_dir: Path,
):
    """
    Load DFT structures and compute reference adsorption energies.

    Fixed atoms are determined per-directory (priority: ASE constraints in
    the PDB → CP2K *.inp in the same directory → no fixed atoms).

    The fixed-atom indices found for the slab are reused for all adsorption
    structures (they share the same surface).

    Parameters
    ----------
    surface_dir : Path
        Directory (or parent of opt_*) containing the clean slab *-pos-1.pdb
        and optionally a CP2K *.inp file.
    gas_dir : Path
        Directory (or parent of opt_*) containing the gas-molecule *-pos-1.pdb.
    ads_dir : Path
        Directory containing opt_* sub-directories, each with a *-pos-1.pdb
        and optionally a CP2K *.inp.

    Returns
    -------
    structures : dict
        "slab"       : (Atoms, float)
        "gas"        : (Atoms, float)
        "adsorption" : list of dicts {opt_dir, site, atoms, energy}
    dft_df : pd.DataFrame
        Columns: opt_dir, site, E_slab_ads_eV, E_slab_eV, E_gas_eV, E_ads_dft_eV
    """
    from ase.constraints import FixAtoms
    from ase.io import read as ase_read

    surface_dir = Path(surface_dir)
    gas_dir = Path(gas_dir)
    ads_dir = Path(ads_dir)

    # ── Load surface/slab ────────────────────────────────────────────────────
    slab_dir = _resolve_structure_dir(surface_dir)
    slab_pdb = _find_pdb(slab_dir)
    slab_energy = extract_cp2k_pdb_energy(slab_pdb)
    slab_atoms = ase_read(str(slab_pdb), index=-1)
    fixed_indices = get_fixed_atoms(slab_atoms, structure_dir=slab_dir)
    if fixed_indices:
        slab_atoms.set_constraint(FixAtoms(indices=fixed_indices))
    logger.info(
        f"Slab: {slab_atoms.get_chemical_formula()}, "
        f"{len(slab_atoms)} atoms, "
        f"{len(fixed_indices)} fixed, "
        f"cell={slab_atoms.get_cell().lengths().round(3)}, "
        f"E_DFT={slab_energy:.4f} eV"
    )

    # ── Load gas molecule ────────────────────────────────────────────────────
    gas_struct_dir = _resolve_structure_dir(gas_dir)
    gas_pdb = _find_pdb(gas_struct_dir)
    gas_energy = extract_cp2k_pdb_energy(gas_pdb)
    gas_atoms = ase_read(str(gas_pdb), index=-1)
    logger.info(
        f"Gas: {gas_atoms.get_chemical_formula()}, "
        f"E_DFT={gas_energy:.4f} eV"
    )

    # ── Load adsorption structures ───────────────────────────────────────────
    # Exclude the surface and gas directories in case ads_dir is their parent
    exclude = {slab_dir.resolve(), gas_struct_dir.resolve()}
    opt_dirs = sorted(
        [
            d for d in ads_dir.iterdir()
            if d.is_dir() and d.name.startswith("opt_")
            and d.resolve() not in exclude
        ],
        key=lambda d: int(d.name.split("_")[1]),
    )

    adsorption_configs = []
    for opt_dir in opt_dirs:
        pdb_file = _find_pdb(opt_dir)
        if pdb_file is None:
            logger.warning(f"No *-pos-1.pdb in {opt_dir.name}, skipping")
            continue
        energy = extract_cp2k_pdb_energy(pdb_file)
        atoms = ase_read(str(pdb_file), index=-1)
        # Apply the same fixed indices determined from the slab
        if fixed_indices:
            atoms.set_constraint(FixAtoms(indices=fixed_indices))
        idx = int(opt_dir.name.split("_")[1])
        site_label = f"site_{idx:02d}"
        adsorption_configs.append(
            {
                "opt_dir": opt_dir.name,
                "site": site_label,
                "atoms": atoms,
                "energy": energy,
            }
        )

    logger.info(f"Adsorption sites loaded: {len(adsorption_configs)}")

    structures = {
        "slab": (slab_atoms, slab_energy),
        "gas": (gas_atoms, gas_energy),
        "adsorption": adsorption_configs,
    }

    rows = []
    for cfg in adsorption_configs:
        E_ads = cfg["energy"] - slab_energy - gas_energy
        rows.append(
            {
                "opt_dir": cfg["opt_dir"],
                "site": cfg["site"],
                "E_slab_ads_eV": cfg["energy"],
                "E_slab_eV": slab_energy,
                "E_gas_eV": gas_energy,
                "E_ads_dft_eV": E_ads,
            }
        )
    dft_df = pd.DataFrame(rows)
    return structures, dft_df


# ─── MLFF benchmark ────────────────────────────────────────────────────────────

def _relax_atoms(atoms, fmax: float, label: str = "", max_steps: int = 500) -> None:
    """Run LBFGS relaxation in-place."""
    from ase.optimize import LBFGS

    try:
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
        logger.debug(f"  Relaxed {label}: {opt.nsteps} steps")
    except Exception as exc:
        logger.warning(f"  Relaxation failed for {label}: {exc}")


def run_adsorption_benchmark(
    calc,
    structures: dict,
    dft_df: pd.DataFrame,
    model_name: str = "mace",
    fmax: float = 0.05,
    slab_ref: str = "dft",
    gas_ref: str = "dft",
    ads_mode: str = "sp",
    e_ref_slab: float = 0.0,
    e_ref_gas: float = 0.0,
    pred_ads: bool = False,
) -> pd.DataFrame:
    """
    Evaluate MACE adsorption energies starting from DFT geometries.

    Parameters
    ----------
    calc : ASE calculator
    structures : dict  (from load_adsorption_system)
    dft_df : pd.DataFrame
    model_name : str  label for this model in the output CSV
    fmax : float  relaxation convergence (eV/Å), used when ads_mode='relax'
    slab_ref : 'mlff' or 'dft'
    gas_ref  : 'mlff' or 'dft'
    ads_mode : 'relax' or 'sp'
    e_ref_slab : float
        Slab reference energy subtracted during energy-referenced training (eV).
        When nonzero and slab_ref='dft', this offset is added back to the MLFF
        prediction of the adsorption structure so that DFT and MLFF energies are
        on the same absolute scale.  Default 0.0 (no correction).
    e_ref_gas : float
        Gas reference energy subtracted during energy-referenced training (eV).
        Applied analogously when gas_ref='dft'.  Default 0.0 (no correction).
    pred_ads : bool
        When True the model was trained to directly predict adsorption energies
        (REF_energy = E_total - E_slab_ref - E_gas_ref in the training data).
        In this mode E_ads_mlff = model.get_potential_energy(ads_structure)
        without any slab or gas subtraction; slab_ref, gas_ref, and e_ref_*
        are ignored.  Default False.
    """
    relax_ads = ads_mode == "relax"
    dft_lookup = dict(zip(dft_df["site"], dft_df["E_ads_dft_eV"]))
    rows = []

    if pred_ads:
        # ── pred_ads mode: model directly outputs E_ads ───────────────────────
        # Training target was REF_energy = E_total - E_slab_ref - E_gas_ref,
        # so get_potential_energy() already returns the adsorption energy.
        # No slab or gas MLFF evaluation is needed.
        for cfg in structures["adsorption"]:
            ads = cfg["atoms"].copy()
            ads.calc = calc
            if relax_ads:
                _relax_atoms(ads, fmax=fmax, label=cfg["opt_dir"])
            E_ads_mlff = ads.get_potential_energy()
            E_ads_dft = dft_lookup.get(cfg["site"], np.nan)
            error = E_ads_mlff - E_ads_dft if not np.isnan(E_ads_dft) else np.nan
            rows.append(
                {
                    "model": model_name,
                    "opt_dir": cfg["opt_dir"],
                    "site": cfg["site"],
                    "slab_ref": "pred_ads",
                    "gas_ref": "pred_ads",
                    "ads_mode": ads_mode,
                    "E_slab_eV": np.nan,
                    "E_gas_eV": np.nan,
                    "E_slab_ads_eV": np.nan,
                    "E_ads_eV": E_ads_mlff,
                    "E_ads_dft_eV": E_ads_dft,
                    "error_eV": error,
                }
            )
        return pd.DataFrame(rows)

    # ── Total-energy mode (pred_ads=False, default) ───────────────────────────
    relax_ref = ads_mode == "relax"

    # ── slab reference ────────────────────────────────────────────────────────
    slab_atoms, slab_energy_dft = structures["slab"]
    if slab_ref == "dft":
        E_slab = slab_energy_dft
    else:
        slab = slab_atoms.copy()
        slab.calc = calc
        if relax_ref:
            _relax_atoms(slab, fmax=fmax, label="slab")
        E_slab = slab.get_potential_energy()

    # ── gas reference ─────────────────────────────────────────────────────────
    gas_atoms, gas_energy_dft = structures["gas"]
    if gas_ref == "dft":
        E_gas = gas_energy_dft
    else:
        gas = gas_atoms.copy()
        gas.calc = calc
        if relax_ref:
            _relax_atoms(gas, fmax=fmax, label="gas")
        E_gas = gas.get_potential_energy()

    # ── adsorption structures ─────────────────────────────────────────────────
    for cfg in structures["adsorption"]:
        ads = cfg["atoms"].copy()
        ads.calc = calc
        if relax_ads:
            _relax_atoms(ads, fmax=fmax, label=cfg["opt_dir"])
        E_ads_atoms = ads.get_potential_energy()
        # Energy-referenced training correction:
        # If the model was trained with shifted targets (e.g. REF_energy = E_total - e_ref_slab - e_ref_gas),
        # its output is on an E_ads scale rather than absolute total energy.
        # When mixing with DFT references (absolute scale), add back the offsets so units match.
        e_offset = (e_ref_slab if slab_ref == "dft" else 0.0) + (e_ref_gas if gas_ref == "dft" else 0.0)
        E_ads_mlff = (E_ads_atoms + e_offset) - E_slab - E_gas
        E_ads_dft = dft_lookup.get(cfg["site"], np.nan)
        error = E_ads_mlff - E_ads_dft if not np.isnan(E_ads_dft) else np.nan
        rows.append(
            {
                "model": model_name,
                "opt_dir": cfg["opt_dir"],
                "site": cfg["site"],
                "slab_ref": slab_ref,
                "gas_ref": gas_ref,
                "ads_mode": ads_mode,
                "E_slab_eV": E_slab,
                "E_gas_eV": E_gas,
                "E_slab_ads_eV": E_ads_atoms,
                "E_ads_eV": E_ads_mlff,
                "E_ads_dft_eV": E_ads_dft,
                "error_eV": error,
            }
        )
    return pd.DataFrame(rows)


def save_parity_plot(df: pd.DataFrame, plot_path: Path) -> None:
    """Save a parity plot (MLFF vs DFT adsorption energies) to *plot_path*."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = df.dropna(subset=["error_eV"])
        if len(valid) == 0:
            return

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            valid["E_ads_dft_eV"],
            valid["E_ads_eV"],
            s=40,
            alpha=0.8,
            label=df["model"].iloc[0],
        )
        lims = [
            min(valid["E_ads_dft_eV"].min(), valid["E_ads_eV"].min()) - 0.05,
            max(valid["E_ads_dft_eV"].max(), valid["E_ads_eV"].max()) + 0.05,
        ]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("DFT $E_{ads}$ (eV)")
        ax.set_ylabel("MACE $E_{ads}$ (eV)")
        mae = valid["error_eV"].abs().mean()
        rmse = np.sqrt((valid["error_eV"] ** 2).mean())
        ax.set_title(f"MAE={mae:.3f} eV  RMSE={rmse:.3f} eV")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=150)
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Could not save parity plot: {exc}")


def save_multi_system_parity_plot(
    system_dfs: List[Tuple[str, pd.DataFrame]],
    plot_path: Path,
    combo_tag: str = "",
) -> None:
    """
    Save a parity plot with one colour per system (surface/adsorbate combination).

    Parameters
    ----------
    system_dfs : list of (system_name, DataFrame) tuples
        Each DataFrame must have columns ``E_ads_dft_eV``, ``E_ads_eV``, ``error_eV``.
    plot_path : Path
        Output PNG path.
    combo_tag : str
        Short description added to the plot title (e.g. ``slab-dft_gas-dft_sp``).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        all_vals = []
        for sys_name, df in system_dfs:
            valid = df.dropna(subset=["error_eV"])
            if len(valid) == 0:
                continue
            sc = ax.scatter(
                valid["E_ads_dft_eV"],
                valid["E_ads_eV"],
                s=40,
                alpha=0.8,
                label=sys_name,
            )
            all_vals.extend(valid["E_ads_dft_eV"].tolist())
            all_vals.extend(valid["E_ads_eV"].tolist())

        if not all_vals:
            plt.close(fig)
            return

        lims = [min(all_vals) - 0.05, max(all_vals) + 0.05]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("DFT $E_{ads}$ (eV)")
        ax.set_ylabel("MACE $E_{ads}$ (eV)")

        # Overall MAE/RMSE across all systems
        all_errors = pd.concat(
            [df.dropna(subset=["error_eV"])["error_eV"] for _, df in system_dfs],
            ignore_index=True,
        )
        if len(all_errors) > 0:
            mae = all_errors.abs().mean()
            rmse = np.sqrt((all_errors ** 2).mean())
            title = f"{combo_tag}  MAE={mae:.3f} eV  RMSE={rmse:.3f} eV"
        else:
            title = combo_tag
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=150)
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Could not save multi-system parity plot: {exc}")


# ─── Callback factory ─────────────────────────────────────────────────────────

def _safe_dirname(name: str) -> str:
    """Convert a system name like 'CO/100' to a filesystem-safe string 'CO_100'."""
    return re.sub(r"[^\w\-]", "_", name)


def make_adsorption_benchmark_fn(
    output_dir: str,
    systems: Optional[List[Dict]] = None,
    surface_dir: Optional[str] = None,
    gas_dir: Optional[str] = None,
    ads_dir: Optional[str] = None,
    device: str = "cpu",
    fmax: float = 0.05,
    e_ref_slab: float = 0.0,
    e_ref_gas: float = 0.0,
    pred_ads: bool = False,
):
    """
    Build and return a callback ``fn(epoch, model)`` for
    :func:`mace.tools.train.train` (*adsorption_benchmark_fn*).

    Supports two calling modes:

    **Multi-system mode** (recommended)::

        systems = [
            {"name": "CO/100", "surface_dir": "...", "gas_dir": "...", "ads_dir": "..."},
            {"name": "CO/111", "surface_dir": "...", "gas_dir": "...", "ads_dir": "..."},
            {"name": "CO2/100", "surface_dir": "...", "gas_dir": "...", "ads_dir": "..."},
        ]
        fn = make_adsorption_benchmark_fn(systems=systems, output_dir="...")

    **Single-system mode** (backward-compatible)::

        fn = make_adsorption_benchmark_fn(
            surface_dir="...", gas_dir="...", ads_dir="...", output_dir="..."
        )

    Each system dict must contain:
    - ``name``: human-readable label (used in logs, filenames, and plot legends)
    - ``surface_dir``: directory with the clean slab CP2K output (*-pos-1.pdb)
    - ``gas_dir``: directory with the gas-molecule CP2K output
    - ``ads_dir``: directory containing opt_* sub-directories

    Runs **all 8 combinations** of reference energy source × ads mode per system:
      slab_ref ∈ {mlff, dft}  ×  gas_ref ∈ {mlff, dft}  ×  ads_mode ∈ {sp, relax}

    Per epoch, per system, each combination produces:
    - ``{output_dir}/{system_name}/adsorption_benchmark_epoch{N}_{combo_tag}.csv``
    - ``{output_dir}/{system_name}/adsorption_benchmark_epoch{N}_{combo_tag}.png``
    Per system, all combos are concatenated into ``adsorption_benchmark_epoch{N}_all.csv``.
    A cross-system parity plot (one colour per system) is saved to the root output_dir
    for the ``slab-dft_gas-dft_sp`` combination.
    """
    import torch
    from itertools import product as iterproduct
    from mace.calculators.mace import MACECalculator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Normalise to a list of system dicts ──────────────────────────────────
    if systems is None:
        if surface_dir is None or gas_dir is None or ads_dir is None:
            logger.error(
                "AdsorptionBenchmark: either 'systems' or all of "
                "'surface_dir', 'gas_dir', 'ads_dir' must be provided."
            )
            return None
        systems = [
            {
                "name": "default",
                "surface_dir": surface_dir,
                "gas_dir": gas_dir,
                "ads_dir": ads_dir,
            }
        ]

    # ── Load DFT reference data once at construction time ───────────────────
    logger.info(
        f"AdsorptionBenchmark: loading DFT reference structures for "
        f"{len(systems)} system(s) …"
    )
    loaded: List[Dict] = []  # [{name, structures, dft_df, out_dir}, ...]
    for sys_def in systems:
        name = sys_def.get("name", "unknown")
        try:
            structures, dft_df = load_adsorption_system(
                surface_dir=Path(sys_def["surface_dir"]),
                gas_dir=Path(sys_def["gas_dir"]),
                ads_dir=Path(sys_def["ads_dir"]),
            )
        except Exception as exc:
            logger.error(
                f"  [{name}] Failed to load reference data: {exc} — skipping this system."
            )
            continue
        n_sites = len(structures["adsorption"])
        sys_out = output_dir / _safe_dirname(name)
        sys_out.mkdir(parents=True, exist_ok=True)
        loaded.append(
            {
                "name": name,
                "structures": structures,
                "dft_df": dft_df,
                "out_dir": sys_out,
                "n_sites": n_sites,
            }
        )
        logger.info(f"  [{name}] {n_sites} adsorption configs loaded → {sys_out}")

    if not loaded:
        logger.error("AdsorptionBenchmark: no systems loaded. Benchmark will be disabled.")
        return None

    # Benchmark combinations:
    # pred_ads=True  → model directly outputs E_ads, only sp/relax vary (2 combos)
    # pred_ads=False → model outputs total energy, all 8 combinations apply
    if pred_ads:
        ALL_COMBOS = [("pred_ads", "pred_ads", mode) for mode in ["sp", "relax"]]
    else:
        ALL_COMBOS = list(iterproduct(["mlff", "dft"], ["mlff", "dft"], ["sp", "relax"]))

    def _benchmark_fn(epoch: int, model: "torch.nn.Module") -> None:
        logging.info(f"\n{'═'*65}")
        logging.info(
            f"AdsorptionBenchmark — Epoch {epoch}  "
            f"({len(loaded)} system(s), {len(ALL_COMBOS)} combos each)"
        )

        # Build MACECalculator from current in-memory model weights
        try:
            calc = MACECalculator(
                models=deepcopy(model).to("cpu"),
                device=device,
                default_dtype="float64",
            )
        except Exception as exc:
            logger.warning(f"  Could not create calculator: {exc}")
            return

        # combo_tag → list of (system_name, df) for cross-system plots
        combo_system_dfs: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}

        for sys_info in loaded:
            sys_name = sys_info["name"]
            structures = sys_info["structures"]
            dft_df = sys_info["dft_df"]
            sys_out = sys_info["out_dir"]

            logging.info(f"\n  ── System: {sys_name} ──")
            sys_dfs = []

            for slab_ref, gas_ref, ads_mode in ALL_COMBOS:
                if slab_ref == "pred_ads":
                    combo_tag = f"pred_ads_{ads_mode}"
                else:
                    combo_tag = f"slab-{slab_ref}_gas-{gas_ref}_{ads_mode}"
                try:
                    df = run_adsorption_benchmark(
                        calc=calc,
                        structures=structures,
                        dft_df=dft_df,
                        model_name=f"mace_epoch{epoch}_{combo_tag}",
                        fmax=fmax,
                        slab_ref=slab_ref,
                        gas_ref=gas_ref,
                        ads_mode=ads_mode,
                        e_ref_slab=e_ref_slab,
                        e_ref_gas=e_ref_gas,
                        pred_ads=pred_ads,
                    )
                except Exception as exc:
                    logger.warning(f"    [{sys_name}] Combo {combo_tag} failed: {exc}")
                    continue

                df["system"] = sys_name

                valid = df.dropna(subset=["error_eV"])
                if len(valid) > 0:
                    mae = valid["error_eV"].abs().mean()
                    rmse = np.sqrt((valid["error_eV"] ** 2).mean())
                    max_err = valid["error_eV"].abs().max()
                    logging.info(
                        f"    [{combo_tag}]  "
                        f"MAE={mae:.3f} eV  RMSE={rmse:.3f} eV  Max={max_err:.3f} eV"
                    )

                # Per-system per-combo CSV and plot
                csv_path = sys_out / f"adsorption_benchmark_epoch{epoch:04d}_{combo_tag}.csv"
                df.to_csv(csv_path, index=False)
                plot_path = sys_out / f"adsorption_benchmark_epoch{epoch:04d}_{combo_tag}.png"
                save_parity_plot(df, plot_path)

                sys_dfs.append(df)
                combo_system_dfs.setdefault(combo_tag, []).append((sys_name, df))

            if not sys_dfs:
                continue

            # Per-system combined CSV
            sys_combined = pd.concat(sys_dfs, ignore_index=True)
            sys_combined.to_csv(
                sys_out / f"adsorption_benchmark_epoch{epoch:04d}_all.csv", index=False
            )

        if not combo_system_dfs:
            return

        # ── Cross-system summary ─────────────────────────────────────────────
        logging.info(f"\n  ── Epoch {epoch} Cross-System Summary ──")
        logging.info(f"  {'system':<20}  {'combo':<35}  {'MAE':>7}  {'RMSE':>7}")
        logging.info("  " + "-" * 70)

        all_epoch_dfs = []
        for combo_tag, sys_df_pairs in sorted(combo_system_dfs.items()):
            for sys_name, df in sys_df_pairs:
                valid = df.dropna(subset=["error_eV"])
                if len(valid) > 0:
                    mae = valid["error_eV"].abs().mean()
                    rmse = np.sqrt((valid["error_eV"] ** 2).mean())
                    logging.info(
                        f"  {sys_name:<20}  {combo_tag:<35}  {mae:>7.3f}  {rmse:>7.3f}"
                    )
                all_epoch_dfs.append(df)

            # Cross-system parity plot per combo
            cross_plot = output_dir / f"adsorption_benchmark_epoch{epoch:04d}_{combo_tag}_all_systems.png"
            save_multi_system_parity_plot(sys_df_pairs, cross_plot, combo_tag=combo_tag)

        # Grand combined CSV (all systems, all combos)
        if all_epoch_dfs:
            grand = pd.concat(all_epoch_dfs, ignore_index=True)
            grand.to_csv(
                output_dir / f"adsorption_benchmark_epoch{epoch:04d}_all.csv", index=False
            )
            logging.info(
                f"  Grand combined CSV: adsorption_benchmark_epoch{epoch:04d}_all.csv"
            )

    return _benchmark_fn
