from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_dataset(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    endpoints = np.array(["fish_96h_lc50", "daphnia_48h_ec50", "algae_72h_ec50"])
    chem_classes = np.array(
        [
            "Pharmaceutical Personal Care Products (PPCPs)",
            "Per- and Polyfluoroalkyl Substances (PFAS)",
            "Conazoles",
            "Neonicotinoids",
            "Strobins",
        ]
    )
    species = {
        "fish_96h_lc50": ("Oncorhynchus mykiss", "Pimephales promelas", "Danio rerio"),
        "daphnia_48h_ec50": ("Daphnia magna", "Ceriodaphnia dubia", "Moina macrocopa"),
        "algae_72h_ec50": ("Pseudokirchneriella subcapitata", "Chlorella vulgaris", "Raphidocelis subcapitata"),
    }
    smiles_pool = np.array(
        [
            "CCO",
            "CCN",
            "CC(=O)O",
            "CCOC(=O)C",
            "c1ccccc1",
            "C1CCCCC1",
            "c1ccncc1",
            "c1ncc[nH]1",
            "c1ccoc1",
            "c1ccsc1",
            "C1CCCC1",
            "C1CCC1",
            "C1CC1",
            "O1CCCCC1",
            "N1CCCCC1",
            "O1CCOCC1",
            "N1CCNCC1",
            "c1ccc2ccccc2c1",
            "c1ccc2[nH]ccc2c1",
            "c1ccc2ncccc2c1",
            "C1CCC2CCCCC2C1",
            "c1ccc2occc2c1",
            "c1ccc2sccc2c1",
            "C1COCCN1",
        ]
    )
    rows = []
    for idx in range(n):
        endpoint = str(rng.choice(endpoints))
        chem_class = str(rng.choice(chem_classes, p=[0.28, 0.14, 0.18, 0.2, 0.2]))
        species_name = str(rng.choice(species[endpoint]))
        genus = species_name.split()[0]
        mw = rng.normal(250, 80)
        logp = rng.normal(3.5, 1.2)
        tpsa = rng.normal(60, 20)
        duration = {"fish_96h_lc50": 96, "daphnia_48h_ec50": 48, "algae_72h_ec50": 72}[endpoint]
        year = int(rng.integers(1995, 2025))
        mech_signal = rng.normal(0, 1)
        class_offset = {
            "Pharmaceutical Personal Care Products (PPCPs)": -0.3,
            "Per- and Polyfluoroalkyl Substances (PFAS)": 0.5,
            "Conazoles": 0.8,
            "Neonicotinoids": 0.2,
            "Strobins": -0.1,
        }[chem_class]
        species_offset = 0.4 if "Daphnia" in species_name else -0.2 if "Oncorhynchus" in species_name else 0.1
        context_shift = 0.3 if year >= 2020 else 0.0
        pfas_class = "Per- and Polyfluoroalkyl Substances (PFAS)"
        deterministic_rejection = (
            chem_class == pfas_class
            and year >= 2022
            and endpoint == "fish_96h_lc50"
        )
        target = (
            -2.0
            + 0.25 * logp
            - 0.004 * mw
            + 0.006 * tpsa
            + 0.5 * mech_signal
            + class_offset
            + species_offset
            + context_shift
            + rng.normal(0, 0.35 if not deterministic_rejection else 0.7)
        )
        rows.append(
            {
                "chemical_id": f"CHEM_{idx:05d}",
                "chemical_name": f"Demo chemical {idx}",
                "casrn": f"{10000 + idx}-00-0",
                "smiles": str(rng.choice(smiles_pool)),
                "endpoint": endpoint,
                "chemical_class": chem_class,
                "species": species_name,
                "genus": genus,
                "family": "DemoFamily",
                "order": "DemoOrder",
                "class_name": "DemoClass",
                "phylum": "Chordata" if endpoint == "fish_96h_lc50" else "Arthropoda" if endpoint == "daphnia_48h_ec50" else "Chlorophyta",
                "trophic_group": "producer" if endpoint == "algae_72h_ec50" else "consumer",
                "duration_h": duration,
                "medium": "freshwater",
                "temperature_c": float(rng.normal(20, 2)),
                "ph": float(rng.normal(7.5, 0.4)),
                "study_year": year,
                "source": "demo",
                "effect": "growth" if endpoint == "algae_72h_ec50" else "mortality",
                "toxicity_value": 10 ** target,
                "toxicity_unit": "mmol/L",
                "target_log_molar": target,
                "physchem_mol_wt": mw,
                "physchem_logp": logp,
                "physchem_tpsa": tpsa,
                "physchem_hba": max(0.0, rng.normal(4, 2)),
                "physchem_hbd": max(0.0, rng.normal(1, 0.8)),
                "mech_hit_rate": mech_signal,
                "mech_stress_response": rng.normal(0, 1),
                "ctx_lab_id": float(rng.integers(1, 6)),
                "is_hard_ood": deterministic_rejection,
                "known_ood": (
                    deterministic_rejection
                    or year >= 2020
                    or chem_class == pfas_class
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a synthetic EcoOOD demo dataset.")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/processed/demo_ecoood.csv"))
    args = parser.parse_args()

    df = make_demo_dataset(n=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
