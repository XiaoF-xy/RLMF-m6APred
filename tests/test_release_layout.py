from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("H_b", "H_k", "H_l", "M_b", "M_h", "M_k", "M_l", "M_t", "R_b", "R_k", "R_l")


def test_release_has_all_fold_weights():
    paths = list((ROOT / "weights").glob("*/fold_*.pt"))
    assert len(paths) == 55


def test_internal_data_roles_are_separate():
    for dataset in DATASETS:
        assert (ROOT / "data" / "internal_benchmark" / f"{dataset}.csv").is_file()
        assert (ROOT / "data" / "independent_test" / f"{dataset}.csv").is_file()


def test_cross_species_files_preserve_public_names():
    root = ROOT / "data" / "external_validation" / "dsnm6a" / "cross_species_5x5"
    assert (root / "zebrafish.csv").is_file()
    assert (root / "arabidopsis.csv").is_file()
