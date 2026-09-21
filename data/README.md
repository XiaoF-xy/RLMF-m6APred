# Released data

| Directory | Role |
| --- | --- |
| `internal_benchmark/` | Eleven benchmark sequence tables used in the manuscript |
| `independent_test/` | Eleven fixed independent-test tables used by `evaluate_independent.py` |
| `external_validation/dsnm6a/mammalian_mrna/` | Six Human/Mouse/Rat external tables: ACA and DRACH |
| `external_validation/dsnm6a/cross_species_5x5/` | Exact Zebrafish and Arabidopsis species pools used in the 5x5 experiment |

Every CSV contains `sequence` and `label`. Cross-species files additionally
preserve the original `dataset_ids` column. All sequences are 41 nt long.

The release does not contain few-shot splits, domain-adaptation corpora,
interpretability subsets, or intermediate derived datasets.
