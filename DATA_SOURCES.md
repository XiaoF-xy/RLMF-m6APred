# Data sources

## Internal benchmark and independent-test data

The release preserves the exact 41-nt sequence tables used for the 11 tissue
datasets in the manuscript. Benchmark and independent-test files are separated
under `data/internal_benchmark/` and `data/independent_test/`.

## DSNm6A mammalian external validation

Human, mouse, and rat mRNA sequences originate from the DSNm6A data release.
The released RLMF-m6APred evaluation files are the exact 41-nt processed tables
used in the manuscript. They are separated by species and by ACA/DRACH protocol.
They were used only for external evaluation.

Original project: https://github.com/BioMLab/DSNm6A

## DSNm6A cross-species 5x5 data

`zebrafish.csv` and `arabidopsis.csv` are byte-identical copies of the species
pools used in the RLMF-m6APred 5x5 cross-species experiment. Both are 41-nt,
DRACH-only datasets. No additional filtering, deduplication, rebalancing,
column renaming, or row reordering was performed for this release.

The original source repository does not currently display an explicit data
license. Confirm redistribution permission before making the public release.
