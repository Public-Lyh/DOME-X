# DOME-X supplementary code

Each dataset directory contains the main experiment as
`DOME-X4{dataset}.py`. Component studies are under `ablation/`, and matched
baselines are under `comparison/`. The controlled channel experiment is in
`Synthetic/`; its leave-one rescue study, which generates the controlled ROST
results, is in `Synthetic/ablation/`.

Replace `Path("your path")` with the local project root before running a
dataset experiment. Scripts write checkpoints and reports beneath the
corresponding dataset directory in `Code/`. The synthetic experiments are
self-contained and write their reports beside the copied scripts.
