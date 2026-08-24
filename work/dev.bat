@echo off
cd /d "%~dp0..\model-dev\learning"
call conda activate dev

set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "OPENBLAS_NUM_THREADS=2"

python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --diagnostics-config configs\diagnostics\vafidis_diagnostics.yaml --profile configs\profiles\block_multirate.yaml
