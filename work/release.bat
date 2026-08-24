@echo off
cd /d "%~dp0..\model-release\learning"
call conda activate pro

set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "OPENBLAS_NUM_THREADS=2"

python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy_seed44.yaml --diagnostics-config configs\diagnostics\vafidis_diagnostics.yaml --profile configs\profiles\block_multirate.yaml
