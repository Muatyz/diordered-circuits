@echo off
cd /d "%~dp0..\model-dev\learning"
call conda activate dev
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --diagnostics-config configs\diagnostics\vafidis_diagnostics.yaml --profile configs\profiles\block_multirate.yaml
