# SeniorProject

This repository contains tools and models for integrated VAD (Valence-Arousal-Dominance) processing, training a VAD regressor, and a GRU encoder used for emotion modeling. The final step is running the visualization script `run_visual.sh`.

## Prerequisites
- Python 3.8+ recommended
- Git
- Bash (for provided shell scripts). On Windows use WSL, Git Bash, or Windows Terminal with WSL.

## Clone
Replace <repo-url> with this repository's URL:

```bash
git clone <repo-url> SeniorProject
cd SeniorProject
```

## Environment setup
Windows (recommended PowerShell) — create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1   # PowerShell
# or
venv\Scripts\activate.bat    # cmd.exe
```

Unix / WSL / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 1) Build integrated VAD dataset
The repository includes `build_integrated_vad_dataset.py` which constructs/merges datasets used for training.

Run:

```bash
python build_integrated_vad_dataset.py
```

Outputs are placed under `integrated_vad_utterance_dataset/` or similar folders. Inspect the script if custom input/output paths are required.

## 2) Train VAD regressor
A training script and helper shell exist:

- Script: `train_vad_regressor.py`
- Shell helper: `run_vad_regressor.sh`

Run via Bash (recommended):

```bash
bash run_vad_regressor.sh
# or directly
python train_vad_regressor.py
```

Models and logs are saved to directories like `vad_deberta_v3_regressor/` or `outputs/` — check the script for exact paths.

## 3) Train GRU encoder
There is a GRU encoder implementation: `GRU_full_encoder.py` and helper shell `run_GRU_encoder.sh`.

Run:

```bash
bash run_GRU_encoder.sh
# or
python GRU_full_encoder.py
```

Check `outputs/` and `emotion_gru_runs/` for saved checkpoints and logs.

## 4) Final: Visualization
The goal is to run the provided visualization script which expects trained models and built dataset.

```bash
bash run_visual.sh
```

On Windows run this from WSL or Git Bash. The script produces demo HTML/visuals (see `reveal_demo.html`, `co_emp_reveal_presentation*.html`) and logs inside `outputs/`.

## Notes
- If a script accepts command-line arguments (data paths, epochs, device), inspect the top of the Python file or the shell script to tune parameters.
- Large datasets and model training require sufficient disk space and GPU when applicable. Activate GPU environment (e.g., CUDA) before training if available.
- If running on Windows without Bash, convert the .sh steps to equivalent PowerShell commands or use WSL.

## Troubleshooting
- Missing modules: re-run `pip install -r requirements.txt` and verify Python version.
- Permission errors running bash scripts on Windows: use WSL or Git Bash.

## Contact
For questions, open an issue or contact the project maintainer.
