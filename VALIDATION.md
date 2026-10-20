# Public release validation

20 October 2026; Windows, Python 3.11.

- Installed `requirements.txt` in a new isolated virtual environment.
- `python -m pip check`: no broken requirements.
- `python examples/synthetic.py`: ten releases, six scored weight fits,
  no solver failures; exact envelope and final remaining-allowance checks pass.
  Coherence is checked at floating-point tolerance.
- `python cpor.py check-results --evidence evidence`: reported group averages
  and the four-arm ablation table are reconstructed successfully.
- All 21 archived numerical Python files match their supplied SHA-256 hashes.

This validates installation and a synthetic online sequence. It does not
rerun the empirical benchmark, reproduce neural backbone training, or establish
bitwise equality between the historical and public dependency environments.
