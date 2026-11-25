# WhereMoves

Static research project page at https://reachomk.github.io/wheremoves/.
No build step or server inference is required.

## Preview and verify

From the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
# Open http://127.0.0.1:8765/wheremoves/
node wheremoves/scripts/verify.mjs
```

The verifier checks all prediction maps, image hashes, 16-step query coverage,
trajectory upsampling, and the 44 measured performance-plot configurations.

## Assets

The first demo uses eight ranked successful K=4 policy rollouts, with native
2048×2048 images at every recorded prediction step. Changing K shows alternative
selector outputs on those same recordings. Prediction coordinates remain on the
original 128×128 model grid. Full-rollout videos and raw model archives are not
included in the website.

Data provenance, source hashes, ranking details, and rendering comparisons are
retained in `data/groups-provenance.json`, `data/policy-rollouts.json`,
`data/image-rendering.json`, and `data/verification.json`. Performance data and
paper timing overrides are recorded in `data/efficiency.json`.

To regenerate browser assets, run from this directory with NumPy and Pillow:

```sh
python3 scripts/prepare_assets.py --downloads /path/to/Downloads --robocasa-source /path/to/wheremoves-site-exports-20261123
python3 scripts/prepare_highres.py --source /path/to/wheremoves-highres-trajectory-20261124
python3 scripts/prepare_policy_rollouts.py --source /path/to/wheremoves-policy-rollouts-20261124
python3 scripts/prepare_results.py --source /path/to/wheremoves-site-exports-20261123/results
```

The results script additionally requires PyMuPDF and uses `assets/figures/efficiency.pdf`
to calibrate the interactive markers. `policy_rollouts.py` ranks and exports source
recordings in the RoboCasa repository's flow-rollout environment;
`render_scene_images.py` renders the separate trajectory demo's source images.
These source tools require that repository's datasets, checkpoints, and simulator
assets. Neither runs the action policy during export.

The arXiv, Code, and Models links remain marked “soon” until final URLs are
available. Update the BibTeX entry when publication details are confirmed.
Typography is inspired by JtT, and the narrative organization by Cues.
The Latin Modern font attribution and license are in `assets/fonts/`.
