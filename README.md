# Time in IoT language-model agents: stale readings and skewed clocks

Code, probe corpora, per-probe model outputs and analysis results for two articles by Abdelbassette Chenna
(LASTIC Laboratory, Université de Batna 2, Algeria):

- **AGE**: *Look Again or Act? Stale Readings and the Re-Read Decisions of Small Open Language-Model Agents in the
  Internet of Things.* When should an agent that holds a remembered device reading pay to read the device again
  before acting? Corpus: **ReRead-IoT**.
- **ORDER**: *Arrived Before Is Not Happened Before: Clock Skew, Root-Cause Analysis of IoT Incidents and Small Open
  Language Models.* How much can skewed device timestamps say about which event caused the other, and do language
  models use that evidence? Corpus: **SkewRCA**.

Everything reported in both articles can be recomputed from this repository on a laptop CPU in a few minutes
(`python reproduce.py`). The only GPU step, running the language models on the probes, has its outputs released in
`runs/`, and the scripts that produced them are included.

## Contents

| Path | What it holds |
|---|---|
| `age/code`, `order/code` | corpus generators, metrics, analyses, reference bounds, numerical checks of the formal results, table and figure scripts |
| `age/results`, `order/results` | the probe corpora (`probes.jsonl` and the validation and revision subsets) and every analysis output (JSON) |
| `age/tables`, `order/tables` | the result tables of each article, as generated (Markdown) |
| `age/figures`, `order/figures` | the figures of each article (PNG, 300 dpi) |
| `runs/` | per-probe outputs of every model: option-letter probabilities, generated texts, tool calls and forced answers, with a `meta__*.json` file recording the model identifier and the run plan |
| `realdata/` | measured staleness curves and event delays derived from the public CASAS, UCI occupancy and HAI datasets (see below) |
| `shared/` | the GPU evaluation harness (`probe_runner.py`), the Kaggle launcher (`push.py`, `queue.sh`), the forced-answer builder, the real-data extraction scripts and the figure style |
| `expected/` | the manuscript values (`*_fills.json`) that `reproduce.py` checks against |
| `reproduce.py` | recomputes every table and manuscript value from `runs/` and compares them with the released ones |

## Reproducing the numbers

```bash
pip install -r requirements.txt
python reproduce.py          # full check: theory verifications, Monte Carlo simulations, all analyses
python reproduce.py --quick  # analyses and tables only (about one minute)
```

The script works in a temporary copy of the results folders, regenerates all analyses and tables, and exits with a
non-zero status if any table or any value quoted in the manuscripts differs from the released one. It was run
successfully on Python 3.12 with the package versions in `requirements.txt`.

Individual steps can be run by hand, for example:

```bash
python age/code/analysis_age_v2.py runs age/results
python age/code/make_tables_res.py age/results age/tables <out_dir> realdata/realdata_age_v2.json
python order/code/analysis_order_v2.py runs order/results
python order/code/misspec_stress.py order/results/misspec_stress.json
python order/code/verify_theory_v3.py order/results/theory_checks_v3.json
```

## Re-running the models

The models were run on a single 16 GB T4 GPU (Kaggle) in 4-bit NF4 quantisation through `shared/probe_runner.py`:
Qwen2.5-7B-Instruct, Qwen3-8B (non-thinking mode), Llama-3.1-8B-Instruct, Granite-3.3-8B-Instruct,
OLMo-2-1124-7B-Instruct and, as a scale check, Qwen2.5-14B-Instruct. Letter probes are scored from the next-token
distribution after the chat template; the ORDER forced-choice parts are scored with the assistant turn prefilled
with `Answer:`. Reasoning arms are generated greedily with a budget of 512 new tokens; when the budget runs out before
a final answer, `shared/make_force.py` builds a second pass that appends `Answer:` to the model's own reasoning and
reads the letter distribution.

`shared/push.py` packages the corpora as a Kaggle dataset (`python shared/push.py dataset`) and pushes one kernel
per model and plan (`python shared/push.py run <slug> <model> <plan>`, plans `full`, `prefix`, `rev`, `big`,
`force`). Set `USER` and `DS` at the top of `push.py` to your own Kaggle account and dataset before use.

## Real-trace data

`realdata/` contains quantities derived from three public datasets: device staleness curves and fitted
parameters from CASAS smart homes and the UCI occupancy recordings, and delays between related events from CASAS
(door opening to the next motion event) and from the HAI industrial control testbed (controller demand step to
actuator half response). The original recordings are not redistributed here; they must be obtained from their
sources under their own terms, and the derived quantities remain subject to those terms. The scripts that compute
them (`shared/realdata.py`, `shared/realdata_v2.py`, `shared/hai_latency.py`) run on Kaggle with the datasets
attached.

- CASAS: D. J. Cook et al., "CASAS: A Smart Home in a Box", *Computer*, 2013.
- UCI occupancy: L. M. Candanedo and V. Feldheim, *Energy and Buildings*, 2016.
- HAI: H.-K. Shin et al., "HAI 1.0: HIL-based Augmented ICS Security Dataset", USENIX CSET 2020.

## Licences

Code is released under the MIT licence (`LICENSE`). The probe corpora, model outputs, analysis results, tables and
figures are released under CC BY 4.0 (`LICENSE-DATA.md`). Model outputs were produced with publicly released open
models and are subject to the terms of those models.

## Citation

If you use the corpora or the code, please cite the corresponding article (details to be added on publication)
and this repository.
