"""GPU side: the Modal image and training runs.

GRAFT's RL code needs Python 3.11, torch 2.10, TRL 0.29 and a patched vLLM
0.17 (see ``data/raw/GRAFT/rl/DEPENDENCIES.md``). The local
environment is Python 3.13, so the image mirrors GRAFT's own
(``rl/tools/modal_train_gr.py``) and the Modal functions here are ordinary
ones that the container imports by module. Keep
this file importable on Python 3.11 (no PEP 695 syntax at module level).

Weights and run outputs live on the Modal volume ``graft-absorption`` (override
with ``MODAL_VOLUME``). The HF token and W&B key are attached per run as
ephemeral secrets (``credentials.py``); nothing is stored in Modal.

    from gpu import *

    smoke()                                        # Gate 1: image builds, train.py imports
    run = train_run(GATE1_SWEEP, GATE1_RUN, overrides={"max_steps": 30})
"""

import importlib.metadata
import os
import shutil
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
GRAFT_RL_DIR = ROOT / "data" / "raw" / "GRAFT" / "rl"

VOLUME_NAME = os.environ.get("MODAL_VOLUME", "graft-absorption")
VOLUME_PATH = "/vol"
RUNS_ROOT = f"{VOLUME_PATH}/runs"
REPO_REMOTE = "/repo"  # GRAFT's rl/, the working directory of every run
RUN_RESULT = "run_result.json"  # train_run's return value, minus the eval rows

if modal.is_local():
    from credentials import hf_secret, wandb_secret

    SECRETS = [hf_secret(), wandb_secret()]
    # The container's Modal client must match the local one.
    MODAL_PIN = f"modal=={importlib.metadata.version('modal')}"
else:
    # Modal matches a function's dependencies by position, so the container
    # must declare as many secrets as the local side. Their values come from
    # the object ids Modal sends, not from these placeholders.
    SECRETS = [modal.Secret.from_dict({}), modal.Secret.from_dict({})]
    MODAL_PIN = "modal"

# GRAFT's freeze (requirements-modal.txt) installed with --no-deps, minus its
# own Modal pin, then the three vLLM patches the code depends on at runtime.
VENV_SITE = "/build/.venv/lib/python3.11/site-packages/vllm"
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .workdir("/build")
    .add_local_file(GRAFT_RL_DIR / "requirements-modal.txt", "/build/requirements-modal.txt", copy=True)
    .add_local_dir(GRAFT_RL_DIR / "vllm_patches", "/build/vllm_patches", copy=True)
    .run_commands(
        "uv venv --python 3.11 --seed",
        "grep -v '^modal==' /build/requirements-modal.txt > /build/requirements.txt",
        "uv pip install --python /build/.venv/bin/python --no-deps -r /build/requirements.txt",
        f"uv pip install --python /build/.venv/bin/python '{MODAL_PIN}'",
        f"cp /build/vllm_patches/model_manager.py {VENV_SITE}/lora/model_manager.py",
        f"cp /build/vllm_patches/qwen3_5_config.py {VENV_SITE}/transformers_utils/configs/qwen3_5.py",
        f"cp /build/vllm_patches/gpu_worker.py {VENV_SITE}/v1/worker/gpu_worker.py",
        "ln -s /build/.venv /opt/venv",
    )
    .env(
        {
            "PATH": "/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONPATH": REPO_REMOTE,
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "HF_HOME": f"{VOLUME_PATH}/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    # Code last, mounted fresh on each call, so edits don't rebuild the image.
    .add_local_dir(
        GRAFT_RL_DIR,
        REPO_REMOTE,
        ignore=["__pycache__", "*.pyc", "output", "wandb", "paper_data", "figures/figs"],
    )
    # Our distance rungs: their reward and detector functions, and configs
    # that GRAFT's sweep entries can point at (``absorb_configs/<name>.yaml``).
    .add_local_file(ROOT / "rungs.py", f"{REPO_REMOTE}/absorb_rungs.py")
    .add_local_dir(ROOT / "configs", f"{REPO_REMOTE}/absorb_configs")
)

APP = modal.App("graft-absorption")
VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# One toy run is tiny (SmolLM2-135M, 16-token completions); the paper packed
# 3-5 per H200 under MPS. A10G is the cheapest card with bf16 flash-attention 2.
DEFAULT_GPU = "A10G"
# vLLM's KV-cache share, as GiB rather than a fraction so it is the same on any
# card. The paper's Modal default is 0.05 of an H200, about 7 GiB.
VLLM_GIB = 6.0

# Gate 1: the paper's canonical object_qa GRAFT arm (monitor sees actually /
# exactly / quite and misses "indeed"), seed 1, from sweeps/<module>.py.
GATE1_SWEEP = "respmon_objinv_gr_canon_3seed"
GATE1_RUN = "object_qa_syco_respmon_iva_inv_gr_cls_canon_hf050_st1000_s1"


@APP.function(image=IMAGE, gpu=DEFAULT_GPU, volumes={VOLUME_PATH: VOLUME}, secrets=SECRETS, timeout=20 * 60)
def _smoke() -> dict:
    import torch
    import transformers
    import trl
    import vllm

    os.chdir(REPO_REMOTE)
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "vllm": vllm.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    from train import train_main  # noqa: F401

    info["train_import"] = True
    return info


def completion(evals: list[dict], max_steps: int, eval_every: int = 10) -> tuple[str, str | None]:
    """("ok", None) if the evals reach the last eval before ``max_steps``,
    else ("incomplete", why).

    A clean exit is not enough: GRAFT catches the SIGINT Modal sends when it
    stops a container (preemption, a spending limit), saves its plots and
    returns normally.
    """
    if not evals:
        return "incomplete", "no evals"
    last = evals[-1]["step"]
    if last + eval_every < max_steps:
        return "incomplete", f"stopped at step {last} of {max_steps}"
    return "ok", None


@APP.function(image=IMAGE, gpu=DEFAULT_GPU, cpu=8, volumes={VOLUME_PATH: VOLUME}, secrets=SECRETS, timeout=8 * 60 * 60)
def _train_run(sweep_module: str, run_name: str, overrides: dict, out_name: str) -> dict:
    """One training run from a GRAFT sweep file, with ``overrides`` applied."""
    import importlib
    import json
    import time

    import torch

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    runs = importlib.import_module(f"sweeps.{sweep_module}").runs
    (params,) = [run for run in runs if run["run_name"] == run_name]

    # Start from an empty directory: GRAFT appends to its eval and trace logs, so
    # files left by an earlier launch, or by an attempt Modal stopped and retried,
    # would mix into this run's.
    out_dir = f"{RUNS_ROOT}/{sweep_module}/{out_name}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    # The run's summary, written at the end so a detached run can be read back
    # from the volume.
    result_path = Path(out_dir) / RUN_RESULT
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    params = {
        **params,
        "no_wandb": True,
        "vllm_spawn": True,
        "vllm_gpu_memory": round(VLLM_GIB / total_gib, 3),
        **overrides,
        "output_dir": out_dir,
        "run_name": out_name,
        "gpu_id": 0,
    }

    import absorb_rungs
    from train import train_main

    absorb_rungs.register()
    start = time.time()
    status, error = "ok", None
    try:
        train_main(dict(params))
    except SystemExit as exit_:
        if exit_.code not in (0, None):
            status, error = "sysexit", str(exit_.code)
    except Exception as exc:  # report the failure with whatever the run wrote
        status, error = "crash", f"{type(exc).__name__}: {exc}"
    duration = time.time() - start

    eval_path = Path(out_dir) / "routing_eval.jsonl"
    evals = [json.loads(line) for line in eval_path.read_text().splitlines()] if eval_path.exists() else []
    if status == "ok":
        status, error = completion(evals, params["max_steps"], params.get("eval_every", 10))
    result = {
        "status": status,
        "error": error,
        "duration_s": duration,
        "device": torch.cuda.get_device_name(0),
        "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "params": {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool, type(None)))},
        "output_dir": out_dir,
    }
    result_path.write_text(json.dumps(result, indent=2))
    VOLUME.commit()
    return {**result, "routing_eval": evals}


def smoke(gpu: str = DEFAULT_GPU) -> dict:
    """Gate 1, part 1: the image builds, the GPU is visible and train.py imports."""
    with modal.enable_output(), APP.run():
        return _smoke.with_options(gpu=gpu).remote()


def train_run(
    sweep_module: str,
    run_name: str,
    overrides: dict | None = None,
    out_name: str | None = None,
    gpu: str = DEFAULT_GPU,
) -> dict:
    """Run one GRAFT training run on Modal; outputs persist under ``RUNS_ROOT``."""
    with modal.enable_output(), APP.run():
        return _train_run.with_options(gpu=gpu).remote(
            sweep_module, run_name, overrides or {}, out_name or run_name
        )


def train_runs(jobs: list[dict], detach: bool = False) -> list[dict]:
    """Run several jobs in parallel, one container each, under one app.

    Each job is a dict of ``train_run``'s arguments. Results come back in the
    order given; a job that fails to return is reported as a ``crash``. With
    ``detach``, the runs carry on if this process stops (the laptop sleeps, the
    network drops); each still writes ``RUN_RESULT`` to its output directory.
    """
    with modal.enable_output(), APP.run(detach=detach):
        calls = [
            _train_run.with_options(gpu=job.get("gpu", DEFAULT_GPU)).spawn(
                job["sweep_module"],
                job["run_name"],
                job.get("overrides") or {},
                job.get("out_name") or job["run_name"],
            )
            for job in jobs
        ]
        results = []
        for job, call in zip(jobs, calls):
            try:
                results.append(call.get())
            except Exception as exc:
                results.append({"status": "crash", "error": f"{type(exc).__name__}: {exc}", "job": job})
        return results
