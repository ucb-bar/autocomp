#!/usr/bin/env python3
"""
Ingest Autocomp output directories into clean JSON for the visualizer.

Usage:
    python ingest.py /path/to/output --model openai::gpt-4o-mini
    python ingest.py /path/to/output --no-summarize
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from autocomp.search.code_repo import CodeCandidate
except ImportError:
    print("Error: The 'autocomp' Python package is required. "
          "Install it or run from the autocomp repository root.", file=sys.stderr)
    sys.exit(1)


def parse_run_config(dirname: str, run_dir: Path = None) -> dict:
    """Extract config params from run_metadata.json if available, else from directory name."""
    config = {"raw_name": dirname}

    if run_dir and (run_dir / "run_metadata.json").exists():
        try:
            with open(run_dir / "run_metadata.json") as f:
                meta = json.load(f)
            config.update(meta)
            all_models = sorted(set(meta.get("plan_models", []) + meta.get("code_models", [])))
            if all_models:
                config["models"] = all_models
            return config
        except Exception:
            pass

    beam_match = re.search(r"_b(\d+)_", dirname)
    if beam_match:
        config["beam_size"] = int(beam_match.group(1))

    iter_match = re.search(r"beam_iters(\d+)", dirname)
    if iter_match:
        config["iterations"] = int(iter_match.group(1))

    hw_match = re.search(r"(Trainium|Gemmini|GPU|CUDA)", dirname, re.IGNORECASE)
    if hw_match:
        config["hardware"] = hw_match.group(1)

    inst_match = re.search(r"(trn\d+\.\w+)", dirname)
    if inst_match:
        config["instance"] = inst_match.group(1)

    prob_match = re.search(r"(trn-tutorial|trn|gemmini|kernelbench|gpumode)_(\d+)_", dirname)
    if prob_match:
        config["problem"] = f"{prob_match.group(1)}_{prob_match.group(2)}"
        config["prob_type"] = prob_match.group(1)
        config["prob_id"] = int(prob_match.group(2))

    config["reimpl"] = "reimpl1" in dirname
    config["fgisa"] = "fgisa1" in dirname

    return config


def flatten_candidate(cand: CodeCandidate) -> dict:
    """Recursively flatten a CodeCandidate into a dict with parent_id linkage."""
    ancestry = []
    current = cand
    while current is not None:
        ancestry.append(current)
        current = current.parent
    ancestry.reverse()

    result = {
        "code": cand.code,
        "score": cand.score,
        "plan": cand.plan,
        "plan_model": cand.plan_gen_model,
        "code_model": cand.code_gen_model,
        "depth": len(ancestry) - 1,
    }
    return result, ancestry


def load_candidates_for_iter(run_dir: Path, iteration: int) -> list[CodeCandidate]:
    """Load all candidate files for a given iteration."""
    cand_dir = run_dir / f"candidates-iter-{iteration}"
    if not cand_dir.exists():
        return []

    candidates = []
    for path in sorted(cand_dir.glob("candidate_*.txt")):
        try:
            cand = eval(path.read_text())
            candidates.append(cand)
        except Exception as e:
            print(f"  Warning: Failed to load {path.name}: {e}")
    return candidates


def load_eval_results(run_dir: Path, iteration: int) -> list[dict]:
    """Load eval result JSON files for a given iteration."""
    results_dir = run_dir / f"eval-results-iter-{iteration}"
    if not results_dir.exists():
        return []

    results = []
    for path in sorted(results_dir.glob("code_*_result.txt")):
        if "_full" in path.name:
            continue
        try:
            data = json.loads(path.read_text())
            results.append(data)
        except Exception:
            pass

    full_results = []
    for path in sorted(results_dir.glob("code_*_result_full.txt")):
        try:
            text = path.read_text()
            first_line = text.split("\n")[0]
            plan_match = re.search(r"Plan: (.+?)(?:\nCodeCandidate|$)", text, re.DOTALL)
            plan_snippet = ""
            if plan_match:
                plan_text = plan_match.group(1).strip()
                plan_snippet = plan_text[:120].strip()
            full_results.append({"plan_snippet": plan_snippet, "raw_first_line": first_line})
        except Exception:
            full_results.append({})

    merged = []
    for i, r in enumerate(results):
        entry = dict(r)
        if i < len(full_results):
            entry.update(full_results[i])
        merged.append(entry)

    return merged


def assign_candidate_ids(iterations_data: list[dict]) -> None:
    """Assign stable IDs, deduplicate carry-forwards, and resolve parent linkage.

    A carry-forward is a candidate whose code is identical to one already seen in
    a previous iteration. We keep the first occurrence and mark later copies so the
    frontend can collapse them.
    """
    code_hash_to_canonical_id = {}
    code_hash_to_iter = {}

    for it_data in iterations_data:
        iteration = it_data["iter"]
        for ci, cand in enumerate(it_data["beam"]):
            cand_id = f"iter{iteration}_cand{ci}"
            cand["id"] = cand_id
            ch = hash(cand["code"]) if cand["code"] else None
            if ch is not None:
                if ch not in code_hash_to_canonical_id:
                    code_hash_to_canonical_id[ch] = cand_id
                    code_hash_to_iter[ch] = iteration
                    cand["_is_carry_forward"] = False
                else:
                    cand["_canonical_id"] = code_hash_to_canonical_id[ch]
                    cand["_is_carry_forward"] = True
            else:
                cand["_is_carry_forward"] = False

    for it_data in iterations_data:
        for cand in it_data["beam"]:
            parent_code = cand.pop("_parent_code", None)
            if parent_code:
                ph = hash(parent_code)
                cand["parent_id"] = code_hash_to_canonical_id.get(ph)
            else:
                cand["parent_id"] = None
            canonical = cand.pop("_canonical_id", None)
            is_cf = cand.pop("_is_carry_forward", False)
            cand["is_carry_forward"] = is_cf
            if is_cf and canonical:
                cand["canonical_id"] = canonical


def ingest_run(run_dir: Path) -> dict | None:
    """Ingest a single run directory into a structured dict."""
    dirname = run_dir.name
    config = parse_run_config(dirname, run_dir)

    cand_dirs = sorted(run_dir.glob("candidates-iter-*"))
    if not cand_dirs:
        return None

    iter_nums = sorted(
        int(d.name.replace("candidates-iter-", "")) for d in cand_dirs
    )

    iterations_data = []

    for iteration in iter_nums:
        candidates = load_candidates_for_iter(run_dir, iteration)
        if not candidates:
            continue

        beam = []
        for ci, cand in enumerate(candidates):
            flat, ancestry = flatten_candidate(cand)
            if cand.parent and cand.parent.code:
                flat["_parent_code"] = cand.parent.code
            beam.append(flat)

        eval_results = load_eval_results(run_dir, iteration)

        beam_codes = {hash(c["code"]) for c in beam if c["code"]}
        beam_scores = {c["score"] for c in beam if c["score"] is not None}

        failed = []
        for er in eval_results:
            correct = er.get("correct", False)
            latency = er.get("latency")
            if not correct:
                failed.append({
                    "correct": False,
                    "score": None,
                    "plan_snippet": er.get("plan_snippet", ""),
                    "error_summary": (er.get("stderr") or "")[:200],
                    "model": er.get("model", ""),
                })
            elif latency is not None and latency not in beam_scores:
                failed.append({
                    "correct": True,
                    "score": latency,
                    "plan_snippet": er.get("plan_snippet", ""),
                    "error_summary": None,
                    "model": er.get("model", ""),
                    "why_rejected": f"score {latency:.3f} ms not in top beam",
                })

        iterations_data.append({
            "iter": iteration,
            "beam": beam,
            "failed": failed,
        })

    assign_candidate_ids(iterations_data)

    if "models" not in config:
        config["models"] = []

    best_score = None
    original_score = None
    for it_data in iterations_data:
        for cand in it_data["beam"]:
            s = cand.get("score")
            if s is not None and s != float("inf"):
                if best_score is None or s < best_score:
                    best_score = s
        if it_data["iter"] == 0 and it_data["beam"]:
            original_score = it_data["beam"][0].get("score")

    speedup = None
    if original_score and best_score and best_score > 0:
        speedup = round(original_score / best_score, 2)

    return {
        "run_id": dirname,
        "config": config,
        "original_score": original_score,
        "best_score": best_score,
        "speedup": speedup,
        "iterations": iterations_data,
    }


def summarize_plans(run_data: dict, model_str: str, report_progress: bool = False) -> None:
    """Use an LLM to generate 1-2 sentence summaries of optimization plans."""
    parts = model_str.split("::", 1)
    if len(parts) == 2:
        provider, model_name = parts
    else:
        provider, model_name = None, parts[0]

    try:
        from autocomp.common.llm_utils import LLMClient
    except ImportError:
        print("Error: Plan summarization requires the 'autocomp' package with LLM utilities.", file=sys.stderr)
        sys.exit(1)
    client = LLMClient(model=model_name, provider=provider)

    prompts = []
    plan_indices = []

    for it_data in run_data["iterations"]:
        for ci, cand in enumerate(it_data["beam"]):
            plan = cand.get("plan")
            if plan and not cand.get("plan_summary"):
                prompt = (
                    "Summarize this code optimization plan in ONE short sentence (max 15 words). "
                    "Be specific and technical. No filler words.\n\n"
                    f"Plan:\n{plan[:3000]}"
                )
                prompts.append(prompt)
                plan_indices.append((it_data["iter"], ci))

    if not prompts:
        if report_progress:
            print(json.dumps({"progress": 0, "total": 0}), flush=True)
        return

    total = len(prompts)
    print(f"  Summarizing {total} plans with {model_str}...")
    if report_progress:
        print(json.dumps({"progress": 0, "total": total}), flush=True)

    BATCH_SIZE = 9
    done = 0
    for batch_start in range(0, total, BATCH_SIZE):
        batch_prompts = prompts[batch_start:batch_start + BATCH_SIZE]
        batch_indices = plan_indices[batch_start:batch_start + BATCH_SIZE]
        responses = client.chat_async(batch_prompts, num_samples=1, temperature=0.3)

        for (iteration, ci), response_list in zip(batch_indices, responses):
            if response_list:
                summary = response_list[0].strip()
                for it_data in run_data["iterations"]:
                    if it_data["iter"] == iteration:
                        it_data["beam"][ci]["plan_summary"] = summary
                        break
        done += len(batch_prompts)
        if report_progress:
            print(json.dumps({"progress": done, "total": total}), flush=True)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "summarize-run":
        parser = argparse.ArgumentParser(description="Summarize plans in a run JSON file")
        parser.add_argument("_cmd")
        parser.add_argument("run_file", help="Path to the run JSON file")
        parser.add_argument("--model", required=True,
                            help="LLM model (e.g. openai::gpt-4o-mini)")
        args = parser.parse_args()
        run_file = Path(args.run_file)
        if not run_file.exists():
            print(f"Error: {run_file} does not exist")
            sys.exit(1)
        with open(run_file) as f:
            run_data = json.load(f)
        summarize_plans(run_data, args.model, report_progress=True)
        with open(run_file, "w") as f:
            json.dump(run_data, f, indent=2, default=str)
        print(json.dumps({"status": "ok"}))
        return

    parser = argparse.ArgumentParser(description="Ingest Autocomp output for visualization")
    parser.add_argument("output_dir", help="Path to the Autocomp output/ directory")
    parser.add_argument("--model", default=None,
                        help="LLM model for plan summarization (e.g. openai::gpt-4o-mini). Omit to skip.")
    parser.add_argument("--no-summarize", action="store_true",
                        help="Skip plan summarization even if --model is set")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: visualizer/public/data/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "public" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and d.name != "exported"
    ])

    print(f"Found {len(run_dirs)} run directories in {output_dir}")

    runs_index = []
    for run_dir in run_dirs:
        print(f"Processing: {run_dir.name}")
        run_data = ingest_run(run_dir)
        if run_data is None:
            print(f"  Skipped (no candidates)")
            continue

        if args.model and not args.no_summarize:
            try:
                summarize_plans(run_data, args.model)
            except Exception as e:
                print(f"  Warning: Plan summarization failed: {e}")

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", run_dir.name)[:100]
        run_file = out_dir / f"{safe_name}.json"
        with open(run_file, "w") as f:
            json.dump(run_data, f, indent=2, default=str)
        print(f"  Wrote {run_file.name} ({len(run_data['iterations'])} iterations)")

        runs_index.append({
            "run_id": run_data["run_id"],
            "file": run_file.name,
            "config": run_data["config"],
            "original_score": run_data["original_score"],
            "best_score": run_data["best_score"],
            "speedup": run_data["speedup"],
            "num_iterations": len(run_data["iterations"]),
        })

    index_file = out_dir / "runs.json"
    with open(index_file, "w") as f:
        json.dump(runs_index, f, indent=2)
    print(f"\nWrote index: {index_file} ({len(runs_index)} runs)")


if __name__ == "__main__":
    main()
