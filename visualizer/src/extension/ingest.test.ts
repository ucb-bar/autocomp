import { describe, it, expect, afterAll } from "vitest";
import * as fs from "fs";
import * as path from "path";
import { parseCodeCandidate, ingestRun, ingestOutputDir } from "./ingest";

const FIXTURES = path.join(__dirname, "__fixtures__");
const SAMPLE_RUN = path.join(FIXTURES, "sample_run");

// ---------------------------------------------------------------------------
// Parser — real candidate files from an Autocomp NKI optimization run
// ---------------------------------------------------------------------------

describe("parseCodeCandidate", () => {
  it("parses iter-0 root candidate (no parent)", () => {
    const text = fs.readFileSync(path.join(SAMPLE_RUN, "candidates-iter-0", "candidate_0.txt"), "utf-8");
    const cand = parseCodeCandidate(text);

    expect(cand.parent).toBeNull();
    expect(cand.plan).toBeNull();
    expect(cand.code).toContain("@nki.jit");
    expect(cand.code).toContain("def test(");
    expect(cand.score).toBe(2.461);
    expect(cand.plan_gen_model).toBe("None");
    expect(cand.code_gen_model).toBe("None");
    expect(cand.hw_feedback).toEqual([]);
    expect(cand.stdout).toContain("Latency:");
    expect(cand.stderr).toBe("");
  });

  it("parses iter-2 candidate with parent chain", () => {
    const text = fs.readFileSync(path.join(SAMPLE_RUN, "candidates-iter-2", "candidate_0.txt"), "utf-8");
    const cand = parseCodeCandidate(text);

    expect(cand.parent).not.toBeNull();
    expect(cand.parent!.parent).toBeNull();
    expect(cand.parent!.score).toBe(2.461);
    expect(cand.parent!.plan).toBeNull();

    expect(cand.plan).toBeTruthy();
    expect(cand.code).toContain("@nki.jit");
    expect(cand.score).toBe(2.083);
    expect(cand.plan_gen_model).toBe("minimax.minimax-m2.5");
    expect(cand.code_gen_model).toBe("us.anthropic.claude-opus-4-5-20251101-v1:0");
  });

  it("preserves newlines and indentation in code", () => {
    const text = fs.readFileSync(path.join(SAMPLE_RUN, "candidates-iter-0", "candidate_0.txt"), "utf-8");
    const cand = parseCodeCandidate(text);

    const lines = cand.code!.split("\n");
    expect(lines[0]).toBe("@nki.jit");
    expect(lines[1]).toMatch(/^def test\(/);
    expect(lines.some((l) => l.startsWith("  "))).toBe(true);
  });

  it("handles plans containing markdown code fences", () => {
    const text = fs.readFileSync(path.join(SAMPLE_RUN, "candidates-iter-2", "candidate_0.txt"), "utf-8");
    const cand = parseCodeCandidate(text);

    // Plans often include ```python blocks
    expect(typeof cand.plan).toBe("string");
    expect(cand.plan!.length).toBeGreaterThan(50);
  });

  it("parses all candidates in every iteration without error", () => {
    for (const dir of fs.readdirSync(SAMPLE_RUN).filter((d) => d.startsWith("candidates-iter-"))) {
      const iterDir = path.join(SAMPLE_RUN, dir);
      for (const file of fs.readdirSync(iterDir).filter((f) => f.endsWith(".txt"))) {
        const text = fs.readFileSync(path.join(iterDir, file), "utf-8");
        const cand = parseCodeCandidate(text);
        expect(cand.code).toBeTruthy();
        expect(typeof cand.score).toBe("number");
      }
    }
  });
});

// ---------------------------------------------------------------------------
// End-to-end ingestion
// ---------------------------------------------------------------------------

describe("ingestRun", () => {
  it("ingests the sample run", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    expect(result).not.toBeNull();
    expect(result.schema_version).toBe(2);
    expect(result.run_id).toBe("sample_run");
  });

  it("loads all three iterations", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    const iters = result.iterations as Record<string, unknown>[];
    expect(iters).toHaveLength(3);
    expect(iters.map((it) => it.iter)).toEqual([0, 1, 2]);
  });

  it("loads correct candidate counts per iteration", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    const iters = result.iterations as Record<string, unknown>[];
    expect((iters[0].beam as unknown[]).length).toBe(1);
    expect((iters[1].beam as unknown[]).length).toBe(2);
    expect((iters[2].beam as unknown[]).length).toBe(4);
  });

  it("extracts config from run_metadata.json", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    const config = result.config as Record<string, unknown>;
    expect(config.beam_size).toBe(4);
    expect(config.metric).toBe("latency");
    expect(config.num_plan_candidates).toBe(4);
    expect((config.plan_models as string[]).length).toBe(4);
  });

  it("computes original score from iter-0", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    expect(result.original_score).toBe(2.461);
  });

  it("computes best score and speedup", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    expect(result.best_score).toBeLessThanOrEqual(2.461);
    expect(result.speedup).toBeGreaterThanOrEqual(1.0);
  });

  it("assigns stable candidate IDs", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    const iters = result.iterations as Record<string, unknown>[];
    const beam0 = iters[0].beam as Record<string, unknown>[];
    const beam2 = iters[2].beam as Record<string, unknown>[];
    expect(beam0[0].id).toBe("iter0_cand0");
    expect(beam2[0].id).toBe("iter2_cand0");
  });

  it("detects carry-forward candidates", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    const iters = result.iterations as Record<string, unknown>[];
    // iter-1 candidate_0 is the same code as iter-0 candidate_0
    const beam1 = iters[1].beam as Record<string, unknown>[];
    const carryForwards = beam1.filter((c) => c.is_carry_forward);
    expect(carryForwards.length).toBeGreaterThanOrEqual(1);
    expect(carryForwards[0].canonical_id).toBe("iter0_cand0");
  });

  it("resolves parent linkage", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    const iters = result.iterations as Record<string, unknown>[];
    const beam2 = iters[2].beam as Record<string, unknown>[];
    // Optimized candidates should link back to their parent
    const withParent = beam2.filter((c) => c.parent_id != null);
    expect(withParent.length).toBeGreaterThan(0);
  });

  it("loads eval results with correct/latency", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    const iters = result.iterations as Record<string, unknown>[];
    const generated = iters[1].generated as Record<string, unknown>[];
    expect(generated.length).toBe(8);
    expect(generated[0].correct).toBe(true);
    expect(typeof generated[0].score).toBe("number");
  });

  it("attaches per-iteration metrics", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    const iters = result.iterations as Record<string, unknown>[];
    // iter-0 has no metrics file, iter-1 and iter-2 do
    expect(iters[0].metrics).toBeUndefined();
    const m1 = iters[1].metrics as Record<string, unknown>;
    expect(m1).toBeDefined();
    expect(m1.iteration).toBe(1);
    expect(m1.iteration_total_s).toBe(552.681);
    expect((m1.evaluation as Record<string, unknown>).num_candidates).toBe(8);
  });

  it("attaches run-level metrics", () => {
    const result = ingestRun(SAMPLE_RUN)!;
    const rm = result.run_metrics as Record<string, unknown>;
    expect(rm).toBeDefined();
    expect(rm.run_total_s).toBe(1786.766);
    expect(rm.total_eval_duration_s).toBe(328.966);
  });

  it("returns null for empty directory", () => {
    const emptyDir = path.join(FIXTURES, "__empty_run__");
    fs.mkdirSync(emptyDir, { recursive: true });
    try {
      expect(ingestRun(emptyDir)).toBeNull();
    } finally {
      fs.rmdirSync(emptyDir);
    }
  });
});

describe("ingestOutputDir", () => {
  const outDir = path.join(FIXTURES, "__test_output__");

  afterAll(() => {
    fs.rmSync(outDir, { recursive: true, force: true });
  });

  it("produces runs.json index and per-run JSON", () => {
    const { runsIndex, errors } = ingestOutputDir(FIXTURES, outDir);
    expect(errors).toHaveLength(0);
    expect(runsIndex.length).toBeGreaterThanOrEqual(1);

    const index = JSON.parse(fs.readFileSync(path.join(outDir, "runs.json"), "utf-8"));
    expect(index.schema_version).toBe(2);

    const run = index.runs.find((r: Record<string, unknown>) => r.run_id === "sample_run");
    expect(run).toBeDefined();
    expect(run.num_iterations).toBe(3);
    expect(run.original_score).toBe(2.461);
  });

  it("run JSON contains full data", () => {
    ingestOutputDir(FIXTURES, outDir);
    const index = JSON.parse(fs.readFileSync(path.join(outDir, "runs.json"), "utf-8"));
    const run = index.runs.find((r: Record<string, unknown>) => r.run_id === "sample_run");
    const runData = JSON.parse(fs.readFileSync(path.join(outDir, run.file), "utf-8"));
    expect(runData.iterations).toHaveLength(3);
    expect((runData.iterations[2].beam as unknown[]).length).toBe(4);
    expect(runData.run_metrics).toBeDefined();
  });
});
