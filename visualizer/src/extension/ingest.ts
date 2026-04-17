/**
 * Pure-TypeScript ingestion of Autocomp output directories into JSON for the
 * visualizer.  Zero Python dependency — candidate files (Python repr format)
 * are parsed with a small recursive-descent parser.
 */

import * as fs from "fs";
import * as path from "path";

const SCHEMA_VERSION = 2;

// ---------------------------------------------------------------------------
// Python repr() parser for CodeCandidate files
// ---------------------------------------------------------------------------

export interface CodeCandidate {
  parent: CodeCandidate | null;
  plan: string | null;
  code: string | null;
  score: number | null;
  translation_score: number | null;
  hw_feedback: string[];
  plan_gen_model: string | null;
  code_gen_model: string | null;
  stdout: string | null;
  stderr: string | null;
}

class Parser {
  private pos = 0;
  constructor(private src: string) {}

  private peek(): string { return this.src[this.pos]; }
  private eof(): boolean { return this.pos >= this.src.length; }
  private advance(n = 1): string {
    const s = this.src.slice(this.pos, this.pos + n);
    this.pos += n;
    return s;
  }
  private skipWs(): void {
    while (!this.eof() && /\s/.test(this.peek())) this.pos++;
  }
  private expect(s: string): void {
    for (const ch of s) {
      if (this.eof() || this.peek() !== ch) {
        throw new Error(`Expected '${s}' at pos ${this.pos}, got '${this.src.slice(this.pos, this.pos + 20)}'`);
      }
      this.pos++;
    }
  }
  private startsWith(s: string): boolean {
    return this.src.startsWith(s, this.pos);
  }

  parse(): CodeCandidate {
    this.skipWs();
    return this.parseValue() as CodeCandidate;
  }

  private parseValue(): unknown {
    this.skipWs();
    if (this.eof()) throw new Error("Unexpected end of input");

    if (this.startsWith("CodeCandidate(")) return this.parseCandidate();
    if (this.startsWith("None")) { this.advance(4); return null; }
    if (this.startsWith("True")) { this.advance(4); return true; }
    if (this.startsWith("False")) { this.advance(5); return false; }
    if (this.startsWith("'''")) return this.parseTripleQuoted();
    if (this.startsWith('"""')) return this.parseTripleQuotedDouble();
    if (this.peek() === "'") return this.parseSingleQuoted();
    if (this.peek() === '"') return this.parseDoubleQuoted();
    if (this.peek() === "[") return this.parseList();
    if (this.peek() === "(") return this.parseTuple();
    if (this.peek() === "{") return this.parseDict();
    if (this.peek() === "b" && this.pos + 1 < this.src.length && (this.src[this.pos + 1] === "'" || this.src[this.pos + 1] === '"')) {
      this.advance(1);
      const s = this.peek() === "'" ? this.parseSingleQuoted() : this.parseDoubleQuoted();
      return s;
    }
    if (/[-+0-9.]/.test(this.peek()) || this.startsWith("inf") || this.startsWith("-inf") || this.startsWith("float(")) {
      return this.parseNumber();
    }
    throw new Error(`Unexpected char '${this.peek()}' at pos ${this.pos}`);
  }

  private parseCandidate(): CodeCandidate {
    this.expect("CodeCandidate(");
    const kwargs: Record<string, unknown> = {};
    this.skipWs();
    while (!this.eof() && this.peek() !== ")") {
      const name = this.parseIdent();
      this.skipWs();
      this.expect("=");
      this.skipWs();
      kwargs[name] = this.parseValue();
      this.skipWs();
      if (this.peek() === ",") { this.advance(); this.skipWs(); }
    }
    this.expect(")");
    return {
      parent: (kwargs.parent as CodeCandidate | null) ?? null,
      plan: (kwargs.plan as string | null) ?? null,
      code: (kwargs.code as string | null) ?? null,
      score: (kwargs.score as number | null) ?? null,
      translation_score: (kwargs.translation_score as number | null) ?? null,
      hw_feedback: (kwargs.hw_feedback as string[]) ?? [],
      plan_gen_model: (kwargs.plan_gen_model as string | null) ?? null,
      code_gen_model: (kwargs.code_gen_model as string | null) ?? null,
      stdout: (kwargs.stdout as string | null) ?? null,
      stderr: (kwargs.stderr as string | null) ?? null,
    };
  }

  private parseIdent(): string {
    const start = this.pos;
    while (!this.eof() && /[a-zA-Z0-9_]/.test(this.peek())) this.pos++;
    return this.src.slice(start, this.pos);
  }

  private parseTripleQuoted(): string {
    this.expect("'''");
    let result = "";
    while (!this.eof()) {
      if (this.startsWith("'''")) { this.advance(3); return this.unescapePython(result); }
      if (this.peek() === "\\" && this.pos + 1 < this.src.length) {
        result += this.advance(2);
      } else {
        result += this.advance();
      }
    }
    throw new Error("Unterminated triple-quoted string");
  }

  private parseTripleQuotedDouble(): string {
    this.expect('"""');
    let result = "";
    while (!this.eof()) {
      if (this.startsWith('"""')) { this.advance(3); return this.unescapePython(result); }
      if (this.peek() === "\\" && this.pos + 1 < this.src.length) {
        result += this.advance(2);
      } else {
        result += this.advance();
      }
    }
    throw new Error("Unterminated triple-double-quoted string");
  }

  private parseSingleQuoted(): string {
    this.expect("'");
    let result = "";
    while (!this.eof() && this.peek() !== "'") {
      if (this.peek() === "\\" && this.pos + 1 < this.src.length) {
        result += this.advance(2);
      } else {
        result += this.advance();
      }
    }
    this.expect("'");
    return this.unescapePython(result);
  }

  private parseDoubleQuoted(): string {
    this.expect('"');
    let result = "";
    while (!this.eof() && this.peek() !== '"') {
      if (this.peek() === "\\" && this.pos + 1 < this.src.length) {
        result += this.advance(2);
      } else {
        result += this.advance();
      }
    }
    this.expect('"');
    return this.unescapePython(result);
  }

  private unescapePython(s: string): string {
    return s.replace(/\\([\\'\"nrtbf0]|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8})/g, (_, esc: string) => {
      if (esc === "\\") return "\\";
      if (esc === "'") return "'";
      if (esc === '"') return '"';
      if (esc === "n") return "\n";
      if (esc === "r") return "\r";
      if (esc === "t") return "\t";
      if (esc === "b") return "\b";
      if (esc === "f") return "\f";
      if (esc === "0") return "\0";
      if (esc.startsWith("x") || esc.startsWith("u") || esc.startsWith("U")) {
        return String.fromCodePoint(parseInt(esc.slice(1), 16));
      }
      return "\\" + esc;
    });
  }

  private parseList(): unknown[] {
    this.expect("[");
    this.skipWs();
    const items: unknown[] = [];
    while (!this.eof() && this.peek() !== "]") {
      items.push(this.parseValue());
      this.skipWs();
      if (this.peek() === ",") { this.advance(); this.skipWs(); }
    }
    this.expect("]");
    return items;
  }

  private parseTuple(): unknown[] {
    this.expect("(");
    this.skipWs();
    const items: unknown[] = [];
    while (!this.eof() && this.peek() !== ")") {
      items.push(this.parseValue());
      this.skipWs();
      if (this.peek() === ",") { this.advance(); this.skipWs(); }
    }
    this.expect(")");
    return items;
  }

  private parseDict(): Record<string, unknown> {
    this.expect("{");
    this.skipWs();
    const result: Record<string, unknown> = {};
    while (!this.eof() && this.peek() !== "}") {
      const key = this.parseValue() as string;
      this.skipWs();
      this.expect(":");
      this.skipWs();
      result[String(key)] = this.parseValue();
      this.skipWs();
      if (this.peek() === ",") { this.advance(); this.skipWs(); }
    }
    this.expect("}");
    return result;
  }

  private parseNumber(): number {
    if (this.startsWith("float('inf')") || this.startsWith("float(\"inf\")")) {
      this.advance(12);
      return Infinity;
    }
    if (this.startsWith("float('-inf')") || this.startsWith("float(\"-inf\")")) {
      this.advance(13);
      return -Infinity;
    }
    if (this.startsWith("float('nan')") || this.startsWith("float(\"nan\")")) {
      this.advance(12);
      return NaN;
    }
    if (this.startsWith("inf")) { this.advance(3); return Infinity; }
    const start = this.pos;
    if (this.peek() === "-" || this.peek() === "+") this.pos++;
    while (!this.eof() && /[0-9.]/.test(this.peek())) this.pos++;
    if (!this.eof() && (this.peek() === "e" || this.peek() === "E")) {
      this.pos++;
      if (!this.eof() && (this.peek() === "+" || this.peek() === "-")) this.pos++;
      while (!this.eof() && /[0-9]/.test(this.peek())) this.pos++;
    }
    const numStr = this.src.slice(start, this.pos);
    return Number(numStr);
  }
}

export function parseCodeCandidate(text: string): CodeCandidate {
  return new Parser(text).parse();
}

// ---------------------------------------------------------------------------
// Ingestion helpers
// ---------------------------------------------------------------------------

interface FlatCandidate {
  code: string | null;
  score: number | null;
  plan: string | null;
  plan_model: string | null;
  code_model: string | null;
  depth: number;
  _parent_code?: string;
  [key: string]: unknown;
}

function flattenCandidate(cand: CodeCandidate): { flat: FlatCandidate; depth: number } {
  let depth = 0;
  let cur: CodeCandidate | null = cand;
  while (cur?.parent) { depth++; cur = cur.parent; }

  const flat: FlatCandidate = {
    code: cand.code,
    score: cand.score,
    plan: cand.plan,
    plan_model: cand.plan_gen_model,
    code_model: cand.code_gen_model,
    depth,
  };
  if (cand.parent?.code) {
    flat._parent_code = cand.parent.code;
  }
  return { flat, depth };
}

function loadCandidatesForIter(runDir: string, iteration: number): CodeCandidate[] {
  const candDir = path.join(runDir, `candidates-iter-${iteration}`);
  if (!fs.existsSync(candDir)) return [];

  const files = fs.readdirSync(candDir)
    .filter((f) => f.startsWith("candidate_") && f.endsWith(".txt"))
    .sort();

  const candidates: CodeCandidate[] = [];
  for (const file of files) {
    try {
      const text = fs.readFileSync(path.join(candDir, file), "utf-8");
      candidates.push(parseCodeCandidate(text));
    } catch (e) {
      console.warn(`Warning: Failed to load ${file}: ${e}`);
    }
  }
  return candidates;
}

function loadEvalResults(runDir: string, iteration: number): Record<string, unknown>[] {
  const resultsDir = path.join(runDir, `eval-results-iter-${iteration}`);
  if (!fs.existsSync(resultsDir)) return [];

  const resultFiles = fs.readdirSync(resultsDir)
    .filter((f) => f.startsWith("code_") && f.endsWith("_result.txt") && !f.includes("_full"))
    .sort();

  const results: Record<string, unknown>[] = [];
  for (const file of resultFiles) {
    try {
      results.push(JSON.parse(fs.readFileSync(path.join(resultsDir, file), "utf-8")));
    } catch { /* skip */ }
  }

  const fullFiles = fs.readdirSync(resultsDir)
    .filter((f) => f.startsWith("code_") && f.endsWith("_result_full.txt"))
    .sort();

  const fullResults: Record<string, string>[] = [];
  for (const file of fullFiles) {
    try {
      const text = fs.readFileSync(path.join(resultsDir, file), "utf-8");
      const firstLine = text.split("\n")[0];
      const planMatch = text.match(/Plan: (.+?)(?:\nCodeCandidate|$)/s);
      const planSnippet = planMatch ? planMatch[1].trim().slice(0, 120).trim() : "";
      fullResults.push({ plan_snippet: planSnippet, raw_first_line: firstLine });
    } catch {
      fullResults.push({});
    }
  }

  return results.map((r, i) => ({ ...r, ...(fullResults[i] || {}) }));
}

function assignCandidateIds(iterationsData: Record<string, unknown>[]): void {
  const codeHashToCanonicalId: Record<string, string> = {};
  const codeHashToIter: Record<string, number> = {};

  for (const itData of iterationsData) {
    const iteration = itData.iter as number;
    const beam = itData.beam as FlatCandidate[];
    for (let ci = 0; ci < beam.length; ci++) {
      const cand = beam[ci];
      const candId = `iter${iteration}_cand${ci}`;
      cand.id = candId;
      const ch = cand.code ? simpleHash(cand.code) : null;
      if (ch !== null) {
        if (!(ch in codeHashToCanonicalId)) {
          codeHashToCanonicalId[ch] = candId;
          codeHashToIter[ch] = iteration;
          cand._is_carry_forward = false;
        } else {
          cand._canonical_id = codeHashToCanonicalId[ch];
          cand._is_carry_forward = true;
        }
      } else {
        cand._is_carry_forward = false;
      }
    }
  }

  for (const itData of iterationsData) {
    const beam = itData.beam as FlatCandidate[];
    for (const cand of beam) {
      const parentCode = cand._parent_code;
      delete cand._parent_code;
      if (parentCode) {
        const ph = simpleHash(parentCode);
        cand.parent_id = codeHashToCanonicalId[ph] ?? null;
      } else {
        cand.parent_id = null;
      }
      const canonical = cand._canonical_id;
      const isCf = cand._is_carry_forward as boolean;
      delete cand._canonical_id;
      delete cand._is_carry_forward;
      cand.is_carry_forward = isCf;
      if (isCf && canonical) {
        cand.canonical_id = canonical;
      }
    }
  }
}

function simpleHash(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return String(h);
}

function parseRunConfig(dirname: string, runDir: string): Record<string, unknown> {
  const config: Record<string, unknown> = { raw_name: dirname };
  const metaPath = path.join(runDir, "run_metadata.json");
  if (fs.existsSync(metaPath)) {
    try {
      Object.assign(config, JSON.parse(fs.readFileSync(metaPath, "utf-8")));
    } catch { /* skip */ }
  }
  return config;
}

// ---------------------------------------------------------------------------
// Main ingestion
// ---------------------------------------------------------------------------

export function ingestRun(runDir: string): Record<string, unknown> | null {
  const dirname = path.basename(runDir);
  const config = parseRunConfig(dirname, runDir);

  const candDirs = fs.readdirSync(runDir)
    .filter((d) => d.startsWith("candidates-iter-") && fs.statSync(path.join(runDir, d)).isDirectory())
    .sort();

  if (candDirs.length === 0) return null;

  const iterNums = candDirs
    .map((d) => parseInt(d.replace("candidates-iter-", ""), 10))
    .sort((a, b) => a - b);

  const iterationsData: Record<string, unknown>[] = [];

  for (const iteration of iterNums) {
    const candidates = loadCandidatesForIter(runDir, iteration);
    if (candidates.length === 0) continue;

    const beam: FlatCandidate[] = [];
    for (const cand of candidates) {
      const { flat } = flattenCandidate(cand);
      beam.push(flat);
    }

    const evalResults = loadEvalResults(runDir, iteration);

    const beamScoreCounts: Record<number, number> = {};
    for (const c of beam) {
      if (c.score != null) beamScoreCounts[c.score] = (beamScoreCounts[c.score] ?? 0) + 1;
    }

    const generated: Record<string, unknown>[] = [];
    const failed: Record<string, unknown>[] = [];

    for (const er of evalResults) {
      const correct = er.correct as boolean ?? false;
      const latency = er.latency as number | undefined;
      let kept = false;
      let whyRejected: string | null = null;

      if (correct && latency != null && (beamScoreCounts[latency] ?? 0) > 0) {
        kept = true;
        beamScoreCounts[latency]--;
      } else if (correct && latency != null) {
        whyRejected = `score ${parseFloat(latency.toFixed(3))} ms not in top beam`;
      }

      const item: Record<string, unknown> = {
        correct: Boolean(correct),
        kept,
        score: correct && latency != null ? latency : null,
        plan_snippet: (er.plan_snippet as string) ?? "",
        error_summary: correct ? null : ((er.stderr as string) ?? "").slice(0, 200),
        model: (er.model as string) ?? "",
      };
      if (whyRejected != null) item.why_rejected = whyRejected;
      generated.push(item);

      if (!correct || !kept) {
        const failedItem = { ...item };
        delete failedItem.kept;
        failed.push(failedItem);
      }
    }

    const itEntry: Record<string, unknown> = { iter: iteration, beam, generated, failed };

    const metricsPath = path.join(runDir, `metrics-iter-${iteration}.json`);
    if (fs.existsSync(metricsPath)) {
      try { itEntry.metrics = JSON.parse(fs.readFileSync(metricsPath, "utf-8")); } catch { /* skip */ }
    }

    iterationsData.push(itEntry);
  }

  assignCandidateIds(iterationsData);

  let bestScore: number | null = null;
  let originalScore: number | null = null;
  for (const itData of iterationsData) {
    const beam = itData.beam as FlatCandidate[];
    for (const cand of beam) {
      if (cand.score != null && cand.score !== Infinity) {
        if (bestScore === null || cand.score < bestScore) bestScore = cand.score;
      }
    }
    if (itData.iter === 0 && beam.length > 0) {
      originalScore = beam[0].score;
    }
  }

  let speedup: number | null = null;
  if (originalScore && bestScore && bestScore > 0) {
    speedup = Math.round((originalScore / bestScore) * 100) / 100;
  }

  let runMetrics: unknown = null;
  const runMetricsPath = path.join(runDir, "run_metrics.json");
  if (fs.existsSync(runMetricsPath)) {
    try { runMetrics = JSON.parse(fs.readFileSync(runMetricsPath, "utf-8")); } catch { /* skip */ }
  }

  const result: Record<string, unknown> = {
    schema_version: SCHEMA_VERSION,
    run_id: dirname,
    config,
    original_score: originalScore,
    best_score: bestScore,
    speedup,
    iterations: iterationsData,
  };
  if (runMetrics != null) result.run_metrics = runMetrics;
  return result;
}

export function ingestOutputDir(outputDir: string, outDir: string): { runsIndex: unknown[]; errors: string[] } {
  fs.mkdirSync(outDir, { recursive: true });

  const runDirs = fs.readdirSync(outputDir)
    .filter((d) => {
      const full = path.join(outputDir, d);
      return fs.statSync(full).isDirectory() && !d.startsWith(".") && d !== "exported";
    })
    .sort();

  const runsIndex: unknown[] = [];
  const errors: string[] = [];

  for (const dir of runDirs) {
    const runDir = path.join(outputDir, dir);
    try {
      const runData = ingestRun(runDir);
      if (!runData) continue;

      const safeName = dir.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 100);
      const runFile = path.join(outDir, `${safeName}.json`);
      fs.writeFileSync(runFile, JSON.stringify(runData, null, 2));

      runsIndex.push({
        run_id: runData.run_id,
        file: `${safeName}.json`,
        config: runData.config,
        original_score: runData.original_score,
        best_score: runData.best_score,
        speedup: runData.speedup,
        num_iterations: (runData.iterations as unknown[]).length,
      });
    } catch (e) {
      errors.push(`${dir}: ${e}`);
    }
  }

  const indexFile = path.join(outDir, "runs.json");
  fs.writeFileSync(indexFile, JSON.stringify({ schema_version: SCHEMA_VERSION, runs: runsIndex }, null, 2));

  return { runsIndex, errors };
}
