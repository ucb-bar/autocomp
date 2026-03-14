import { useEffect, useState, useMemo } from "react";
import type { RunData, RunIndexEntry, BeamCandidate, FailedCandidate } from "./lib/types";
import { formatModel } from "./lib/format";
import ScoreChart from "./components/ScoreChart";
import BeamTree from "./components/BeamTree";
import PlanCard from "./components/PlanCard";
import Dropdown from "./components/Dropdown";

declare function acquireVsCodeApi(): {
  postMessage(msg: unknown): void;
  getState(): unknown;
  setState(state: unknown): void;
};

const vscode = acquireVsCodeApi();

function ModelList({ label, models, className }: { label: string; models: string[]; className?: string }) {
  if (!models.length) return null;
  const unique = [...new Set(models.map(formatModel))];
  return (
    <span className={`inline-flex items-center gap-1 ${className ?? ""}`}>
      <span className="text-stone-400 text-[10px] uppercase tracking-wide">{label}</span>
      {unique.map((m, i) => (
        <span key={i} className="font-mono text-xs text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded">{m}</span>
      ))}
    </span>
  );
}

function MetaBadge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-stone-100 text-stone-600 font-mono">
      {children}
    </span>
  );
}

function FailedItem({ item }: { item: FailedCandidate }) {
  const [expanded, setExpanded] = useState(false);
  const hasDetails = (item.plan_snippet && item.plan_snippet.length > 80)
    || (item.error_summary && item.error_summary.length > 100);

  return (
    <div className="text-xs font-mono rounded bg-stone-50 border border-stone-100">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full text-left px-3 py-2 flex items-start gap-2 hover:bg-stone-100 rounded transition-colors"
      >
        <span className="flex-shrink-0 mt-px">
          {!item.correct ? <span className="text-red-500">✗</span> : <span className="text-amber-500">~</span>}
        </span>
        <span className="flex-1 min-w-0">
          {item.plan_snippet && (
            <span className="text-stone-600">
              {expanded ? item.plan_snippet : (
                item.plan_snippet.length > 120
                  ? item.plan_snippet.slice(0, 120) + "..."
                  : item.plan_snippet
              )}
            </span>
          )}
          {!expanded && item.error_summary && (
            <span className="text-red-400 ml-1">
              {item.error_summary.length > 120
                ? item.error_summary.slice(0, 120) + "..."
                : item.error_summary}
            </span>
          )}
        </span>
        <span className="flex-shrink-0 flex items-center gap-2 text-stone-400">
          {item.score !== null && <span>{item.score.toFixed(3)}</span>}
          {item.model && <span className="text-stone-300">{formatModel(item.model)}</span>}
          {hasDetails && <span className="text-[10px]">{expanded ? "▾" : "▸"}</span>}
        </span>
      </button>
      {expanded && (
        <div className="px-3 pb-2.5 pt-0.5 space-y-1.5 border-t border-stone-100">
          {item.error_summary && (
            <div>
              <span className="text-stone-400">Error: </span>
              <span className="text-red-400 whitespace-pre-wrap break-words">{item.error_summary}</span>
            </div>
          )}
          {item.why_rejected && (
            <div>
              <span className="text-stone-400">Rejected: </span>
              <span className="text-amber-500 whitespace-pre-wrap break-words">{item.why_rejected}</span>
            </div>
          )}
          {item.plan_snippet && (
            <div>
              <span className="text-stone-400">Plan: </span>
              <span className="text-stone-600 whitespace-pre-wrap break-words">{item.plan_snippet}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [runs, setRuns] = useState<RunIndexEntry[]>([]);
  const [run, setRun] = useState<RunData | null>(null);
  const [runFile, setRunFile] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<BeamCandidate | null>(null);
  const [diffOpen, setDiffOpen] = useState(true);
  const [summarizing, setSummarizing] = useState(false);
  const [showFailed, setShowFailed] = useState<number | null>(null);

  useEffect(() => {
    const handler = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.type === "runs") {
        setRuns(msg.data);
        setLoading(false);
      } else if (msg.type === "runData") {
        setRun(msg.data);
        setSelected(null);
        setLoading(false);
        setSummarizing(false);
      } else if (msg.type === "summarizeResult") {
        setSummarizing(false);
      }
    };
    window.addEventListener("message", handler);
    vscode.postMessage({ type: "ready" });
    return () => window.removeEventListener("message", handler);
  }, []);

  const allCandidates = useMemo(() => {
    if (!run) return [];
    return run.iterations.flatMap((it) => it.beam);
  }, [run]);

  const parentOf = useMemo(() => {
    if (!selected || !selected.parent_id) return null;
    return allCandidates.find((c) => c.id === selected.parent_id) ?? null;
  }, [selected, allCandidates]);

  const parentScore = parentOf?.score ?? run?.original_score ?? null;

  const ancestorChain = useMemo(() => {
    if (!selected) return [];
    const chain: { id: string; label: string; code: string }[] = [];
    let cur: BeamCandidate | undefined = selected;
    while (cur?.parent_id) {
      const parent = allCandidates.find((c) => c.id === cur!.parent_id);
      if (!parent) break;
      chain.push({
        id: parent.id,
        label: `${parent.id}${parent.score != null ? ` (${parent.score.toFixed(3)})` : ""}`,
        code: parent.code ?? "",
      });
      cur = parent;
    }
    const original = allCandidates[0];
    if (original && (chain.length === 0 || chain[chain.length - 1].id !== original.id)) {
      chain.push({
        id: "__original__",
        label: `Original${run?.original_score != null ? ` (${run.original_score.toFixed(3)})` : ""}`,
        code: original.code ?? "",
      });
    }
    return chain;
  }, [selected, allCandidates, run]);

  const [diffAncestor, setDiffAncestor] = useState<string | null>(null);

  const effectiveDiffAncestor = ancestorChain.find((a) => a.id === diffAncestor) ?? ancestorChain[0] ?? null;

  const handleCandidateSelect = (candidate: BeamCandidate | null) => {
    setSelected(candidate);
    setDiffAncestor(null);
  };

  const handleOpenDiff = () => {
    if (!selected || !effectiveDiffAncestor) return;
    vscode.postMessage({
      type: "openDiff",
      parentCode: effectiveDiffAncestor.code,
      candidateCode: selected.code ?? "",
      parentLabel: effectiveDiffAncestor.id === "__original__" ? "Original" : effectiveDiffAncestor.id,
      candidateLabel: selected.id,
    });
  };

  const handleRunSelect = (file: string) => {
    setLoading(true);
    setRun(null);
    setRunFile(file);
    vscode.postMessage({ type: "selectRun", file });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-stone-50 flex items-center justify-center">
        <div className="text-stone-400 font-mono text-sm">Loading...</div>
      </div>
    );
  }

  if (!run) {
    const sortedRuns = [...runs]
      .filter((r) => r.num_iterations > 1)
      .sort((a, b) => (b.speedup ?? 0) - (a.speedup ?? 0));

    return (
      <main className="min-h-screen bg-stone-50">
        <div className="max-w-6xl mx-auto px-6 py-10">
          <div className="mb-8">
            <h1 className="text-2xl font-semibold text-stone-900 tracking-tight">
              Autocomp Trace Visualizer
            </h1>
            <p className="text-stone-500 text-sm mt-1">
              {sortedRuns.length} optimization runs with results
            </p>
          </div>
          <div className="space-y-3">
            {sortedRuns.map((r) => {
              const hasSeparateModels = (r.config.plan_models?.length ?? 0) > 0 || (r.config.code_models?.length ?? 0) > 0;
              return (
              <button key={r.run_id} onClick={() => handleRunSelect(r.file)}
                className="block w-full text-left">
                <div className="bg-white border border-stone-200 rounded-lg px-5 py-4 hover:border-indigo-300 hover:shadow-sm transition-all duration-150">
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        {r.config.problem && (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-50 text-indigo-700 font-mono">
                            {r.config.problem}
                          </span>
                        )}
                        {r.config.beam_size && <MetaBadge>beam {r.config.beam_size}</MetaBadge>}
                        <MetaBadge>{r.num_iterations - 1} iters</MetaBadge>
                        {r.config.metric && <MetaBadge>{r.config.metric}</MetaBadge>}
                      </div>
                      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                        {hasSeparateModels ? (
                          <>
                            <ModelList label="plan" models={r.config.plan_models ?? []} />
                            <ModelList label="code" models={r.config.code_models ?? []} />
                          </>
                        ) : (
                          r.config.models.length > 0 && (
                            <ModelList label="models" models={r.config.models} />
                          )
                        )}
                      </div>
                    </div>
                    <div className="text-right flex-shrink-0">
                      {r.speedup && (
                        <div className="text-lg font-semibold text-indigo-700 font-mono">
                          {r.speedup}x
                        </div>
                      )}
                      <div className="text-xs text-stone-400 font-mono mt-0.5">
                        {r.original_score?.toFixed(3)} → {r.best_score?.toFixed(3)}
                      </div>
                    </div>
                  </div>
                </div>
              </button>
            );})}
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-stone-50">
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="mb-6">
          <button onClick={() => { setRun(null); setRunFile(null); }}
            className="text-sm text-stone-400 hover:text-indigo-600 transition-colors">
            ← All runs
          </button>
          <div className="flex items-center gap-4 mt-2">
            <h1 className="text-xl font-semibold text-stone-900 tracking-tight">
              {run.config.problem ?? "Run"}
            </h1>
            {run.speedup && (
              <span className="text-lg font-semibold text-indigo-700 font-mono">
                {run.speedup}x speedup
              </span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2 mt-2">
            {run.config.beam_size && <MetaBadge>beam {run.config.beam_size}</MetaBadge>}
            <MetaBadge>{run.iterations.length - 1} iterations</MetaBadge>
            {run.config.num_plan_candidates && <MetaBadge>{run.config.num_plan_candidates} plans</MetaBadge>}
            {run.config.num_code_candidates && <MetaBadge>{run.config.num_code_candidates} codes</MetaBadge>}
            {run.config.metric && <MetaBadge>{run.config.metric}</MetaBadge>}
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2">
            {(run.config.plan_models?.length ?? 0) > 0 || (run.config.code_models?.length ?? 0) > 0 ? (
              <>
                <ModelList label="plan" models={run.config.plan_models ?? []} />
                <ModelList label="code" models={run.config.code_models ?? []} />
              </>
            ) : (
              run.config.models.length > 0 && (
                <ModelList label="models" models={run.config.models} />
              )
            )}
          </div>
          <div className="text-sm text-stone-400 font-mono mt-2">
            {run.original_score?.toFixed(3)} → {run.best_score?.toFixed(3)}
          </div>
          <details className="mt-2 group">
            <summary className="text-xs text-stone-400 hover:text-stone-600 cursor-pointer select-none transition-colors">
              Run config
            </summary>
            <div className="mt-2 rounded-lg border border-stone-150 bg-stone-50 px-4 py-3">
              <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs font-mono">
                {run.config.beam_size != null && <><dt className="text-stone-400">beam_size</dt><dd className="text-stone-700">{run.config.beam_size}</dd></>}
                {run.config.num_plan_candidates != null && <><dt className="text-stone-400">num_plan_candidates</dt><dd className="text-stone-700">{run.config.num_plan_candidates}</dd></>}
                {run.config.num_code_candidates != null && <><dt className="text-stone-400">num_code_candidates</dt><dd className="text-stone-700">{run.config.num_code_candidates}</dd></>}
                <dt className="text-stone-400">iterations</dt><dd className="text-stone-700">{run.iterations.length - 1}</dd>
                {run.config.metric && <><dt className="text-stone-400">metric</dt><dd className="text-stone-700">{run.config.metric}</dd></>}
                {run.config.simulator && <><dt className="text-stone-400">simulator</dt><dd className="text-stone-700">{run.config.simulator}</dd></>}
                {run.config.hardware && <><dt className="text-stone-400">hardware</dt><dd className="text-stone-700">{run.config.hardware}</dd></>}
                {run.config.instance && <><dt className="text-stone-400">instance</dt><dd className="text-stone-700">{run.config.instance}</dd></>}
                {run.config.give_score_feedback != null && <><dt className="text-stone-400">give_score_feedback</dt><dd className="text-stone-700">{String(run.config.give_score_feedback)}</dd></>}
                {run.config.give_hw_feedback != null && <><dt className="text-stone-400">give_hw_feedback</dt><dd className="text-stone-700">{String(run.config.give_hw_feedback)}</dd></>}
                {run.config.include_ancestors != null && <><dt className="text-stone-400">include_ancestors</dt><dd className="text-stone-700">{String(run.config.include_ancestors)}</dd></>}
                {(run.config.plan_models?.length ?? 0) > 0 && <><dt className="text-stone-400">plan_models</dt><dd className="text-stone-700">{run.config.plan_models!.map(formatModel).join(", ")}</dd></>}
                {(run.config.code_models?.length ?? 0) > 0 && <><dt className="text-stone-400">code_models</dt><dd className="text-stone-700">{run.config.code_models!.map(formatModel).join(", ")}</dd></>}
                <dt className="text-stone-400">raw_name</dt><dd className="text-stone-700 break-all">{run.config.raw_name}</dd>
              </dl>
            </div>
          </details>
          <div className="mt-3">
            <button
              onClick={() => {
                if (!runFile || summarizing) return;
                setSummarizing(true);
                vscode.postMessage({ type: "summarizePlans", file: runFile });
              }}
              disabled={summarizing}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                summarizing
                  ? "text-stone-400 bg-stone-100 cursor-wait"
                  : "text-violet-700 bg-violet-50 hover:bg-violet-100"
              }`}
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
                <path d="M7.657 6.247c.11-.33.576-.33.686 0l.645 1.937a2.89 2.89 0 0 0 1.829 1.828l1.936.645c.33.11.33.576 0 .686l-1.937.645a2.89 2.89 0 0 0-1.828 1.829l-.645 1.936a.361.361 0 0 1-.686 0l-.645-1.937a2.89 2.89 0 0 0-1.828-1.828l-1.937-.645a.361.361 0 0 1 0-.686l1.937-.645a2.89 2.89 0 0 0 1.828-1.829zM3.794 1.148a.217.217 0 0 1 .412 0l.387 1.162c.173.518.579.924 1.097 1.097l1.162.387a.217.217 0 0 1 0 .412l-1.162.387A1.73 1.73 0 0 0 4.593 5.69l-.387 1.162a.217.217 0 0 1-.412 0L3.407 5.69a1.73 1.73 0 0 0-1.097-1.097l-1.162-.387a.217.217 0 0 1 0-.412l1.162-.387A1.73 1.73 0 0 0 3.407 2.31z"/>
              </svg>
              {summarizing ? "Summarizing..." : "Summarize Plans"}
            </button>
          </div>
        </div>

        <section className="bg-white border border-stone-200 rounded-lg p-5 mb-6">
          <h2 className="text-sm font-medium text-stone-500 mb-4">Score Progression</h2>
          <ScoreChart iterations={run.iterations}
            onCandidateSelect={handleCandidateSelect} />
        </section>

        <section className="bg-white border border-stone-200 rounded-lg mb-6">
          <button onClick={() => selected && setDiffOpen((v) => !v)}
            className={`w-full flex items-center justify-between px-5 py-3 text-left border-b ${
              selected && diffOpen ? "border-stone-100" : "border-transparent"
            } ${selected ? "hover:bg-stone-50 cursor-pointer" : "cursor-default"}`}>
            <h2 className="text-sm font-medium text-stone-700">
              Candidate Summary
              {selected && (
                <span className="ml-2 font-mono text-stone-400 font-normal">
                  {selected.id}{selected.score != null && ` · ${selected.score.toFixed(3)}`}
                </span>
              )}
            </h2>
            {selected && (
              <span className="text-stone-600 text-sm font-medium bg-stone-100 rounded px-2 py-0.5">
                {diffOpen ? "Collapse" : "Expand"}
              </span>
            )}
          </button>
          {!selected && (
            <div className="px-5 py-8 text-center">
              <p className="text-stone-400 text-sm">
                Click a point in the chart or a node in the beam tree to view candidate details
              </p>
            </div>
          )}
          {selected && diffOpen && (
            <div className="px-5 pb-5">
              <PlanCard plan={selected.plan} planSummary={selected.plan_summary}
                planModel={selected.plan_model} codeModel={selected.code_model}
                score={selected.score} parentScore={parentScore} />
              {ancestorChain.length > 0 && (
                <div className="mt-3 flex items-center gap-2">
                  <span className="text-xs text-stone-400">Diff against:</span>
                  <Dropdown
                    options={ancestorChain}
                    value={effectiveDiffAncestor?.id ?? ""}
                    onChange={(id) => setDiffAncestor(id)}
                  />
                  <button onClick={handleOpenDiff}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium text-indigo-700 bg-indigo-50 hover:bg-indigo-100 transition-colors">
                    Open Diff
                  </button>
                </div>
              )}
            </div>
          )}
        </section>

        <section className="bg-white border border-stone-200 rounded-lg p-5 mb-6">
          <h2 className="text-sm font-medium text-stone-500 mb-4">Beam Tree</h2>
          <BeamTree iterations={run.iterations} selectedCandidate={selected}
            onCandidateSelect={handleCandidateSelect} />
        </section>

        <section className="bg-white border border-stone-200 rounded-lg p-5 mb-6">
          <h2 className="text-sm font-medium text-stone-500 mb-3">Failed / Rejected Attempts</h2>
          <div className="space-y-1">
            {run.iterations.filter((it) => it.failed.length > 0).map((it) => (
              <div key={it.iter}>
                <button onClick={() => setShowFailed(showFailed === it.iter ? null : it.iter)}
                  className="w-full text-left px-3 py-2 rounded hover:bg-stone-50 flex items-center justify-between text-sm">
                  <span className="font-mono text-stone-600">Iter {it.iter}</span>
                  <span className="text-stone-400 text-xs font-mono">
                    {it.beam.length} survived / {it.failed.length} rejected
                    <span className="ml-2">{showFailed === it.iter ? "▾" : "▸"}</span>
                  </span>
                </button>
                {showFailed === it.iter && (
                  <div className="ml-4 mt-1 mb-2 space-y-1.5">
                    {it.failed.map((f, fi) => (
                      <FailedItem key={fi} item={f} />
                    ))}
                  </div>
                )}
              </div>
            ))}
            {run.iterations.filter((it) => it.failed.length > 0).length === 0 && (
              <p className="text-xs text-stone-400 font-mono px-3 py-2">No failed attempt data available</p>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
