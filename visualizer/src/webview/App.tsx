import { useEffect, useState, useMemo, Fragment } from "react";
import type { RunData, RunIndexEntry, BeamCandidate, FailedCandidate, GeneratedImplementation, IterationMetrics, ModelUsage } from "./lib/types";
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

type Provider = "openai" | "anthropic" | "bedrock" | "gemini" | "vertex-ai";

const PROVIDERS: { id: Provider; label: string; placeholder: string }[] = [
  { id: "openai", label: "OpenAI", placeholder: "gpt-5.4-mini" },
  { id: "anthropic", label: "Anthropic", placeholder: "claude-haiku-4-5-20251001" },
  { id: "bedrock", label: "AWS Bedrock", placeholder: "us.anthropic.claude-haiku-4-5-20251001-v1:0" },
  { id: "gemini", label: "Google AI Studio", placeholder: "gemini-3-flash-preview" },
  { id: "vertex-ai", label: "Google Vertex AI", placeholder: "gemini-3-flash-preview" },
];

interface SettingsData {
  provider: Provider;
  model: string;
  hasApiKey: boolean;
  hasAwsSecretKey?: boolean;
  awsRegion?: string;
  gcpProject?: string;
  gcpLocation?: string;
  outputDir?: string;
}

function SettingsPage({
  settings,
  onBack,
  hasRuns,
  ingesting,
  onIngest,
}: {
  settings: SettingsData | null;
  onBack: () => void;
  hasRuns: boolean;
  ingesting: boolean;
  onIngest: (dir: string) => void;
}) {
  const [provider, setProvider] = useState<Provider>(settings?.provider ?? "openai");
  const [model, setModel] = useState(settings?.model ?? "");
  const [apiKey, setApiKey] = useState("");
  const [awsSecretKey, setAwsSecretKey] = useState("");
  const [awsRegion, setAwsRegion] = useState(settings?.awsRegion ?? "us-east-1");
  const [gcpProject, setGcpProject] = useState(settings?.gcpProject ?? "");
  const [gcpLocation, setGcpLocation] = useState(settings?.gcpLocation ?? "global");
  const [outputDir, setOutputDir] = useState(settings?.outputDir ?? "");
  const [saved, setSaved] = useState(false);
  const [validating, setValidating] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validationOk, setValidationOk] = useState(false);

  useEffect(() => {
    const handler = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.type === "validateResult") {
        setValidating(false);
        if (msg.error) {
          setValidationError(msg.error);
          setValidationOk(false);
        } else {
          setValidationError(null);
          setValidationOk(true);
          setTimeout(() => setValidationOk(false), 4000);
        }
      }
    };
    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, []);

  useEffect(() => {
    if (settings) {
      setProvider(settings.provider);
      setModel(settings.model);
      setAwsRegion(settings.awsRegion ?? "us-east-1");
      setGcpProject(settings.gcpProject ?? "");
      setGcpLocation(settings.gcpLocation ?? "global");
      if (settings.outputDir) setOutputDir(settings.outputDir);
    }
  }, [settings]);

  const provInfo = PROVIDERS.find((p) => p.id === provider)!;
  const isBedrock = provider === "bedrock";
  const isVertexAI = provider === "vertex-ai";
  const needsApiKey = !isBedrock && !isVertexAI;

  const handleSave = () => {
    vscode.postMessage({
      type: "saveSettings",
      data: {
        provider,
        model: model || provInfo.placeholder,
        apiKey: apiKey || undefined,
        awsRegion: isBedrock ? (awsRegion || "us-east-1") : undefined,
        awsSecretKey: isBedrock ? (awsSecretKey || undefined) : undefined,
        gcpProject: isVertexAI ? (gcpProject || undefined) : undefined,
        gcpLocation: isVertexAI ? (gcpLocation || "global") : undefined,
      },
    });
    setApiKey("");
    setAwsSecretKey("");
    setSaved(true);
    setValidating(true);
    setValidationError(null);
    setValidationOk(false);
    setTimeout(() => setSaved(false), 2000);
    setTimeout(() => vscode.postMessage({ type: "validateSettings" }), 100);
  };

  return (
    <main className="min-h-screen bg-stone-50">
      <div className="max-w-xl mx-auto px-6 py-10">
        {hasRuns && (
          <button onClick={onBack} className="text-sm text-stone-400 hover:text-indigo-600 transition-colors mb-4">
            ← Back to traces
          </button>
        )}
        <h1 className="text-2xl font-semibold text-stone-900 tracking-tight mb-1">Autocomp Trace Visualizer</h1>
        <p className="text-stone-400 text-sm mb-8">Configure output directory and summarization settings.</p>

        <div className="bg-white border border-stone-200 rounded-lg p-5 mb-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-medium text-stone-700">Output Directory</h2>
            <span className="text-[10px] text-stone-400 uppercase tracking-wide">this workspace</span>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              placeholder="/path/to/autocomp/output"
              className="flex-1 border border-stone-200 rounded-md px-3 py-2 text-xs font-mono text-stone-700 bg-white focus:outline-none focus:ring-1 focus:ring-indigo-300"
            />
            <button
              onClick={() => vscode.postMessage({ type: "browseDir" })}
              className="px-3 py-2 rounded-md text-xs font-medium text-stone-600 bg-stone-100 hover:bg-stone-200 transition-colors whitespace-nowrap"
            >
              Browse
            </button>
          </div>
          <div className="mt-3">
            <button
              onClick={() => { if (outputDir) onIngest(outputDir); }}
              disabled={!outputDir || ingesting}
              className={`px-4 py-2 rounded-md text-xs font-semibold transition-colors ${
                !outputDir || ingesting
                  ? "text-stone-400 bg-stone-100 cursor-not-allowed"
                  : "text-white bg-indigo-600 hover:bg-indigo-700"
              }`}
            >
              {ingesting ? "Ingesting..." : "Ingest & Open Traces"}
            </button>
          </div>
        </div>

        <div className="bg-white border border-stone-200 rounded-lg p-5 space-y-5">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-stone-700">Plan Summarization</h2>
            <span className="text-[10px] text-stone-400 uppercase tracking-wide">all workspaces</span>
          </div>

          <div>
            <label className="block text-xs font-medium text-stone-500 mb-2">Provider</label>
            <div className="flex gap-1.5 flex-wrap">
              {PROVIDERS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => setProvider(p.id)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    provider === p.id
                      ? "bg-violet-100 text-violet-700 ring-1 ring-violet-300"
                      : "bg-stone-50 text-stone-500 hover:bg-stone-100 border border-stone-200"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-stone-500 mb-1">Model</label>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder={provInfo.placeholder}
              className="w-full border border-stone-200 rounded-md px-3 py-2 text-xs font-mono text-stone-700 bg-white focus:outline-none focus:ring-1 focus:ring-violet-300"
            />
          </div>

          {isBedrock && (
            <div>
              <label className="block text-xs font-medium text-stone-500 mb-1">AWS Region</label>
              <input
                type="text"
                value={awsRegion}
                onChange={(e) => setAwsRegion(e.target.value)}
                placeholder="us-east-1"
                className="w-full max-w-[14rem] border border-stone-200 rounded-md px-3 py-2 text-xs font-mono text-stone-700 bg-white focus:outline-none focus:ring-1 focus:ring-violet-300"
              />
            </div>
          )}

          {isVertexAI && (
            <>
              <div>
                <label className="block text-xs font-medium text-stone-500 mb-1">GCP Project ID</label>
                <input
                  type="text"
                  value={gcpProject}
                  onChange={(e) => setGcpProject(e.target.value)}
                  placeholder="my-gcp-project"
                  className="w-full max-w-[20rem] border border-stone-200 rounded-md px-3 py-2 text-xs font-mono text-stone-700 bg-white focus:outline-none focus:ring-1 focus:ring-violet-300"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-stone-500 mb-1">GCP Location</label>
                <input
                  type="text"
                  value={gcpLocation}
                  onChange={(e) => setGcpLocation(e.target.value)}
                  placeholder="global"
                  className="w-full max-w-[14rem] border border-stone-200 rounded-md px-3 py-2 text-xs font-mono text-stone-700 bg-white focus:outline-none focus:ring-1 focus:ring-violet-300"
                />
                <p className="text-[11px] text-stone-400 mt-1">Uses <code className="text-stone-500">gcloud auth</code> credentials from the machine.</p>
              </div>
            </>
          )}

          {!isVertexAI && (
          <div>
            <label className="block text-xs font-medium text-stone-500 mb-1">
              {isBedrock ? "Access Key ID (optional)" : "API Key"}
            </label>
            <div className="flex items-center gap-2">
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={settings?.hasApiKey ? "••••••• (saved)" : isBedrock ? "AKIA..." : "sk-..."}
                className="flex-1 border border-stone-200 rounded-md px-3 py-2 text-xs font-mono text-stone-700 bg-white focus:outline-none focus:ring-1 focus:ring-violet-300"
              />
              {settings?.hasApiKey && (
                <span className="text-emerald-600 text-[10px] font-semibold uppercase tracking-wide">saved</span>
              )}
            </div>
            {needsApiKey && (
              <p className="text-[11px] text-stone-400 mt-1">Stored securely in VS Code SecretStorage.</p>
            )}
            {settings?.hasApiKey && (
              <button
                onClick={() => vscode.postMessage({ type: "clearKey", provider })}
                className="text-[11px] text-red-500 hover:text-red-600 underline mt-1"
              >
                Remove saved {isBedrock ? "credentials" : "key"}
              </button>
            )}
          </div>
          )}

          {isBedrock && (
            <div>
              <label className="block text-xs font-medium text-stone-500 mb-1">Secret Access Key (optional)</label>
              <div className="flex items-center gap-2">
                <input
                  type="password"
                  value={awsSecretKey}
                  onChange={(e) => setAwsSecretKey(e.target.value)}
                  placeholder={settings?.hasAwsSecretKey ? "••••••• (saved)" : ""}
                  className="flex-1 border border-stone-200 rounded-md px-3 py-2 text-xs font-mono text-stone-700 bg-white focus:outline-none focus:ring-1 focus:ring-violet-300"
                />
                {settings?.hasAwsSecretKey && (
                  <span className="text-emerald-600 text-[10px] font-semibold uppercase tracking-wide">saved</span>
                )}
              </div>
              <p className="text-[11px] text-stone-400 mt-1">Leave both blank to use EC2 instance profile / environment credentials.</p>
            </div>
          )}

          <div className="pt-1">
            <div className="flex items-center gap-3">
              <button
                onClick={handleSave}
                disabled={validating}
                className={`px-4 py-2 rounded-md text-xs font-semibold transition-colors ${
                  validating
                    ? "text-stone-400 bg-stone-100 cursor-wait"
                    : "text-white bg-violet-600 hover:bg-violet-700"
                }`}
              >
                {validating ? "Validating..." : "Save & Validate"}
              </button>
              {saved && !validating && !validationError && !validationOk && (
                <span className="text-xs text-emerald-600 font-medium">Saved</span>
              )}
              {validationOk && (
                <span className="text-xs text-emerald-600 font-medium">Connected successfully</span>
              )}
            </div>
            {validationError && (
              <div className="mt-2 rounded-md border border-red-200 bg-red-50 px-3 py-2">
                <p className="text-xs font-medium text-red-700">Connection failed</p>
                <p className="text-[11px] text-red-600 mt-0.5 font-mono break-all">{validationError}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

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

function GeneratedItem({ item }: { item: GeneratedImplementation }) {
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
          {!item.correct ? <span className="text-red-500">✗</span> : item.kept ? <span className="text-emerald-500">✓</span> : <span className="text-amber-500">~</span>}
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
          {item.kept && (
            <span className="rounded bg-emerald-50 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-700">
              kept
            </span>
          )}
          {item.score !== null && <span>{formatScore(item.score)}</span>}
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

function formatScore(n: number, maxDecimals = 3): string {
  return parseFloat(n.toFixed(maxDecimals)).toString();
}

function formatDuration(seconds?: number): string {
  if (seconds == null || isNaN(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s.toFixed(0)}s`;
}

function formatTokens(n?: number): string {
  if (n == null || isNaN(n)) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function PhaseTable({ label, data }: { label: string; data?: Record<string, ModelUsage> }) {
  if (!data || Object.keys(data).length === 0) return null;
  const [open, setOpen] = useState(false);
  const entries = Object.entries(data);
  const totals = entries.reduce(
    (acc, [, v]) => ({
      calls: acc.calls + (v.calls ?? 0),
      input_tokens: acc.input_tokens + (v.input_tokens ?? 0),
      output_tokens: acc.output_tokens + (v.output_tokens ?? 0),
      duration_s: acc.duration_s + (v.duration_s ?? 0),
    }),
    { calls: 0, input_tokens: 0, output_tokens: 0, duration_s: 0 },
  );
  const getSlowest = (usage: ModelUsage) => {
    if (usage.max_duration_s != null) return usage.max_duration_s;
    return (usage.duration_s ?? 0) / Math.max(usage.calls ?? 1, 1);
  };
  const overallSlowest = Math.max(...entries.map(([, v]) => getSlowest(v)));
  return (
    <div className="mb-1">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 w-full text-left text-xs py-1 hover:bg-stone-100 rounded px-1 -mx-1 transition-colors"
      >
        <span className="text-[10px] text-stone-400">{open ? "▾" : "▸"}</span>
        <span className="font-medium text-stone-500">{label}</span>
        <span className="text-stone-400 font-mono">
          {totals.calls} calls · {formatTokens(totals.input_tokens)} in / {formatTokens(totals.output_tokens)} out
        </span>
      </button>
      {open && (
        <table className="w-full text-xs font-mono mt-1 ml-3">
          <thead>
            <tr className="text-stone-400 text-left">
              <th className="pr-3 py-0.5 font-normal">Model</th>
              <th className="pr-3 py-0.5 font-normal text-right">Calls</th>
              <th className="pr-3 py-0.5 font-normal text-right">In Tokens</th>
              <th className="pr-3 py-0.5 font-normal text-right">Out Tokens</th>
              <th className="py-0.5 font-normal text-right">Avg</th>
              <th className="py-0.5 font-normal text-right">Slowest</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([model, usage]) => {
              const calls = Math.max(usage.calls ?? 1, 1);
              const avg = (usage.duration_s ?? 0) / calls;
              const slow = usage.max_duration_s ?? avg;
              return (
              <tr key={model} className="text-stone-600">
                <td className="pr-3 py-0.5 truncate max-w-[200px]" title={model}>{formatModel(model)}</td>
                <td className="pr-3 py-0.5 text-right">{usage.calls ?? 0}</td>
                <td className="pr-3 py-0.5 text-right">{formatTokens(usage.input_tokens)}</td>
                <td className="pr-3 py-0.5 text-right">{formatTokens(usage.output_tokens)}</td>
                <td className="py-0.5 text-right">{formatDuration(avg)}</td>
                <td className="py-0.5 text-right">{formatDuration(slow)}</td>
              </tr>
              );
            })}
            {entries.length > 1 && (
              <tr className="text-stone-800 font-medium border-t border-stone-100">
                <td className="pr-3 py-0.5">Total</td>
                <td className="pr-3 py-0.5 text-right">{totals.calls}</td>
                <td className="pr-3 py-0.5 text-right">{formatTokens(totals.input_tokens)}</td>
                <td className="pr-3 py-0.5 text-right">{formatTokens(totals.output_tokens)}</td>
                <td className="py-0.5 text-right">{formatDuration(totals.duration_s / Math.max(totals.calls, 1))}</td>
                <td className="py-0.5 text-right">{formatDuration(overallSlowest)}</td>
              </tr>
            )}
          </tbody>
        </table>
      )}
    </div>
  );
}

function IterationMetricsRow({ metrics }: { metrics: IterationMetrics }) {
  const [open, setOpen] = useState(false);
  const hasPlan = metrics.plan_duration_s != null;
  const hasCode = metrics.code_duration_s != null;
  const hasEval = !!metrics.evaluation;

  return (
    <div className="rounded border border-stone-100 bg-stone-50 px-3 py-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-3 w-full text-left text-xs hover:bg-stone-100 rounded px-1 -mx-1 py-0.5 transition-colors"
      >
        <span className="text-[10px] text-stone-400">{open ? "▾" : "▸"}</span>
        <span className="font-mono font-medium text-stone-700">Iter {metrics.iteration}</span>
        <span className="text-stone-400">{formatDuration(metrics.iteration_total_s)}</span>
        {hasPlan && <span className="text-stone-300">|</span>}
        {hasPlan && <span className="text-stone-400">Plan {formatDuration(metrics.plan_duration_s)}</span>}
        {hasCode && <span className="text-stone-400">Code {formatDuration(metrics.code_duration_s)}</span>}
        {hasEval && <span className="text-stone-400">Eval {formatDuration(metrics.evaluation!.duration_s)}{metrics.evaluation!.num_candidates != null ? ` (${metrics.evaluation!.num_candidates})` : ""}</span>}
      </button>
      {open && (
        <div className="mt-2 space-y-1 ml-4">
          <PhaseTable label="Context Selection" data={metrics.context_selection} />
          <PhaseTable label="Menu Generation" data={metrics.menu_generation} />
          <PhaseTable label="Plan Generation" data={metrics.plan_generation} />
          <PhaseTable label="Code Generation" data={metrics.code_generation} />
        </div>
      )}
    </div>
  );
}

function MetricsPanel({ run }: { run: RunData }) {
  const iterationsWithMetrics = run.iterations.filter((it) => it.metrics);
  if (iterationsWithMetrics.length === 0 && !run.run_metrics) return null;

  const [open, setOpen] = useState(false);
  const rm = run.run_metrics;

  return (
    <section className="bg-white border border-stone-200 rounded-lg mb-6">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 w-full text-left px-5 py-3 hover:bg-stone-50 transition-colors rounded-lg"
      >
        <span className="text-[10px] text-stone-400">{open ? "▾" : "▸"}</span>
        <span className="text-sm font-medium text-stone-500">Metrics</span>
        {rm && (
          <span className="text-xs text-stone-400 font-mono ml-1">
            {formatDuration(rm.run_total_s)} · {formatTokens(rm.total_input_tokens)} in / {formatTokens(rm.total_output_tokens)} out
          </span>
        )}
      </button>

      {open && (
        <div className="px-5 pb-4">
          {rm && (
            <div className="flex flex-wrap gap-3 mb-3 text-xs font-mono">
              <div className="rounded bg-indigo-50 px-2.5 py-1.5">
                <span className="text-stone-400 text-[10px] uppercase tracking-wide mr-1.5">Total</span>
                <span className="text-indigo-700 font-medium">{formatDuration(rm.run_total_s)}</span>
              </div>
              <div className="rounded bg-indigo-50 px-2.5 py-1.5">
                <span className="text-stone-400 text-[10px] uppercase tracking-wide mr-1.5">LLM</span>
                <span className="text-indigo-700 font-medium">{formatDuration(rm.total_llm_duration_s)}</span>
              </div>
              <div className="rounded bg-indigo-50 px-2.5 py-1.5">
                <span className="text-stone-400 text-[10px] uppercase tracking-wide mr-1.5">Eval</span>
                <span className="text-indigo-700 font-medium">{formatDuration(rm.total_eval_duration_s)}</span>
              </div>
              <div className="rounded bg-emerald-50 px-2.5 py-1.5">
                <span className="text-stone-400 text-[10px] uppercase tracking-wide mr-1.5">In</span>
                <span className="text-emerald-700 font-medium">{formatTokens(rm.total_input_tokens)}</span>
              </div>
              <div className="rounded bg-emerald-50 px-2.5 py-1.5">
                <span className="text-stone-400 text-[10px] uppercase tracking-wide mr-1.5">Out</span>
                <span className="text-emerald-700 font-medium">{formatTokens(rm.total_output_tokens)}</span>
              </div>
            </div>
          )}

          <div className="space-y-1.5">
            {iterationsWithMetrics.map((it) => (
              <IterationMetricsRow key={it.iter} metrics={it.metrics!} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [runs, setRuns] = useState<RunIndexEntry[]>([]);
  const [run, setRun] = useState<RunData | null>(null);
  const [runFile, setRunFile] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<BeamCandidate | null>(null);
  const [diffOpen, setDiffOpen] = useState(true);
  const [summarizing, setSummarizing] = useState(false);
  const [showGenerated, setShowGenerated] = useState<number | null>(null);
  const [page, setPage] = useState<"settings" | "main">("settings");
  const [settings, setSettings] = useState<SettingsData | null>(null);
  const [ingesting, setIngesting] = useState(false);

  useEffect(() => {
    const handler = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.type === "runs") {
        setRuns(msg.data);
        setLoading(false);
        setIngesting(false);
        setPage("main");
      } else if (msg.type === "runData") {
        setRun(msg.data);
        setSelected(null);
        setLoading(false);
        setSummarizing(false);
      } else if (msg.type === "summarizeResult") {
        setSummarizing(false);
      } else if (msg.type === "settings") {
        setSettings(msg.data);
      } else if (msg.type === "dirSelected") {
        setSettings((prev) => prev ? { ...prev, outputDir: msg.dir } : prev);
      } else if (msg.type === "ingestError") {
        setIngesting(false);
      } else if (msg.type === "showSettings") {
        setPage("settings");
        vscode.postMessage({ type: "getSettings" });
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
        label: `${parent.id}${parent.score != null ? ` (${formatScore(parent.score)})` : ""}`,
        code: parent.code ?? "",
      });
      cur = parent;
    }
    const original = allCandidates[0];
    if (original && (chain.length === 0 || chain[chain.length - 1].id !== original.id)) {
      chain.push({
        id: "__original__",
        label: `Original${run?.original_score != null ? ` (${formatScore(run.original_score)})` : ""}`,
        code: original.code ?? "",
      });
    }
    return chain;
  }, [selected, allCandidates, run]);

  const [diffAncestor, setDiffAncestor] = useState<string | null>(null);

  const effectiveDiffAncestor = ancestorChain.find((a) => a.id === diffAncestor) ?? ancestorChain[0] ?? null;

  const generatedForIteration = (iteration: RunData["iterations"][number]): GeneratedImplementation[] => {
    if (iteration.generated) return iteration.generated;

    const kept = iteration.beam.map((candidate) => ({
      correct: true,
      kept: true,
      score: candidate.score,
      plan_snippet: candidate.plan ?? "",
      error_summary: null,
      model: candidate.code_model ?? candidate.plan_model ?? "",
    }));

    const rejected = iteration.failed.map((candidate: FailedCandidate) => ({
      ...candidate,
      kept: false,
    }));

    return [...kept, ...rejected];
  };

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

  if (page === "settings") {
    return (
      <SettingsPage
        settings={settings}
        onBack={() => setPage("main")}
        hasRuns={runs.length > 0}
        ingesting={ingesting}
        onIngest={(dir) => {
          setIngesting(true);
          vscode.postMessage({ type: "ingestDir", dir });
        }}
      />
    );
  }

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
          <div className="mb-8 flex items-start justify-between">
            <div>
              <h1 className="text-2xl font-semibold text-stone-900 tracking-tight">
                Autocomp Trace Visualizer
              </h1>
              <p className="text-stone-500 text-sm mt-1">
                {sortedRuns.length} optimization runs with results
              </p>
            </div>
            <button
              onClick={() => { vscode.postMessage({ type: "getSettings" }); setPage("settings"); }}
              className="p-2 rounded-md text-stone-400 hover:text-stone-600 hover:bg-stone-100 transition-colors"
              title="Settings"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
            </button>
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
                      <div className="text-[11px] text-stone-400 font-mono truncate mb-1.5" title={r.run_id}>{r.run_id}</div>
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
                          {formatScore(r.speedup, 2)}x
                        </div>
                      )}
                      <div className="text-xs text-stone-400 font-mono mt-0.5">
                        {r.original_score != null ? formatScore(r.original_score) : "—"} → {r.best_score != null ? formatScore(r.best_score) : "—"}
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
                {formatScore(run.speedup, 2)}x speedup
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
            {run.original_score != null ? formatScore(run.original_score) : "—"} → {run.best_score != null ? formatScore(run.best_score) : "—"}
          </div>
          <details className="mt-2 group">
            <summary className="text-xs text-stone-400 hover:text-stone-600 cursor-pointer select-none transition-colors">
              Run config
            </summary>
            <div className="mt-2 rounded-lg border border-stone-150 bg-stone-50 px-4 py-3">
              <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs font-mono">
                <dt className="text-stone-400">iterations</dt><dd className="text-stone-700">{run.iterations.length - 1}</dd>
                {Object.entries(run.config)
                  .filter(([k, v]) => k !== "raw_name" && k !== "models" && v != null && v !== "")
                  .map(([k, v]) => (
                    <Fragment key={k}>
                      <dt className="text-stone-400">{k}</dt>
                      <dd className="text-stone-700 break-all">
                        {Array.isArray(v) ? v.map(formatModel).join(", ") : String(v)}
                      </dd>
                    </Fragment>
                  ))}
                <dt className="text-stone-400">raw_name</dt><dd className="text-stone-700 break-all">{run.config.raw_name}</dd>
              </dl>
            </div>
          </details>
          <div className="mt-3 flex items-center gap-2">
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
            <button
              onClick={() => {
                vscode.postMessage({ type: "getSettings" });
                setPage("settings");
              }}
              title="Summarization settings"
              className="inline-flex items-center justify-center w-8 h-8 rounded text-stone-400 hover:text-violet-600 hover:bg-violet-50 transition-colors"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
            </button>
          </div>
        </div>

        <section className="bg-white border border-stone-200 rounded-lg p-5 mb-6">
          <h2 className="text-sm font-medium text-stone-500 mb-4">Score Progression</h2>
          <ScoreChart iterations={run.iterations}
            onCandidateSelect={handleCandidateSelect} />
        </section>

        <MetricsPanel run={run} />

        <section className="bg-white border border-stone-200 rounded-lg mb-6">
          <button onClick={() => selected && setDiffOpen((v) => !v)}
            className={`w-full flex items-center justify-between px-5 py-3 text-left border-b ${
              selected && diffOpen ? "border-stone-100" : "border-transparent"
            } ${selected ? "hover:bg-stone-50 cursor-pointer" : "cursor-default"}`}>
            <h2 className="text-sm font-medium text-stone-700">
              Candidate Summary
              {selected && (
                <span className="ml-2 font-mono text-stone-400 font-normal">
                  {selected.id}{selected.score != null && ` · ${formatScore(selected.score)}`}
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
          <h2 className="text-sm font-medium text-stone-500 mb-3">Generated Implementations</h2>
          <div className="space-y-1">
            {run.iterations
              .map((it) => ({ it, generated: generatedForIteration(it) }))
              .filter(({ generated }) => generated.length > 0)
              .map(({ it, generated }) => {
                const correctCount = generated.filter((item) => item.correct).length;
                const failedCount = generated.length - correctCount;
                return (
              <div key={it.iter}>
                <button onClick={() => setShowGenerated(showGenerated === it.iter ? null : it.iter)}
                  className="w-full text-left px-3 py-2 rounded hover:bg-stone-50 flex items-center justify-between text-sm">
                  <span className="font-mono text-stone-600">Iter {it.iter}</span>
                  <span className="text-stone-400 text-xs font-mono">
                    {correctCount} correct / {failedCount} failed / {generated.length} total
                    <span className="ml-2">{showGenerated === it.iter ? "▾" : "▸"}</span>
                  </span>
                </button>
                {showGenerated === it.iter && (
                  <div className="ml-4 mt-1 mb-2 space-y-1.5">
                    {generated.map((item, index) => (
                      <GeneratedItem key={index} item={item} />
                    ))}
                  </div>
                )}
              </div>
            );})}
            {run.iterations.every((it) => generatedForIteration(it).length === 0) && (
              <p className="text-xs text-stone-400 font-mono px-3 py-2">No generated implementation data available</p>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
