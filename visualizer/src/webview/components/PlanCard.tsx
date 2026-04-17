import { useState } from "react";
import { formatModel } from "../lib/format";

function fmt(n: number, maxDecimals = 3): string {
  return parseFloat(n.toFixed(maxDecimals)).toString();
}

interface PlanCardProps {
  plan: string | null;
  planSummary?: string;
  planModel?: string | null;
  codeModel?: string | null;
  score?: number | null;
  parentScore?: number | null;
}

export default function PlanCard({
  plan,
  planSummary,
  planModel,
  codeModel,
  score,
  parentScore,
}: PlanCardProps) {
  const [expanded, setExpanded] = useState(false);

  if (plan === null) return null;

  const improvement =
    score != null && parentScore != null && parentScore > 0
      ? ((parentScore - score) / parentScore) * 100
      : null;

  return (
    <div className="rounded-lg border border-stone-200 bg-white px-4 py-3" style={{ fontSize: 14 }}>
      {planSummary && (
        <p className="font-bold text-gray-900 leading-snug">{planSummary}</p>
      )}
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        {planModel && (
          <span className="inline-block rounded-full bg-blue-50 px-2 py-0.5 text-xs text-blue-700">
            plan: {formatModel(planModel)}
          </span>
        )}
        {codeModel && (
          <span className="inline-block rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700">
            code: {formatModel(codeModel)}
          </span>
        )}
        {improvement !== null && score != null && parentScore != null && (
          <span className="inline-block text-xs text-gray-600">
            {fmt(parentScore)} → {fmt(score)}{" "}
            <span className={improvement > 0 ? "font-medium text-green-700" : "text-red-600"}>
              ({improvement > 0 ? "+" : ""}{fmt(improvement, 1)}% {improvement > 0 ? "faster" : "slower"})
            </span>
          </span>
        )}
      </div>
      <button
        onClick={() => setExpanded((v) => !v)}
        className="mt-2 text-xs text-blue-600 hover:text-blue-800 hover:underline"
      >
        {expanded ? "▾ Hide full plan" : "▸ Show full plan"}
      </button>
      {expanded && (
        <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-words rounded bg-gray-50 p-3 text-xs leading-relaxed text-gray-800 font-mono">
          {plan}
        </pre>
      )}
    </div>
  );
}
