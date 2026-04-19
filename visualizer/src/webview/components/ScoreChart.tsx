import { useMemo, useState, useCallback } from "react";
import type { IterationData, BeamCandidate } from "../lib/types";

interface ScoreChartProps {
  iterations: IterationData[];
  onCandidateSelect?: (candidate: BeamCandidate | null) => void;
}

interface TracePoint {
  iter: number;
  score: number;
  candidate: BeamCandidate;
}

interface Trace {
  id: string;
  points: TracePoint[];
  finalScore: number;
  isBestTrace: boolean;
}

const PALETTE = [
  "#3730a3", "#0369a1", "#047857", "#b45309", "#9333ea",
  "#be123c", "#0e7490", "#4338ca", "#a16207", "#7c3aed",
  "#0891b2", "#059669", "#d97706", "#6d28d9", "#dc2626",
];

export default function ScoreChart({
  iterations,
  onCandidateSelect,
}: ScoreChartProps) {
  const [hoveredTrace, setHoveredTrace] = useState<string | null>(null);
  const [hoveredDot, setHoveredDot] = useState<TracePoint | null>(null);
  const [selectedCandId, setSelectedCandId] = useState<string | null>(null);

  const { traces, xMax, yMax, yMin, globalBest } =
    useMemo(() => {
      const byId = new Map<string, BeamCandidate & { iter: number }>();
      for (const it of iterations) {
        for (const c of it.beam) {
          byId.set(c.id, { ...c, iter: it.iter });
        }
      }

      const lastIter = iterations.length > 0 ? iterations[iterations.length - 1] : null;
      const leaves = lastIter
        ? lastIter.beam.filter((c) => !c.is_carry_forward)
        : [];

      const traceList: Trace[] = [];
      let gBest: number | null = null;

      for (const it of iterations) {
        for (const c of it.beam) {
          if (c.score != null && (gBest === null || c.score < gBest)) {
            gBest = c.score;
          }
        }
      }

      for (const leaf of leaves) {
        const points: TracePoint[] = [];
        let current: string | null = leaf.id;
        const visited = new Set<string>();
        while (current && !visited.has(current)) {
          visited.add(current);
          const cand = byId.get(current);
          if (!cand || cand.score == null) break;
          points.unshift({ iter: cand.iter, score: cand.score, candidate: cand });
          current = cand.parent_id;
        }
        if (points.length > 0) {
          const finalScore = points[points.length - 1].score;
          traceList.push({ id: leaf.id, points, finalScore, isBestTrace: false });
        }
      }

      const tracedIds = new Set(traceList.flatMap((t) => t.points.map((p) => p.candidate.id)));
      for (let i = iterations.length - 2; i >= 1; i--) {
        for (const c of iterations[i].beam) {
          if (c.is_carry_forward || tracedIds.has(c.id)) continue;
          const points: TracePoint[] = [];
          let current: string | null = c.id;
          const visited = new Set<string>();
          while (current && !visited.has(current)) {
            visited.add(current);
            const cand = byId.get(current);
            if (!cand || cand.score == null) break;
            points.unshift({ iter: cand.iter, score: cand.score, candidate: cand });
            current = cand.parent_id;
          }
          if (points.length > 1) {
            const finalScore = points[points.length - 1].score;
            traceList.push({ id: c.id, points, finalScore, isBestTrace: false });
            for (const p of points) tracedIds.add(p.candidate.id);
          }
        }
      }

      traceList.sort((a, b) => a.finalScore - b.finalScore);
      if (traceList.length > 0) {
        traceList[0].isBestTrace = true;
      }

      let xM = 0;
      let yM = 0;
      let yMn = Infinity;
      for (const t of traceList) {
        for (const p of t.points) {
          xM = Math.max(xM, p.iter);
          yM = Math.max(yM, p.score);
          yMn = Math.min(yMn, p.score);
        }
      }

      return {
        traces: traceList,
        xMax: xM,
        yMax: yM,
        yMin: yMn === Infinity ? 0 : yMn,
        globalBest: gBest,
      };
    }, [iterations]);

  const handleClick = useCallback(
    (candidate: BeamCandidate) => {
      setSelectedCandId(candidate.id);
      onCandidateSelect?.(candidate);
    },
    [onCandidateSelect],
  );

  if (iterations.length === 0 || traces.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center font-mono text-sm text-stone-400">
        No iteration data
      </div>
    );
  }

  const W = 780;
  const H = 380;
  const PAD = { top: 30, right: 54, bottom: 50, left: 64 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const yPad = (yMax - yMin) * 0.08 || 0.1;
  const yDomainMin = Math.max(0, yMin - yPad);
  const yDomainMax = yMax + yPad;

  const xScale = (x: number) => PAD.left + (x / Math.max(xMax, 1)) * plotW;
  const yScale = (y: number) =>
    PAD.top + (1 - (y - yDomainMin) / (yDomainMax - yDomainMin)) * plotH;

  const xTicks = Array.from({ length: xMax + 1 }, (_, i) => i);
  const yTickCount = 6;
  const yTicks = Array.from({ length: yTickCount }, (_, i) => {
    return yDomainMin + (i / (yTickCount - 1)) * (yDomainMax - yDomainMin);
  });

  const infoDot = hoveredDot;
  const infoTrace = hoveredTrace ? traces.find((t) => t.id === hoveredTrace) : null;

  return (
    <div>
      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full"
          style={{ maxHeight: 420, fontFamily: "ui-monospace, monospace" }}
          onClick={() => { setSelectedCandId(null); onCandidateSelect?.(null); }}
        >
          {yTicks.map((y, i) => (
            <line
              key={`yg${i}`}
              x1={PAD.left} y1={yScale(y)}
              x2={W - PAD.right} y2={yScale(y)}
              stroke="#e7e5e4" strokeDasharray="3 3"
            />
          ))}
          {yTicks.map((y, i) => (
            <text key={`yl${i}`} x={PAD.left - 8} y={yScale(y) + 3}
              textAnchor="end" className="fill-stone-400" fontSize={10}>
              {y.toFixed(y < 1 ? 2 : 1)}
            </text>
          ))}
          <text x={14} y={PAD.top + plotH / 2} textAnchor="middle"
            transform={`rotate(-90, 14, ${PAD.top + plotH / 2})`}
            className="fill-stone-400" fontSize={11}>
            Score
          </text>
          {xTicks.map((x) => (
            <text key={`xl${x}`} x={xScale(x)} y={H - PAD.bottom + 18}
              textAnchor="middle" className="fill-stone-400" fontSize={10}>
              {x}
            </text>
          ))}
          <text x={PAD.left + plotW / 2} y={H - 6} textAnchor="middle"
            className="fill-stone-400" fontSize={11}>
            Iteration
          </text>

          {globalBest !== null && (
            <>
              <line
                x1={PAD.left} y1={yScale(globalBest)}
                x2={W - PAD.right} y2={yScale(globalBest)}
                stroke="#059669" strokeWidth={1} strokeDasharray="4 3"
                opacity={0.5}
              />
              <text
                x={W - PAD.right + 4} y={yScale(globalBest) + 3}
                fontSize={9} className="fill-emerald-600" fontWeight={600}
              >
                {globalBest.toFixed(globalBest < 1 ? 3 : 2)}
              </text>
            </>
          )}

          {(() => {
            const selectedTraceId = selectedCandId
              ? traces.find((t) => t.points.some((p) => p.candidate.id === selectedCandId))?.id ?? null
              : null;
            const ordered = selectedTraceId
              ? [...traces.filter((t) => t.id !== selectedTraceId), ...traces.filter((t) => t.id === selectedTraceId)]
              : traces;
            return ordered.map((trace) => {
              const ti = traces.indexOf(trace);
              const color = trace.isBestTrace
                ? PALETTE[0]
                : PALETTE[(ti % (PALETTE.length - 1)) + 1];
              const isHovered = hoveredTrace === trace.id;
              const hasSelection = selectedCandId !== null;
              const traceHasSelected = trace.points.some(
                (p) => p.candidate.id === selectedCandId,
              );
              const dimmed = hasSelection && !traceHasSelected && !isHovered;

              const pathD = trace.points
                .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.iter)} ${yScale(p.score)}`)
                .join(" ");

              return (
                <g key={trace.id}>
                  <path
                    d={pathD} fill="none" stroke="transparent" strokeWidth={14}
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => { setHoveredTrace(trace.id); setHoveredDot(null); }}
                    onMouseLeave={() => { setHoveredTrace(null); setHoveredDot(null); }}
                  />
                  <path
                    d={pathD} fill="none" stroke={color}
                    strokeWidth={trace.isBestTrace ? 2.5 : isHovered ? 2 : 1.5}
                    strokeLinecap="round" strokeLinejoin="round"
                    opacity={dimmed ? 0.15 : isHovered || traceHasSelected ? 1 : 0.6}
                    style={{ transition: "opacity 0.15s", pointerEvents: "none" }}
                  />
                  {trace.points.map((p) => {
                    const isSelected = p.candidate.id === selectedCandId;
                    const isDotHovered = hoveredDot?.candidate.id === p.candidate.id;
                    const r = isSelected ? 6 : isDotHovered ? 5.5 : isHovered ? 4.5 : trace.isBestTrace ? 4 : 3.5;
                    return (
                      <circle
                        key={p.candidate.id}
                        cx={xScale(p.iter)} cy={yScale(p.score)} r={r}
                        fill={isSelected ? "#f59e0b" : color}
                        stroke={isSelected ? "#d97706" : "white"}
                        strokeWidth={isSelected ? 2 : 1.5}
                        opacity={dimmed ? 0.2 : 1}
                        style={{ cursor: "pointer", transition: "r 0.1s, opacity 0.15s" }}
                        onMouseEnter={() => { setHoveredTrace(trace.id); setHoveredDot(p); }}
                        onMouseLeave={() => { setHoveredTrace(null); setHoveredDot(null); }}
                        onClick={(e) => { e.stopPropagation(); handleClick(p.candidate); }}
                      />
                    );
                  })}
                </g>
              );
            });
          })()}
        </svg>
      </div>

      <div className="h-10 px-2 flex items-center gap-3 font-mono text-xs text-stone-500 border-t border-stone-100">
        {(() => {
          const selectedPoint = selectedCandId
            ? traces.flatMap((t) => t.points).find((p) => p.candidate.id === selectedCandId) ?? null
            : null;
          const dot = infoDot ?? selectedPoint;
          if (dot) {
            return (
              <>
                <span className="font-semibold text-stone-800">
                  iter {dot.iter}
                </span>
                <span className="text-stone-700">{dot.score.toFixed(3)}</span>
                {dot.candidate.plan_summary ? (
                  <span className="text-stone-500 truncate">
                    {dot.candidate.plan_summary}
                  </span>
                ) : dot.iter === 0 ? (
                  <span className="text-stone-300 italic">original code</span>
                ) : null}
              </>
            );
          }
          if (infoTrace) {
            return (
              <>
                <span className="font-semibold text-stone-800">
                  {infoTrace.points[0].score.toFixed(3)} → {infoTrace.finalScore.toFixed(3)}
                </span>
                <span className="text-emerald-600 font-semibold">
                  {(infoTrace.points[0].score / infoTrace.finalScore).toFixed(1)}x
                </span>
                <span className="text-stone-400">
                  {infoTrace.points.length} steps
                </span>
              </>
            );
          }
          return <span className="text-stone-300">Hover a dot or trace to see details</span>;
        })()}
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1 px-2 pt-2">
        {(() => {
          const legendTraces = traces.slice(0, 12);
          const speedups = legendTraces.map((t) => {
            const first = t.points[0];
            const last = t.points[t.points.length - 1];
            return first.score > 0 ? first.score / last.score : 0;
          });
          const bestSpeedup = Math.max(...speedups);
          return legendTraces.map((trace, ti) => {
            const color = trace.isBestTrace
              ? PALETTE[0]
              : PALETTE[(ti % (PALETTE.length - 1)) + 1];
            const last = trace.points[trace.points.length - 1];
            const traceSpeedup = speedups[ti];
            const isBestSpeedup = traceSpeedup === bestSpeedup && bestSpeedup > 1;
            const isTraceSelected = trace.points.some(
              (p) => p.candidate.id === selectedCandId,
            );
            return (
              <button
                key={trace.id}
                className={`flex items-center gap-1.5 text-xs font-mono transition-colors ${
                  isTraceSelected
                    ? "text-stone-900 font-semibold"
                    : "text-stone-500 hover:text-stone-800"
                }`}
                onMouseEnter={() => setHoveredTrace(trace.id)}
                onMouseLeave={() => setHoveredTrace(null)}
                onClick={() => handleClick(last.candidate)}
              >
                <span
                  className="inline-block w-3 h-0.5 rounded"
                  style={{ backgroundColor: color }}
                />
                {last.score.toFixed(3)}
                {traceSpeedup > 1 && (
                  <span className={isBestSpeedup ? "text-emerald-600 font-semibold" : "text-stone-400"}>
                    ({traceSpeedup.toFixed(1)}x)
                  </span>
                )}
              </button>
            );
          });
        })()}
      </div>
    </div>
  );
}
