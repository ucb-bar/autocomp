import { useMemo } from "react";
import type { IterationData, BeamCandidate } from "../lib/types";

import { formatModel } from "../lib/format";

interface BeamTreeProps {
  iterations: IterationData[];
  selectedCandidate?: BeamCandidate | null;
  onCandidateSelect: (candidate: BeamCandidate | null) => void;
}

const NODE_WIDTH = 120;
const NODE_HEIGHT = 52;
const ROW_GAP = 60;
const COL_GAP = 20;
const PADDING = 40;

function formatScore(score: number | null): string {
  if (score === null) return "N/A";
  return parseFloat(score.toFixed(3)).toString();
}

function scoreColor(
  score: number | null,
  minScore: number,
  maxScore: number,
): { bg: string; text: string; subtext: string } {
  if (score === null) return { bg: "#f5f5f4", text: "#57534e", subtext: "#a8a29e" };
  const range = maxScore - minScore;
  if (range === 0) return { bg: "#e0e7ff", text: "#3730a3", subtext: "#6366f1" };
  const linear = (score - minScore) / range;
  const t = 1 - Math.log1p(linear * 99) / Math.log(100);
  const lightness = 96 - t * 18;
  const saturation = 20 + t * 50;
  const bg = `hsl(235, ${saturation}%, ${lightness}%)`;
  const text = t > 0.6 ? "#312e81" : "#44403c";
  const subtext = t > 0.6 ? "#4f46e5" : "#78716c";
  return { bg, text, subtext };
}

export default function BeamTree({
  iterations,
  selectedCandidate,
  onCandidateSelect,
}: BeamTreeProps) {
  const { nodes, edges, svgWidth, svgHeight, minScore, maxScore } =
    useMemo(() => {
      if (!iterations.length) {
        return { nodes: [], edges: [], svgWidth: 0, svgHeight: 0, minScore: 0, maxScore: 0 };
      }

      const filteredIters = iterations.map((it) => ({
        ...it,
        beam: it.beam.filter((c) => !c.is_carry_forward),
      }));

      const allCandidates = filteredIters.flatMap((it) => it.beam);
      const scores = allCandidates.map((c) => c.score).filter((s): s is number => s !== null);
      const min = scores.length ? Math.min(...scores) : 0;
      const max = scores.length ? Math.max(...scores) : 0;

      const maxBeamWidth = Math.max(...filteredIters.map((it) => it.beam.length), 1);
      const activeIters = filteredIters.filter((it) => it.beam.length > 0);
      const numRows = activeIters.length;

      const totalWidth = maxBeamWidth * NODE_WIDTH + (maxBeamWidth - 1) * COL_GAP + PADDING * 2;
      const totalHeight = numRows * NODE_HEIGHT + (numRows - 1) * ROW_GAP + PADDING * 2;

      const posMap = new Map<string, { x: number; y: number }>();
      const nodeList: { candidate: BeamCandidate; x: number; y: number; isRoot: boolean }[] = [];

      activeIters.forEach((iteration, rowIdx) => {
        const count = iteration.beam.length;
        const rowWidth = count * NODE_WIDTH + (count - 1) * COL_GAP;
        const offsetX = (totalWidth - rowWidth) / 2;
        const y = PADDING + rowIdx * (NODE_HEIGHT + ROW_GAP);

        iteration.beam.forEach((candidate, colIdx) => {
          const x = offsetX + colIdx * (NODE_WIDTH + COL_GAP);
          posMap.set(candidate.id, { x, y });
          nodeList.push({ candidate, x, y, isRoot: iteration.iter === 0 });
        });
      });

      const edgeList: { x1: number; y1: number; x2: number; y2: number; key: string }[] = [];
      for (const node of nodeList) {
        const { candidate, x, y } = node;
        if (candidate.parent_id && posMap.has(candidate.parent_id)) {
          const parent = posMap.get(candidate.parent_id)!;
          edgeList.push({
            x1: parent.x + NODE_WIDTH / 2, y1: parent.y + NODE_HEIGHT,
            x2: x + NODE_WIDTH / 2, y2: y,
            key: `${candidate.parent_id}->${candidate.id}`,
          });
        }
      }

      return { nodes: nodeList, edges: edgeList, svgWidth: totalWidth, svgHeight: totalHeight, minScore: min, maxScore: max };
    }, [iterations]);

  const ancestorIds = useMemo(() => {
    if (!selectedCandidate) return new Set<string>();
    const ids = new Set<string>();
    const byId = new Map<string, BeamCandidate>();
    for (const n of nodes) byId.set(n.candidate.id, n.candidate);
    let current = selectedCandidate.parent_id;
    while (current) {
      ids.add(current);
      const cand = byId.get(current);
      if (!cand) break;
      current = cand.parent_id;
    }
    return ids;
  }, [selectedCandidate, nodes]);

  const descendantIds = useMemo(() => {
    if (!selectedCandidate) return new Set<string>();
    const children = new Map<string, string[]>();
    for (const n of nodes) {
      if (n.candidate.parent_id) {
        const list = children.get(n.candidate.parent_id) ?? [];
        list.push(n.candidate.id);
        children.set(n.candidate.parent_id, list);
      }
    }
    const ids = new Set<string>();
    const queue = [selectedCandidate.id];
    while (queue.length) {
      const id = queue.pop()!;
      for (const child of children.get(id) ?? []) {
        ids.add(child);
        queue.push(child);
      }
    }
    return ids;
  }, [selectedCandidate, nodes]);

  if (!nodes.length) {
    return (
      <div className="flex items-center justify-center p-8 text-sm text-stone-400">
        No iterations to display
      </div>
    );
  }

  return (
    <div className="overflow-auto rounded-lg bg-stone-50 border border-stone-200">
      <svg width="100%" height={svgHeight} viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="block"
        preserveAspectRatio="xMinYMin meet"
        onClick={() => onCandidateSelect(null)}>
        {edges.map((e) => {
          const midY = (e.y1 + e.y2) / 2;
          const d = `M ${e.x1} ${e.y1} C ${e.x1} ${midY}, ${e.x2} ${midY}, ${e.x2} ${e.y2}`;
          const childId = e.key.split("->")[1];
          const isOnAncestorPath = selectedCandidate && (childId === selectedCandidate.id || ancestorIds.has(childId));
          const isOnDescendantPath = selectedCandidate && descendantIds.has(childId);
          const isOnPath = isOnAncestorPath || isOnDescendantPath;
          const hasSelection = !!selectedCandidate;
          return (
            <path key={e.key} d={d} fill="none"
              stroke={isOnAncestorPath ? "#818cf8" : "#c7d2fe"}
              strokeWidth={isOnAncestorPath ? 2.5 : isOnDescendantPath ? 2 : 1.5}
              opacity={hasSelection && !isOnPath ? 0.2 : isOnDescendantPath ? 0.7 : 1}
              style={{ transition: "opacity 0.15s, stroke 0.15s" }}
            />
          );
        })}

        {nodes.map(({ candidate, x, y, isRoot }) => {
          const isSelected = selectedCandidate?.id === candidate.id;
          const isAncestor = ancestorIds.has(candidate.id);
          const isDescendant = descendantIds.has(candidate.id);
          const hasSelection = !!selectedCandidate;
          const onPath = isSelected || isAncestor;
          const related = onPath || isDescendant;
          const colors = scoreColor(candidate.score, minScore, maxScore);
          const w = isRoot ? NODE_WIDTH + 8 : NODE_WIDTH;
          const h = isRoot ? NODE_HEIGHT + 6 : NODE_HEIGHT;
          const rx = isRoot ? x - 4 : x;
          const ry = isRoot ? y - 3 : y;
          const model = formatModel(candidate.plan_model ?? "");

          return (
            <g key={candidate.id} onClick={(e) => { e.stopPropagation(); onCandidateSelect(candidate); }}
              style={{ cursor: "pointer", transition: "opacity 0.15s" }}
              pointerEvents="all"
              opacity={hasSelection && !related ? 0.25 : isDescendant ? 0.7 : 1}
            >
              {isSelected && (
                <rect x={rx - 3} y={ry - 3} width={w + 6} height={h + 6} rx={10}
                  fill="none" stroke="#6366f1" strokeWidth={2.5} pointerEvents="none" />
              )}
              {isAncestor && !isSelected && (
                <rect x={rx - 2} y={ry - 2} width={w + 4} height={h + 4} rx={9}
                  fill="none" stroke="#a5b4fc" strokeWidth={1.5} strokeDasharray="4 2" pointerEvents="none" />
              )}
              {isDescendant && (
                <rect x={rx - 1.5} y={ry - 1.5} width={w + 3} height={h + 3} rx={9}
                  fill="none" stroke="#c7d2fe" strokeWidth={1} pointerEvents="none" />
              )}
              <rect x={rx} y={ry} width={w} height={h} rx={8}
                fill={colors.bg} stroke={isRoot ? "#818cf8" : "#d6d3d1"} strokeWidth={isRoot ? 1.5 : 1} />
              <text x={rx + w / 2} y={ry + (model ? 20 : 28)} textAnchor="middle"
                fill={colors.text} fontSize={11}
                style={{ fontFamily: "ui-monospace, monospace", pointerEvents: "none", fontWeight: 500 }}>
                {formatScore(candidate.score)}
              </text>
              {model && (
                <text x={rx + w / 2} y={ry + 36} textAnchor="middle"
                  fill={colors.subtext} fontSize={9} style={{ pointerEvents: "none" }}>
                  {model}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
