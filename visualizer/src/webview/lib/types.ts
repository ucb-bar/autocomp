export interface RunConfig {
  raw_name: string;
  models: string[];
  plan_models?: string[];
  code_models?: string[];
  problem?: string;
  beam_size?: number;
  num_plan_candidates?: number;
  num_code_candidates?: number;
  metric?: string;
  [key: string]: unknown;
}

export interface BeamCandidate {
  id: string;
  code: string | null;
  score: number | null;
  plan: string | null;
  plan_summary?: string;
  plan_model: string | null;
  code_model: string | null;
  parent_id: string | null;
  depth: number;
  is_carry_forward?: boolean;
  canonical_id?: string;
}

export interface FailedCandidate {
  correct: boolean;
  score: number | null;
  plan_snippet: string;
  error_summary: string | null;
  model: string;
  why_rejected?: string;
}

export interface IterationData {
  iter: number;
  beam: BeamCandidate[];
  failed: FailedCandidate[];
}

export interface RunData {
  run_id: string;
  config: RunConfig;
  original_score: number | null;
  best_score: number | null;
  speedup: number | null;
  iterations: IterationData[];
}

export interface RunIndexEntry {
  run_id: string;
  file: string;
  config: RunConfig;
  original_score: number | null;
  best_score: number | null;
  speedup: number | null;
  num_iterations: number;
}
