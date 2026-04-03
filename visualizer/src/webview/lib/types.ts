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

export interface ModelUsage {
  calls?: number;
  input_tokens?: number;
  output_tokens?: number;
  duration_s?: number;
  max_duration_s?: number;
  [key: string]: unknown;
}

export interface EvaluationMetrics {
  duration_s?: number;
  num_candidates?: number;
  [key: string]: unknown;
}

export interface IterationMetrics {
  iteration: number;
  iteration_total_s?: number;
  plan_duration_s?: number;
  code_duration_s?: number;
  plan_generation?: Record<string, ModelUsage>;
  code_generation?: Record<string, ModelUsage>;
  context_selection?: Record<string, ModelUsage>;
  menu_generation?: Record<string, ModelUsage>;
  evaluation?: EvaluationMetrics;
  [key: string]: unknown;
}

export interface RunMetrics {
  run_total_s?: number;
  total_input_tokens?: number;
  total_output_tokens?: number;
  total_llm_duration_s?: number;
  total_eval_duration_s?: number;
  iterations?: IterationMetrics[];
  [key: string]: unknown;
}

export interface IterationData {
  iter: number;
  beam: BeamCandidate[];
  failed: FailedCandidate[];
  generated?: GeneratedImplementation[];
  metrics?: IterationMetrics;
}

export interface RunData {
  run_id: string;
  config: RunConfig;
  original_score: number | null;
  best_score: number | null;
  speedup: number | null;
  iterations: IterationData[];
  run_metrics?: RunMetrics;
}

export interface GeneratedImplementation {
  correct: boolean;
  kept?: boolean;
  score: number | null;
  plan_snippet: string;
  error_summary: string | null;
  model: string;
  why_rejected?: string;
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
