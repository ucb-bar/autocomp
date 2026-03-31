/**
 * Clean up a model name for display by stripping the provider:: prefix.
 * e.g. "aws::us.anthropic.claude-opus-4-5-20251101-v1:0" → "us.anthropic.claude-opus-4-5-20251101-v1:0"
 */
export function formatModel(model: string): string {
  return model.includes("::") ? model.split("::").pop()! : model;
}
