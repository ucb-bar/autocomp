import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";

export type Provider = "openai" | "anthropic" | "bedrock" | "gemini";

export interface SummarizeConfig {
  provider: Provider;
  model: string;
  apiKey: string;
  /** AWS region for Bedrock (defaults to us-east-1) */
  awsRegion?: string;
}

const SYSTEM_PROMPT = `You are a concise technical summarizer. Given an optimization plan for a code transformation, produce a 1-2 sentence summary that captures the key strategy. Focus on what the plan does, not how it was generated. Be specific about the optimization technique. Do not use filler phrases.`;

async function callOpenAI(
  config: SummarizeConfig,
  userPrompt: string,
): Promise<string> {
  const client = new OpenAI({ apiKey: config.apiKey });
  const resp = await client.chat.completions.create({
    model: config.model,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ],
    max_completion_tokens: 256,
    temperature: 0.3,
  });
  return resp.choices[0]?.message?.content?.trim() ?? "";
}

async function callAnthropic(
  config: SummarizeConfig,
  userPrompt: string,
): Promise<string> {
  const client = new Anthropic({ apiKey: config.apiKey });
  const resp = await client.messages.create({
    model: config.model,
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content: userPrompt }],
    max_tokens: 256,
    temperature: 0.3,
  });
  const block = resp.content[0];
  return block.type === "text" ? block.text.trim() : "";
}

async function callBedrock(
  config: SummarizeConfig,
  userPrompt: string,
): Promise<string> {
  const client = new BedrockRuntimeClient({
    region: config.awsRegion ?? "us-east-1",
  });
  const body = JSON.stringify({
    anthropic_version: "bedrock-2023-05-31",
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content: userPrompt }],
    max_tokens: 256,
    temperature: 0.3,
  });
  const cmd = new InvokeModelCommand({
    modelId: config.model,
    contentType: "application/json",
    accept: "application/json",
    body: new TextEncoder().encode(body),
  });
  const resp = await client.send(cmd);
  const parsed = JSON.parse(new TextDecoder().decode(resp.body));
  return (parsed.content?.[0]?.text ?? "").trim();
}

async function callGemini(
  config: SummarizeConfig,
  userPrompt: string,
): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${config.model}:generateContent?key=${config.apiKey}`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents: [{ parts: [{ text: userPrompt }] }],
      generationConfig: { maxOutputTokens: 256, temperature: 0.3 },
    }),
  });
  if (!resp.ok) {
    throw new Error(`Gemini API error: ${resp.status} ${await resp.text()}`);
  }
  const data = await resp.json();
  return (
    data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? ""
  );
}

async function complete(
  config: SummarizeConfig,
  userPrompt: string,
): Promise<string> {
  switch (config.provider) {
    case "openai":
      return callOpenAI(config, userPrompt);
    case "anthropic":
      return callAnthropic(config, userPrompt);
    case "bedrock":
      return callBedrock(config, userPrompt);
    case "gemini":
      return callGemini(config, userPrompt);
    default:
      throw new Error(`Unknown provider: ${config.provider}`);
  }
}

export interface PlanToSummarize {
  candidateId: string;
  plan: string;
}

export interface SummarizeResult {
  candidateId: string;
  summary: string;
  error?: string;
}

/**
 * Summarize a batch of plans. Runs requests sequentially to avoid
 * rate-limit issues; callers can report per-plan progress.
 */
export async function summarizePlans(
  config: SummarizeConfig,
  plans: PlanToSummarize[],
  onProgress?: (done: number, total: number) => void,
): Promise<SummarizeResult[]> {
  const results: SummarizeResult[] = [];
  for (let i = 0; i < plans.length; i++) {
    const { candidateId, plan } = plans[i];
    try {
      const summary = await complete(
        config,
        `Summarize this optimization plan:\n\n${plan}`,
      );
      results.push({ candidateId, summary });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      results.push({ candidateId, summary: "", error: msg });
    }
    onProgress?.(i + 1, plans.length);
  }
  return results;
}
