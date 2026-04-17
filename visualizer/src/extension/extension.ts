import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import { ingestOutputDir } from "./ingest";
import {
  summarizePlans,
  validateConfig,
  type Provider,
  type SummarizeConfig,
  type PlanToSummarize,
} from "./summarize";

const SCHEME = "autocomp-code";
const SETTINGS_KEY = "autocomp.summarizeSettings";
const LAST_DIR_KEY = "autocomp.lastOutputDir";

let codeContentStore = new Map<string, string>();
let currentPanel: vscode.WebviewPanel | undefined;

class CodeContentProvider implements vscode.TextDocumentContentProvider {
  provideTextDocumentContent(uri: vscode.Uri): string {
    return codeContentStore.get(uri.path) ?? "";
  }
}

interface StoredSettings {
  provider: Provider;
  model: string;
  models?: Partial<Record<Provider, string>>;
  awsRegion?: string;
  gcpProject?: string;
  gcpLocation?: string;
}

function getStoredSettings(context: vscode.ExtensionContext): StoredSettings {
  return context.globalState.get<StoredSettings>(SETTINGS_KEY) ?? {
    provider: "openai",
    model: "gpt-5.4-mini",
  };
}

async function getApiKey(
  context: vscode.ExtensionContext,
  provider: Provider,
): Promise<string | undefined> {
  return context.secrets.get(`autocomp.apiKey.${provider}`);
}

async function sendSettings(
  context: vscode.ExtensionContext,
  webview: vscode.Webview,
) {
  const settings = getStoredSettings(context);
  const hasKey = !!(await getApiKey(context, settings.provider));
  const hasAwsSecretKey = !!(await context.secrets.get("autocomp.awsSecretKey"));
  const lastDir = context.workspaceState.get<string>(LAST_DIR_KEY) ?? "";
  const hasCache = lastDir
    ? fs.existsSync(path.join(lastDir, ".visualizer-data", "runs.json"))
    : false;
  webview.postMessage({
    type: "settings",
    data: { ...settings, hasApiKey: hasKey, hasAwsSecretKey, outputDir: lastDir, hasCache },
  });
}

export function activate(context: vscode.ExtensionContext) {
  const contentProvider = new CodeContentProvider();
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(SCHEME, contentProvider),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("autocomp.openTraces", () => {
      openPanel(context);
    }),
  );
}

function loadRunsIndex(dataDir: string, panel: vscode.WebviewPanel): void {
  const runsJson = path.join(dataDir, "runs.json");
  if (!fs.existsSync(runsJson)) {
    vscode.window.showErrorMessage("Ingestion produced no runs.json");
    panel.webview.postMessage({ type: "ingestError", error: "No runs found" });
    return;
  }
  try {
    const raw = JSON.parse(fs.readFileSync(runsJson, "utf-8"));
    const EXPECTED_SCHEMA = 2;
    let data: unknown[];
    if (Array.isArray(raw)) {
      data = raw;
    } else {
      if (raw.schema_version > EXPECTED_SCHEMA) {
        vscode.window.showWarningMessage(
          `runs.json schema v${raw.schema_version} is newer than expected (v${EXPECTED_SCHEMA}). ` +
          "Consider updating the Autocomp Visualizer extension.",
        );
      }
      data = raw.runs;
    }
    panel.webview.postMessage({ type: "runs", data });
  } catch {
    vscode.window.showErrorMessage("Failed to read runs.json");
  }
}

function openPanel(context: vscode.ExtensionContext) {
  if (currentPanel) {
    currentPanel.reveal(vscode.ViewColumn.One);
    return;
  }

  const panel = vscode.window.createWebviewPanel(
    "autocompVisualizer",
    "Autocomp Traces",
    vscode.ViewColumn.One,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
      localResourceRoots: [
        vscode.Uri.file(path.join(context.extensionPath, "dist", "webview")),
      ],
    },
  );
  currentPanel = panel;

  panel.onDidDispose(() => {
    currentPanel = undefined;
  });

  let dataDir: string | undefined;

  panel.webview.onDidReceiveMessage(
    async (msg) => {
      if (msg.type === "ready") {
        await sendSettings(context, panel.webview);
      } else if (msg.type === "browseDir") {
        const uris = await vscode.window.showOpenDialog({
          canSelectFolders: true,
          canSelectFiles: false,
          openLabel: "Select Autocomp output directory",
        });
        if (uris && uris.length > 0) {
          const dir = uris[0].fsPath;
          await context.workspaceState.update(LAST_DIR_KEY, dir);
          panel.webview.postMessage({ type: "dirSelected", dir });
        }
      } else if (msg.type === "ingestDir") {
        const outputDir = msg.dir as string;
        if (!outputDir || !fs.existsSync(outputDir)) {
          vscode.window.showErrorMessage(`Directory not found: ${outputDir}`);
          panel.webview.postMessage({ type: "ingestError", error: "Directory not found" });
          return;
        }
        await context.workspaceState.update(LAST_DIR_KEY, outputDir);
        dataDir = path.join(outputDir, ".visualizer-data");
        fs.mkdirSync(dataDir, { recursive: true });

        await vscode.window.withProgress(
          { location: vscode.ProgressLocation.Notification, title: "Ingesting traces..." },
          async () => {
            const { errors } = ingestOutputDir(outputDir, dataDir!);
            if (errors.length > 0) {
              vscode.window.showWarningMessage(`Ingestion warnings: ${errors.join("; ")}`);
            }
          },
        );

        if (msg.open !== false) {
          loadRunsIndex(dataDir, panel);
        } else {
          panel.webview.postMessage({ type: "ingestDone" });
          await sendSettings(context, panel.webview);
        }
      } else if (msg.type === "openTraces") {
        const outputDir = msg.dir as string;
        if (!outputDir || !fs.existsSync(outputDir)) {
          vscode.window.showErrorMessage(`Directory not found: ${outputDir}`);
          panel.webview.postMessage({ type: "ingestError", error: "Directory not found" });
          return;
        }
        await context.workspaceState.update(LAST_DIR_KEY, outputDir);
        dataDir = path.join(outputDir, ".visualizer-data");
        if (!fs.existsSync(path.join(dataDir, "runs.json"))) {
          panel.webview.postMessage({ type: "ingestError", error: "No cached data — ingest first" });
          vscode.window.showInformationMessage("No cached traces found. Click Re-ingest to process the output directory.");
          return;
        }
        loadRunsIndex(dataDir, panel);
      } else if (msg.type === "selectRun") {
        if (!dataDir) return;
        const runFile = path.join(dataDir, path.basename(msg.file));
        try {
          const data = JSON.parse(fs.readFileSync(runFile, "utf-8"));
          panel.webview.postMessage({ type: "runData", data });
        } catch {
          vscode.window.showErrorMessage(`Failed to read run file: ${msg.file}`);
        }
      } else if (msg.type === "openDiff") {
        const ts = Date.now();
        const leftKey = `/left/${ts}.py`;
        const rightKey = `/right/${ts}.py`;
        codeContentStore.set(leftKey, msg.parentCode);
        codeContentStore.set(rightKey, msg.candidateCode);
        const leftUri = vscode.Uri.parse(`${SCHEME}:${leftKey}`);
        const rightUri = vscode.Uri.parse(`${SCHEME}:${rightKey}`);
        vscode.commands.executeCommand(
          "vscode.diff",
          leftUri,
          rightUri,
          `${msg.parentLabel} ↔ ${msg.candidateLabel}`,
        );
      } else if (msg.type === "summarizePlans") {
        if (!dataDir) return;
        handleSummarize(context, panel, dataDir, msg.file).catch((err) => {
          vscode.window.showErrorMessage(`Summarization failed: ${err.message ?? err}`);
          panel.webview.postMessage({ type: "summarizeResult", cancelled: true });
        });
      } else if (msg.type === "getSettings") {
        await sendSettings(context, panel.webview);
      } else if (msg.type === "checkApiKey") {
        const provider = msg.provider as Provider;
        const hasKey = !!(await getApiKey(context, provider));
        panel.webview.postMessage({ type: "apiKeyStatus", provider, hasApiKey: hasKey });
      } else if (msg.type === "saveSettings") {
        const { provider, model, apiKey, awsRegion, awsSecretKey, gcpProject, gcpLocation } = msg.data as {
          provider: Provider;
          model: string;
          apiKey?: string;
          awsRegion?: string;
          awsSecretKey?: string;
          gcpProject?: string;
          gcpLocation?: string;
        };
        const existing = getStoredSettings(context);
        const models = { ...existing.models, [provider]: model };
        await context.globalState.update(SETTINGS_KEY, {
          provider,
          model,
          models,
          awsRegion: awsRegion ?? existing.awsRegion,
          gcpProject: gcpProject ?? existing.gcpProject,
          gcpLocation: gcpLocation ?? existing.gcpLocation,
        });
        if (apiKey) {
          await context.secrets.store(`autocomp.apiKey.${provider}`, apiKey);
        }
        if (awsSecretKey) {
          await context.secrets.store("autocomp.awsSecretKey", awsSecretKey);
        }
        await sendSettings(context, panel.webview);
      } else if (msg.type === "validateSettings") {
        const settings = getStoredSettings(context);
        const apiKey = await getApiKey(context, settings.provider);
        const config: SummarizeConfig = {
          provider: settings.provider,
          model: settings.model,
          apiKey: apiKey ?? "",
          awsRegion: settings.awsRegion,
          awsSecretKey: settings.provider === "bedrock"
            ? (await context.secrets.get("autocomp.awsSecretKey")) ?? undefined
            : undefined,
          gcpProject: settings.gcpProject,
          gcpLocation: settings.gcpLocation,
        };
        const error = await validateConfig(config);
        panel.webview.postMessage({ type: "validateResult", error });
      } else if (msg.type === "clearKey") {
        const provider = msg.provider as Provider;
        await context.secrets.delete(`autocomp.apiKey.${provider}`);
        if (provider === "bedrock") {
          await context.secrets.delete("autocomp.awsSecretKey");
        }
        await sendSettings(context, panel.webview);
      }
    },
    undefined,
    context.subscriptions,
  );

  panel.webview.html = getVisualizerHtml(context, panel.webview);
}

async function handleSummarize(
  context: vscode.ExtensionContext,
  panel: vscode.WebviewPanel,
  dataDir: string,
  runFileName: string,
) {
  const settings = getStoredSettings(context);
  const apiKey = await getApiKey(context, settings.provider);

  if (!apiKey && settings.provider !== "bedrock" && settings.provider !== "vertex-ai") {
    panel.webview.postMessage({ type: "summarizeResult", cancelled: true });
    panel.webview.postMessage({ type: "showSettings" });
    vscode.window.showWarningMessage(
      `No API key configured for ${settings.provider}. Please configure it in Settings.`,
    );
    return;
  }

  const runFile = path.join(dataDir, path.basename(runFileName));
  const runData = JSON.parse(fs.readFileSync(runFile, "utf-8"));

  const plans: PlanToSummarize[] = [];
  for (const iter of runData.iterations ?? []) {
    for (const cand of iter.beam ?? []) {
      if (cand.plan && !cand.plan_summary) {
        plans.push({ candidateId: cand.id, plan: cand.plan });
      }
    }
  }

  if (plans.length === 0) {
    panel.webview.postMessage({ type: "summarizeResult", cancelled: true });
    vscode.window.showInformationMessage("All plans already have summaries (or no plans found).");
    return;
  }

  const config: SummarizeConfig = {
    provider: settings.provider,
    model: settings.model,
    apiKey: apiKey ?? "",
    awsRegion: settings.awsRegion,
    awsSecretKey: settings.provider === "bedrock"
      ? (await context.secrets.get("autocomp.awsSecretKey")) ?? undefined
      : undefined,
    gcpProject: settings.gcpProject,
    gcpLocation: settings.gcpLocation,
  };

  const results = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Summarizing plans...",
      cancellable: false,
    },
    async (progress) => {
      return summarizePlans(config, plans, (done, total) => {
        progress.report({
          message: `${done}/${total}`,
          increment: (1 / total) * 100,
        });
      });
    },
  );

  const summaryMap = new Map(
    results
      .filter((r) => r.summary)
      .map((r) => [r.candidateId, r.summary]),
  );
  for (const iter of runData.iterations ?? []) {
    for (const cand of iter.beam ?? []) {
      const s = summaryMap.get(cand.id);
      if (s) cand.plan_summary = s;
    }
  }

  fs.writeFileSync(runFile, JSON.stringify(runData, null, 2), "utf-8");
  panel.webview.postMessage({ type: "runData", data: runData });

  const errors = results.filter((r) => r.error);
  if (errors.length > 0) {
    vscode.window.showWarningMessage(
      `${errors.length}/${plans.length} plans failed to summarize: ${errors[0].error}`,
    );
  }
}

function getVisualizerHtml(context: vscode.ExtensionContext, webview: vscode.Webview): string {
  const distPath = path.join(context.extensionPath, "dist", "webview");
  const scriptPath = path.join(distPath, "assets", "index.js");

  if (!fs.existsSync(scriptPath)) {
    return "<html><body><h2>Webview not built. Run <code>npm run build</code> first.</h2></body></html>";
  }

  const scriptUri = webview.asWebviewUri(vscode.Uri.file(scriptPath));

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Autocomp Trace Visualizer</title>
</head>
<body class="antialiased">
  <div id="root"><p style="padding:2rem;font-family:monospace;color:#888">Loading visualizer...</p></div>
  <script src="${scriptUri}"></script>
</body>
</html>`;
}

export function deactivate() {
  codeContentStore.clear();
}
