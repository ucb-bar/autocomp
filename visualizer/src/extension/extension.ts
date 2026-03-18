import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import { spawn } from "child_process";

const SCHEME = "autocomp-code";

let codeContentStore = new Map<string, string>();
let lastModel = "";

class CodeContentProvider implements vscode.TextDocumentContentProvider {
  provideTextDocumentContent(uri: vscode.Uri): string {
    return codeContentStore.get(uri.path) ?? "";
  }
}

export function activate(context: vscode.ExtensionContext) {
  const provider = new CodeContentProvider();
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(SCHEME, provider),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("autocomp.openTraces", async () => {
      const uris = await vscode.window.showOpenDialog({
        canSelectFolders: true,
        canSelectFiles: false,
        openLabel: "Select ingested data directory (containing runs.json)",
      });
      if (!uris || uris.length === 0) return;
      const dataDir = uris[0].fsPath;
      const runsJsonPath = path.join(dataDir, "runs.json");
      if (!fs.existsSync(runsJsonPath)) {
        vscode.window.showErrorMessage(`No runs.json found in ${dataDir}`);
        return;
      }
      openPanel(context, dataDir);
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("autocomp.ingestAndOpen", async () => {
      const outputUris = await vscode.window.showOpenDialog({
        canSelectFolders: true,
        canSelectFiles: false,
        openLabel: "Select Autocomp output directory",
      });
      if (!outputUris || outputUris.length === 0) return;
      const outputDir = outputUris[0].fsPath;

      const dataDir = path.join(outputDir, ".visualizer-data");
      fs.mkdirSync(dataDir, { recursive: true });

      await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: "Ingesting traces..." },
        () => runIngest(outputDir, dataDir),
      );

      if (!fs.existsSync(path.join(dataDir, "runs.json"))) {
        vscode.window.showErrorMessage("Ingestion produced no runs.json");
        return;
      }
      openPanel(context, dataDir);
    }),
  );
}

function runIngest(outputDir: string, dataDir: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn("python3", ["-m", "autocomp.visualizer.ingest", outputDir, "--out", dataDir, "--no-summarize"]);
    let stderr = "";
    proc.stderr.on("data", (d) => { stderr += d.toString(); });
    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ingest exited with code ${code}: ${stderr}`));
    });
    proc.on("error", reject);
  });
}

function runSummarize(
  runFile: string,
  model: string,
  progress: vscode.Progress<{ message?: string; increment?: number }>,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn("python3", ["-m", "autocomp.visualizer.ingest", "summarize-run", runFile, "--model", model]);
    let stderr = "";
    let stdoutBuf = "";
    proc.stdout.on("data", (d: Buffer) => {
      stdoutBuf += d.toString();
      const lines = stdoutBuf.split("\n");
      stdoutBuf = lines.pop() ?? "";
      for (const line of lines) {
        try {
          const msg = JSON.parse(line);
          if (msg.total != null && msg.progress != null) {
            if (msg.total === 0) {
              progress.report({ message: "No plans to summarize" });
            } else {
              progress.report({ message: `${msg.progress}/${msg.total} plans` });
            }
          }
        } catch { /* ignore non-JSON lines */ }
      }
    });
    proc.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });
    proc.on("close", (code: number) => {
      if (code === 0) resolve();
      else reject(new Error(`summarize-run failed (code ${code}): ${stderr}`));
    });
    proc.on("error", reject);
  });
}

function openPanel(context: vscode.ExtensionContext, dataDir: string) {
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

  panel.webview.html = getWebviewHtml(context, panel.webview);

  panel.webview.onDidReceiveMessage(
    (msg) => {
      if (msg.type === "ready") {
        const runsJson = path.join(dataDir, "runs.json");
        try {
          const raw = JSON.parse(fs.readFileSync(runsJson, "utf-8"));
          const EXPECTED_SCHEMA = 1;
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
      } else if (msg.type === "selectRun") {
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
        (async () => {
          const model = await vscode.window.showInputBox({
            prompt: "LLM model for plan summarization. Format: provider::model",
            placeHolder: "openai::gpt-4o-mini, anthropic::claude-sonnet-4-20250514, aws::us.anthropic.claude-sonnet-4-20250514-v1:0",
            value: lastModel,
          });
          if (!model) {
            panel.webview.postMessage({ type: "summarizeResult", cancelled: true });
            return;
          }
          lastModel = model;
          const runFile = path.join(dataDir, path.basename(msg.file));
          try {
            await vscode.window.withProgress(
              { location: vscode.ProgressLocation.Notification, title: "Summarizing plans...", cancellable: false },
              (progress) => runSummarize(runFile, model, progress),
            );
            const data = JSON.parse(fs.readFileSync(runFile, "utf-8"));
            panel.webview.postMessage({ type: "runData", data });
          } catch (e: unknown) {
            const errMsg = e instanceof Error ? e.message : String(e);
            vscode.window.showErrorMessage(`Summarization failed: ${errMsg}`);
            panel.webview.postMessage({ type: "summarizeResult", error: errMsg });
          }
        })();
      }
    },
    undefined,
    context.subscriptions,
  );
}

function getWebviewHtml(context: vscode.ExtensionContext, webview: vscode.Webview): string {
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
