"""
Run the AgentBuilder pipeline: build an agent from local source directories
and/or webpage URLs, inspect generated components, or compare two agents.

Output files (built configs, exported configs) are written to the --output-dir
directory (defaults to <repo>/autocomp/agent_builder/.built/).

Usage:
    # Dry run (test ingestion only, no LLM calls):
    python -m autocomp.agent_builder.run_agent_builder --agent-name my_agent --source-dir path/to/source --dry-run

    # Full build from a directory:
    python -m autocomp.agent_builder.run_agent_builder --agent-name my_agent --source-dir path/to/source

    # Full build from webpage URLs:
    python -m autocomp.agent_builder.run_agent_builder --agent-name my_agent --source-url https://docs.example.com/api

    # Mix directory and URL sources:
    python -m autocomp.agent_builder.run_agent_builder --agent-name my_agent --source-dir path/to/source --source-url https://docs.example.com

    # Inspect an already-built config dir:
    python -m autocomp.agent_builder.run_agent_builder --agent-name my_agent --inspect output/built/my_agent

    # Re-run just one component (rules, optimization_menu, isa, architecture, examples):
    python -m autocomp.agent_builder.run_agent_builder --agent-name my_agent --inspect output/built/my_agent --rerun rules --source-dir path/to/source
"""

import argparse
import re
import yaml
from pathlib import Path

from autocomp.common import REPO_ROOT


# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------

def build_agent(agent_name: str, output_dir: str,
                llm_model: str, light_llm_model: str | None = None,
                agent_scope: str = "",
                source_dir: str | None = None,
                source_files: list[str] | None = None,
                source_urls: list[str] | None = None,
                max_depth: int = 2, max_pages: int = 50,
                context_budget: int = 150_000) -> Path:
    """Run the AgentBuilder pipeline on directory, file, and/or URL sources."""
    from autocomp.agent_builder import AgentBuilder
    builder = AgentBuilder(llm_model=llm_model, light_llm_model=light_llm_model,
                           agent_scope=agent_scope, context_budget=context_budget)
    if source_dir:
        builder.add_source("directory", path=source_dir)
    for f in (source_files or []):
        builder.add_source("file", path=f)
    for url in (source_urls or []):
        builder.add_source("webpage", url=url, max_depth=max_depth, max_pages=max_pages)
    config_dir = builder.build(agent_name=agent_name, output_dir=output_dir)
    return config_dir


# ------------------------------------------------------------------
# Rerun a single component
# ------------------------------------------------------------------

def rerun_components(components: list[str], config_dir: Path,
                     model: str, light_model: str | None = None,
                     agent_scope: str = "",
                     source_dir: str | None = None,
                     source_files: list[str] | None = None,
                     source_urls: list[str] | None = None,
                     max_depth: int = 2, max_pages: int = 50,
                     context_budget: int = 150_000):
    """
    Re-ingest sources once, route once, then re-synthesize each requested
    component and overwrite its file in the existing config dir.
    """
    from autocomp.agent_builder.ingestor import KnowledgeIngestor
    from autocomp.agent_builder.synthesizer import ComponentSynthesizer
    from autocomp.common import LLMClient

    print(f"Re-running synthesis for: {', '.join(components)}")
    print(f"  Config dir: {config_dir}")
    if source_dir:
        print(f"  Source dir: {source_dir}")
    for url in (source_urls or []):
        print(f"  Source URL: {url}")
    print(f"  Model:      {model}")
    if light_model:
        print(f"  Light model: {light_model}")

    if "::" in model:
        provider, model_name = model.split("::", 1)
    else:
        provider, model_name = None, model
    llm = LLMClient(model_name, provider)

    light_llm = None
    if light_model:
        if "::" in light_model:
            lp, lm = light_model.split("::", 1)
        else:
            lp, lm = None, light_model
        light_llm = LLMClient(lm, lp)

    synth = ComponentSynthesizer(llm, light_llm, agent_scope=agent_scope,
                                 context_budget=context_budget)

    if not agent_scope:
        cfg_path = config_dir / "agent_config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            saved_scope = cfg.get("build", {}).get("agent_scope", "") or cfg.get("build", {}).get("description", "")
            if saved_scope:
                synth = ComponentSynthesizer(llm, light_llm, agent_scope=saved_scope,
                                             context_budget=context_budget)
                print(f"  Agent scope (from agent_config.yaml): {saved_scope[:120]}..."
                      if len(saved_scope) > 120 else f"  Agent scope (from agent_config.yaml): {saved_scope}")

    # Ingest and route ONCE for all components
    ingestor = KnowledgeIngestor()
    if source_dir:
        ingestor.add_source("directory", path=source_dir)
    for f in (source_files or []):
        ingestor.add_source("file", path=f)
    for url in (source_urls or []):
        ingestor.add_source("webpage", url=url, max_depth=max_depth, max_pages=max_pages)
    indices = ingestor.ingest()
    buckets = synth._llm_route_content(indices)

    architecture = (config_dir / "architecture.md").read_text() if (config_dir / "architecture.md").exists() else ""
    isa_docs = (config_dir / "isa_docs.md").read_text() if (config_dir / "isa_docs.md").exists() else ""

    for component in components:
        print(f"\n--- Synthesizing: {component} ---")
        bucket_name = {
            "rules": "rules",
            "optimization_menu": "optimization",
            "translate_menu": "optimization",
            "isa": "isa",
            "architecture": "architecture",
            "examples": "examples",
        }[component]

        items = buckets[bucket_name]
        print(f"  Routed {bucket_name}: {len(items)} files")

        if component == "rules":
            result = synth._synthesize_rules(items, architecture=architecture, isa_docs=isa_docs)
            with open(config_dir / "rules.yaml", "w") as f:
                yaml.dump(result, f, default_flow_style=False)
            n = sum(len(v) for v in result.values())
            print(f"  Wrote rules.yaml: {n} rules across {len(result)} categories")

        elif component == "optimization_menu":
            result = synth._synthesize_optimization_menu(items)
            menu_data = {"optimizations": [{"strategy": s} for s in result]}
            with open(config_dir / "optimization_menu.yaml", "w") as f:
                yaml.dump(menu_data, f, default_flow_style=False, width=120)
            print(f"  Wrote optimization_menu.yaml: {len(result)} strategies")

        elif component == "translate_menu":
            result = synth._synthesize_translate_menu(
                items, architecture=architecture, isa_docs=isa_docs,
                code_examples_raw=buckets.get("examples", []),
            )
            menu_data = {"strategies": [{"strategy": s} for s in result]}
            with open(config_dir / "translate_menu.yaml", "w") as f:
                yaml.dump(menu_data, f, default_flow_style=False, width=120)
            print(f"  Wrote translate_menu.yaml: {len(result)} strategies")

        elif component == "isa":
            result = synth._extract_isa_docs(items)
            (config_dir / "isa_docs.md").write_text(result)
            print(f"  Wrote isa_docs.md: {len(result):,} chars")

        elif component == "architecture":
            result = synth._synthesize_architecture(items)
            (config_dir / "architecture.md").write_text(result)
            print(f"  Wrote architecture.md: {len(result):,} chars")

        elif component == "examples":
            result = synth._extract_code_examples(items)
            (config_dir / "code_examples.md").write_text(result)
            print(f"  Wrote code_examples.md: {len(result):,} chars")

        print(f"  Done: {component}")

    print(f"\nAll {len(components)} components re-synthesized.")


# ------------------------------------------------------------------
# Dry-run: validate ingestion without LLM calls
# ------------------------------------------------------------------

def dry_run(source_dir: str | None = None, source_files: list[str] | None = None,
            source_urls: list[str] | None = None,
            max_depth: int = 2, max_pages: int = 50):
    """Ingest sources and report statistics."""
    from autocomp.agent_builder.ingestor import KnowledgeIngestor
    print("=" * 72)
    print("DRY RUN: Testing ingestion (no LLM calls)")
    print("=" * 72)

    ingestor = KnowledgeIngestor()
    if source_dir:
        ingestor.add_source("directory", path=source_dir)
    for f in (source_files or []):
        ingestor.add_source("file", path=f)
    for url in (source_urls or []):
        ingestor.add_source("webpage", url=url, max_depth=max_depth, max_pages=max_pages)
    indices = ingestor.ingest()

    for idx in indices:
        print(f"\nSource:            {idx.source_id}")
        print(f"Metadata length:   {len(idx.structural_metadata):,} chars")
        print(f"Content sections:  {len(idx.content):,}")
        total_chars = sum(len(v) for v in idx.content.values())
        print(f"Total content:     {total_chars:,} chars ({total_chars / 1_000_000:.1f} MB)")

        # Show top extensions / URL patterns by count
        ext_counts: dict[str, int] = {}
        for key in idx.content:
            if key.startswith("http"):
                ext = "(webpage)"
            else:
                ext = Path(key).suffix or "(no ext)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        top_exts = sorted(ext_counts.items(), key=lambda x: -x[1])[:10]
        print("\nTop content types:")
        for ext, count in top_exts:
            print(f"  {ext:12s} {count:4d} files")

        # Show largest content sections
        by_size = sorted(idx.content.items(), key=lambda x: -len(x[1]))[:10]
        print("\nLargest content sections:")
        for key, text in by_size:
            print(f"  {len(text):>8,} chars  {key}")

    print("\nDry run complete. Ingestion is working correctly.")
    print("Run without --dry-run to perform the full build (requires LLM API key).")
    return True


# ------------------------------------------------------------------
# Inspection (replaces the old comparison)
# ------------------------------------------------------------------

def inspect_built_agent(config_dir: Path):
    """Inspect the components generated for a built agent."""
    print("\n" + "=" * 72)
    print("INSPECTION: Built Agent Components")
    print(f"Config dir: {config_dir}")
    print("=" * 72)

    _inspect_optimization_menu(config_dir)
    _inspect_rules(config_dir)
    _inspect_isa_docs(config_dir)
    _inspect_architecture(config_dir)
    _inspect_code_examples(config_dir)

    print("\n" + "=" * 72)
    print("Inspection complete.")
    print("=" * 72)


def _inspect_optimization_menu(config_dir: Path):
    """Report on the optimization menu."""
    path = config_dir / "optimization_menu.yaml"
    if not path.exists():
        print("\n--- Optimization Menu: NOT FOUND ---")
        return

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    items = data.get("optimizations", [])
    strategies = [
        item["strategy"] if isinstance(item, dict) else str(item)
        for item in items
    ]

    print("\n--- Optimization Menu ---")
    print(f"Strategies: {len(strategies)}")
    for i, s in enumerate(strategies, 1):
        print(f"  {i:2d}. {s}")

    lengths = [len(s.split()) for s in strategies]
    if lengths:
        print(f"\nWord counts: min={min(lengths)}, max={max(lengths)}, "
              f"avg={sum(lengths)/len(lengths):.1f}")


def _inspect_rules(config_dir: Path):
    """Report on the rules."""
    path = config_dir / "rules.yaml"
    if not path.exists():
        print("\n--- Rules: NOT FOUND ---")
        return

    with open(path) as f:
        rules = yaml.safe_load(f) or {}

    print("\n--- Rules ---")
    total = 0
    for category in sorted(rules.keys()):
        rule_list = rules[category]
        if not isinstance(rule_list, list):
            continue
        total += len(rule_list)
        print(f"  {category}: {len(rule_list)} rules")
        for r in rule_list:
            text = str(r)
            print(f"    - {text[:120]}{'...' if len(text) > 120 else ''}")
    print(f"  Total: {total} rules")


def _inspect_isa_docs(config_dir: Path):
    """Report on ISA documentation."""
    path = config_dir / "isa_docs.md"
    if not path.exists():
        print("\n--- ISA Documentation: NOT FOUND ---")
        return

    text = path.read_text()
    print("\n--- ISA Documentation ---")
    print(f"Total size: {len(text):,} chars")

    # Extract section headers (## level)
    sections = re.findall(r"^## (.+)$", text, re.MULTILINE)
    print(f"Sections ({len(sections)}):")
    for section in sections:
        # Count entries (### level) within each section
        pattern = rf"## {re.escape(section)}\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            entries = re.findall(r"^### (.+)$", match.group(1), re.MULTILINE)
            section_chars = len(match.group(1))
            print(f"  {section}: {len(entries)} entries, {section_chars:,} chars")
        else:
            print(f"  {section}")

    # Total entries
    all_entries = re.findall(r"^### (.+)$", text, re.MULTILINE)
    print(f"Total entries: {len(all_entries)}")


def _inspect_architecture(config_dir: Path):
    """Report on architecture summary."""
    path = config_dir / "architecture.md"
    if not path.exists():
        print("\n--- Architecture Summary: NOT FOUND ---")
        return

    text = path.read_text()
    print("\n--- Architecture Summary ---")
    print(f"Length: {len(text):,} chars")

    # Show section structure
    headers = re.findall(r"^(#{1,3}) (.+)$", text, re.MULTILINE)
    if headers:
        print("Sections:")
        for level, title in headers:
            indent = "  " * len(level)
            print(f"  {indent}{title}")


def _inspect_code_examples(config_dir: Path):
    """Report on code examples."""
    path = config_dir / "code_examples.md"
    if not path.exists():
        print("\n--- Code Examples: NOT FOUND ---")
        return

    text = path.read_text()
    print("\n--- Code Examples ---")
    print(f"Length: {len(text):,} chars")

    headers = re.findall(r"^## (.+)$", text, re.MULTILINE)
    code_blocks = re.findall(r"```\w*\n", text)
    print(f"Example sections: {len(headers)}")
    print(f"Code blocks: {len(code_blocks)}")
    for h in headers:
        print(f"  - {h}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

_DEFAULT_OUTPUT = str(REPO_ROOT / "autocomp" / "agent_builder" / ".built")

def main():
    parser = argparse.ArgumentParser(
        description="Build an agent and inspect the generated components"
    )
    parser.add_argument("--source-dir", default=None,
                        help="Path to source directory to ingest")
    parser.add_argument("--source-file", action="append", default=None, dest="source_files",
                        help="Path to a single file to ingest (PDF or text; can be repeated)")
    parser.add_argument("--source-url", action="append", default=None, dest="source_urls",
                        help="URL to crawl. Only links under the same path prefix are followed, "
                             "so provide one URL per doc subtree (can be repeated)")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Max link-following depth for webpage sources (default: 2)")
    parser.add_argument("--max-pages", type=int, default=250,
                        help="Max pages to fetch per webpage source (default: 250)")
    parser.add_argument("--agent-name", required=True,
                        help="Name for the built agent")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT,
                        help="Base output directory for built agents (default: ./output/built)")
    parser.add_argument("--agent-scope", default="",
                        help="Defines what the agent covers and what is out of scope. "
                             "Prepended to every LLM prompt. Be specific: what "
                             "code level the agent optimizes (kernels, operators), the "
                             "target hardware, programming interface, and what's excluded.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test ingestion only (no LLM calls needed)")
    parser.add_argument("--model", default="aws::us.anthropic.claude-opus-4-6-v1",
                        help="LLM model for synthesis")
    parser.add_argument("--light-model", default="aws::us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        help="Optional cheaper/faster model for high-token extraction tasks")
    parser.add_argument("--context-budget", type=int, default=150_000,
                        help="Max characters (not tokens) of source content per LLM call "
                             "(default: 150000; ~37-50K tokens depending on content)")
    parser.add_argument("--inspect", metavar="CONFIG_DIR",
                        help="Skip build, just inspect an existing config directory")
    parser.add_argument("--rerun", metavar="COMPONENT", nargs="+",
                        choices=["rules", "optimization_menu", "translate_menu", "isa", "architecture", "examples"],
                        help="Re-run synthesis for one or more components using an existing "
                             "built config dir (requires --inspect and --source-dir)")
    args = parser.parse_args()

    if args.dry_run:
        if not args.source_dir and not args.source_files and not args.source_urls:
            parser.error("--dry-run requires at least one of --source-dir, --source-file, or --source-url")
        dry_run(source_dir=args.source_dir, source_files=args.source_files,
                source_urls=args.source_urls,
                max_depth=args.max_depth, max_pages=args.max_pages)
        return

    if args.rerun:
        if not args.inspect:
            parser.error("--rerun requires --inspect to specify the config directory")
        config_dir = Path(args.inspect)
        if not config_dir.is_absolute():
            config_dir = REPO_ROOT / config_dir
        rerun_components(
            components=args.rerun,
            config_dir=config_dir,
            model=args.model,
            light_model=args.light_model,
            agent_scope=args.agent_scope,
            source_dir=args.source_dir,
            source_files=args.source_files,
            source_urls=args.source_urls,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            context_budget=args.context_budget,
        )
        return

    if args.inspect:
        config_dir = Path(args.inspect)
        if not config_dir.is_absolute():
            config_dir = REPO_ROOT / config_dir
        inspect_built_agent(config_dir)
        return

    if not args.source_dir and not args.source_files and not args.source_urls:
        parser.error("At least one of --source-dir, --source-file, or --source-url is required for building")

    output_dir = str(Path(args.output_dir) / args.agent_name)
    print(f"Building {args.agent_name} agent...")
    if args.source_dir:
        print(f"  Source dir: {args.source_dir}")
    for f in (args.source_files or []):
        print(f"  Source file: {f}")
    for url in (args.source_urls or []):
        print(f"  Source URL: {url}")
    print(f"  Output: {output_dir}")
    print(f"  Model:  {args.model}")
    if args.light_model:
        print(f"  Light:  {args.light_model}")

    config_dir = build_agent(
        agent_name=args.agent_name,
        output_dir=output_dir,
        llm_model=args.model,
        light_llm_model=args.light_model,
        agent_scope=args.agent_scope,
        source_dir=args.source_dir,
        source_files=args.source_files,
        source_urls=args.source_urls,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        context_budget=args.context_budget,
    )
    inspect_built_agent(config_dir)


if __name__ == "__main__":
    main()
