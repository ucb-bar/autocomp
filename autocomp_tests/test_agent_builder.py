"""
Test the AgentBuilder by building an agent from a local source directory
and inspecting the generated components.

Output files (built configs, exported configs) are written to ./output/
in the current working directory, NOT inside the package.

Usage:
    # Dry run (test ingestion only, no LLM calls):
    python -m autocomp_tests.test_agent_builder --source-dir path/to/source --dry-run

    # Full build from a directory:
    python -m autocomp_tests.test_agent_builder --source-dir path/to/source --agent-name my_agent

    # Full build from webpage URLs:
    python -m autocomp_tests.test_agent_builder --source-url https://docs.example.com/api --agent-name my_agent

    # Mix directory and URL sources:
    python -m autocomp_tests.test_agent_builder --source-dir path/to/source --source-url https://docs.example.com --agent-name my_agent

    # Full build with a specific model:
    python -m autocomp_tests.test_agent_builder --source-dir path/to/source --model aws::us.anthropic.claude-opus-4-6-v1

    # Inspect an already-built config dir:
    python -m autocomp_tests.test_agent_builder --inspect output/built/trn

    # Compare two built agents:
    python -m autocomp_tests.test_agent_builder --inspect output/built/new_agent --compare-to output/built/ref_agent

    # Build and compare against the hand-crafted trn agent:
    python -m autocomp_tests.test_agent_builder --source-dir path/to/source --compare-to-agent trn

    # Inspect a built agent and compare against the hand-crafted trn agent:
    python -m autocomp_tests.test_agent_builder --inspect output/built/trn --compare-to-agent trn

    # Re-run just one component (rules, optimization_menu, isa, architecture, examples):
    python -m autocomp_tests.test_agent_builder --inspect output/built/trn --rerun rules
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
                description: str = "",
                source_dir: str | None = None,
                source_urls: list[str] | None = None,
                max_depth: int = 2, max_pages: int = 50) -> Path:
    """Run the AgentBuilder pipeline on directory and/or URL sources."""
    from autocomp.agent_builder import AgentBuilder
    builder = AgentBuilder(llm_model=llm_model, light_llm_model=light_llm_model,
                           description=description)
    if source_dir:
        builder.add_source("directory", path=source_dir)
    for url in (source_urls or []):
        builder.add_source("webpage", url=url, max_depth=max_depth, max_pages=max_pages)
    config_dir = builder.build(agent_name=agent_name, output_dir=output_dir)
    return config_dir


# ------------------------------------------------------------------
# Rerun a single component
# ------------------------------------------------------------------

def rerun_component(component: str, config_dir: Path,
                    model: str, light_model: str | None = None,
                    description: str = "",
                    source_dir: str | None = None,
                    source_urls: list[str] | None = None,
                    max_depth: int = 2, max_pages: int = 50):
    """
    Re-ingest the source, re-route, and re-synthesize a single component,
    then overwrite just that file in the existing config dir.
    """
    from autocomp.agent_builder.ingestor import KnowledgeIngestor
    from autocomp.agent_builder.synthesizer import ComponentSynthesizer
    from autocomp.common import LLMClient

    print(f"Re-running '{component}' synthesis")
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

    synth = ComponentSynthesizer(llm, light_llm, description=description)

    # Re-ingest and route
    ingestor = KnowledgeIngestor()
    if source_dir:
        ingestor.add_source("directory", path=source_dir)
    for url in (source_urls or []):
        ingestor.add_source("webpage", url=url, max_depth=max_depth, max_pages=max_pages)
    indices = ingestor.ingest()
    buckets = synth._llm_route_content(indices)

    bucket_name = {
        "rules": "rules",
        "optimization_menu": "optimization",
        "isa": "isa",
        "architecture": "architecture",
        "examples": "examples",
    }[component]

    # Routing already filters via LLM, just grab the relevant bucket
    items = buckets[bucket_name]
    print(f"  Routed {bucket_name}: {len(items)} files")

    # Read existing built files for context (rules needs architecture + ISA)
    architecture = (config_dir / "architecture.md").read_text() if (config_dir / "architecture.md").exists() else ""
    isa_docs = (config_dir / "isa_docs.md").read_text() if (config_dir / "isa_docs.md").exists() else ""

    # Synthesize the single component
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
            yaml.dump(menu_data, f, default_flow_style=False)
        print(f"  Wrote optimization_menu.yaml: {len(result)} strategies")

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

    print("Done.")


# ------------------------------------------------------------------
# Dry-run: validate ingestion without LLM calls
# ------------------------------------------------------------------

def dry_run(source_dir: str | None = None, source_urls: list[str] | None = None,
            max_depth: int = 2, max_pages: int = 50):
    """Ingest sources and report statistics."""
    from autocomp.agent_builder.ingestor import KnowledgeIngestor
    print("=" * 72)
    print("DRY RUN: Testing ingestion (no LLM calls)")
    print("=" * 72)

    ingestor = KnowledgeIngestor()
    if source_dir:
        ingestor.add_source("directory", path=source_dir)
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
# Export hand-crafted agent to config dir (for comparison)
# ------------------------------------------------------------------

def export_agent_config(agent_module: str, output_dir: Path) -> Path:
    """
    Instantiate a hand-crafted agent and export its components to a config
    directory compatible with compare_agents().

    Args:
        agent_module: Dotted path like 'trn' that maps to autocomp.agents.<name>.
        output_dir: Where to write the exported config files.
    """
    import importlib

    output_dir.mkdir(parents=True, exist_ok=True)

    if agent_module == "trn":
        mod = importlib.import_module("autocomp.agents.trn.trn_agent")
        from autocomp.agents.trn.nki_isa_generator import NkiIsaGenerator
        from autocomp.hw_config.trn_config import TrnHardwareConfig
        from autocomp.search.prob import Prob

        agent = mod.TrnLLMAgent.__new__(mod.TrnLLMAgent)
        agent.hw_config = TrnHardwareConfig("trn1.2xlarge")
        agent.nki_isa_generator = NkiIsaGenerator()

        prob = Prob("trn-tutorial", 1)

        # Optimization menu
        strategies = agent.get_opt_menu_options(prob)
        menu_data = {"optimizations": [{"strategy": s} for s in strategies]}
        with open(output_dir / "optimization_menu.yaml", "w") as f:
            yaml.dump(menu_data, f, default_flow_style=False)

        # Rules — extract the non-problem-specific, non-hw-config rules
        # by calling _get_prompt_rules and parsing the numbered list
        agent.eval_backend = type("Stub", (), {"get_backend_specific_rules": lambda self: []})()
        agent.hw_config = type("Stub", (), {"get_hw_config_specific_rules": lambda self: []})()

        import random
        random.seed(0)
        planning_rules_text = mod.TrnLLMAgent._get_prompt_rules(agent, planning=True, coding=False, prob=prob)
        random.seed(0)
        coding_rules_text = mod.TrnLLMAgent._get_prompt_rules(agent, planning=False, coding=True, prob=prob)

        def _parse_rules(text: str) -> list[str]:
            lines = text.strip().split("\n")
            return [re.sub(r"^\d+\.\s*", "", line).strip() for line in lines if line.strip()]

        planning_rules = _parse_rules(planning_rules_text)
        coding_rules = _parse_rules(coding_rules_text)
        general_rules = []
        # Rules common to both planning and coding are "general"
        planning_set = set(planning_rules)
        coding_set = set(coding_rules)
        general = planning_set & coding_set
        general_rules = [r for r in planning_rules if r in general]
        planning_only = [r for r in planning_rules if r not in general]
        coding_only = [r for r in coding_rules if r not in general]

        rules_data = {"general": general_rules, "planning": planning_only, "coding": coding_only}
        with open(output_dir / "rules.yaml", "w") as f:
            yaml.dump(rules_data, f, default_flow_style=False)

        # ISA docs
        isa_text = agent.nki_isa_generator.generate_isa(prob)
        (output_dir / "isa_docs.md").write_text(isa_text)

        # Architecture — hand-crafted agent has a one-liner; extract what we can
        arch_text = ("# Trainium/Inferentia (NKI)\n\n"
                     "The NKI (Neuron Kernel Interface) is used for writing "
                     "high-performance kernels on AWS Trainium and Inferentia chips.\n")
        (output_dir / "architecture.md").write_text(arch_text)

        # No separate code examples in the hand-crafted agent
        (output_dir / "code_examples.md").write_text("")

        print(f"Exported hand-crafted '{agent_module}' agent to {output_dir}")
        return output_dir
    else:
        raise ValueError(
            f"Unknown agent module '{agent_module}'. "
            f"Add export logic for this agent in export_agent_config()."
        )


# ------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------

def _load_agent_config(config_dir: Path) -> dict:
    """Load all components from a config directory into a dict."""
    config: dict = {"dir": config_dir}

    menu_path = config_dir / "optimization_menu.yaml"
    if menu_path.exists():
        with open(menu_path) as f:
            data = yaml.safe_load(f) or {}
        config["strategies"] = [
            item["strategy"] if isinstance(item, dict) else str(item)
            for item in data.get("optimizations", [])
        ]
    else:
        config["strategies"] = []

    rules_path = config_dir / "rules.yaml"
    if rules_path.exists():
        with open(rules_path) as f:
            config["rules"] = yaml.safe_load(f) or {}
    else:
        config["rules"] = {}

    for name in ("isa_docs.md", "architecture.md", "code_examples.md"):
        path = config_dir / name
        config[name] = path.read_text() if path.exists() else ""

    return config


def compare_agents(config_dir: Path, ref_dir: Path, model: str | None = None):
    """
    Compare a built agent against a reference agent config directory.

    When a --model is provided, uses an LLM for semantic comparison.
    Otherwise falls back to structural/keyword comparison.
    """
    print("\n" + "=" * 72)
    print("COMPARISON")
    print(f"  Built: {config_dir}")
    print(f"  Ref:   {ref_dir}")
    print("=" * 72)

    built = _load_agent_config(config_dir)
    ref = _load_agent_config(ref_dir)

    _compare_optimization_menu(built, ref)
    _compare_rules(built, ref)
    _compare_isa_docs(built, ref)
    _compare_architecture(built, ref)
    _compare_code_examples(built, ref)

    if model:
        _llm_semantic_comparison(built, ref, model)

    print("\n" + "=" * 72)
    print("Comparison complete.")
    print("=" * 72)


def _compare_optimization_menu(built: dict, ref: dict):
    """Compare optimization strategies between built and reference agents."""
    b_strats = built["strategies"]
    r_strats = ref["strategies"]

    print("\n--- Optimization Menu ---")
    print(f"Built: {len(b_strats)} strategies | Ref: {len(r_strats)} strategies")

    b_lower = {s.lower().strip() for s in b_strats}
    r_lower = {s.lower().strip() for s in r_strats}
    exact_shared = b_lower & r_lower
    if exact_shared:
        print(f"\nExact matches ({len(exact_shared)}):")
        for s in sorted(exact_shared):
            print(f"  = {s}")

    b_kw = _extract_keywords(b_strats)
    r_kw = _extract_keywords(r_strats)
    shared_kw = b_kw & r_kw
    built_only_kw = b_kw - r_kw
    ref_only_kw = r_kw - b_kw

    print(f"\nTheme overlap: {len(shared_kw)} shared, "
          f"{len(built_only_kw)} built-only, {len(ref_only_kw)} ref-only")
    if shared_kw:
        print(f"  Shared:     {', '.join(sorted(shared_kw))}")
    if built_only_kw:
        print(f"  Built-only: {', '.join(sorted(built_only_kw))}")
    if ref_only_kw:
        print(f"  Ref-only:   {', '.join(sorted(ref_only_kw))}")

    coverage = len(shared_kw) / len(r_kw) * 100 if r_kw else 100
    print(f"  Theme coverage of ref: {coverage:.0f}%")


def _compare_rules(built: dict, ref: dict):
    """Compare rules between built and reference agents."""
    b_rules = built["rules"]
    r_rules = ref["rules"]
    all_cats = sorted(set(list(b_rules.keys()) + list(r_rules.keys())))

    print("\n--- Rules ---")
    for cat in all_cats:
        b_list = b_rules.get(cat, [])
        r_list = r_rules.get(cat, [])
        if not isinstance(b_list, list):
            b_list = []
        if not isinstance(r_list, list):
            r_list = []
        print(f"  {cat}: built={len(b_list)}, ref={len(r_list)}")


def _compare_isa_docs(built: dict, ref: dict):
    """Compare ISA documentation coverage."""
    b_text = built["isa_docs.md"]
    r_text = ref["isa_docs.md"]

    b_entries = set(re.findall(r"^### (.+)$", b_text, re.MULTILINE))
    r_entries = set(re.findall(r"^### (.+)$", r_text, re.MULTILINE))

    shared = b_entries & r_entries
    built_only = b_entries - r_entries
    ref_only = r_entries - b_entries

    print("\n--- ISA Documentation ---")
    print(f"Built: {len(b_text):,} chars, {len(b_entries)} entries | "
          f"Ref: {len(r_text):,} chars, {len(r_entries)} entries")
    print(f"Entry overlap: {len(shared)} shared, "
          f"{len(built_only)} built-only, {len(ref_only)} ref-only")

    if ref_only and len(ref_only) <= 20:
        print(f"  Missing from built: {', '.join(sorted(ref_only))}")
    elif ref_only:
        print(f"  Missing from built: {', '.join(sorted(list(ref_only))[:20])}... "
              f"(+{len(ref_only) - 20} more)")

    coverage = len(shared) / len(r_entries) * 100 if r_entries else 100
    print(f"  Entry coverage of ref: {coverage:.0f}%")


def _compare_architecture(built: dict, ref: dict):
    """Compare architecture summaries."""
    b_text = built["architecture.md"]
    r_text = ref["architecture.md"]

    print("\n--- Architecture ---")
    print(f"Built: {len(b_text):,} chars | Ref: {len(r_text):,} chars")

    b_headers = set(re.findall(r"^#{1,3} (.+)$", b_text, re.MULTILINE))
    r_headers = set(re.findall(r"^#{1,3} (.+)$", r_text, re.MULTILINE))
    shared = b_headers & r_headers
    if shared:
        print(f"  Shared section headings: {', '.join(sorted(shared))}")
    built_only = b_headers - r_headers
    ref_only = r_headers - b_headers
    if built_only:
        print(f"  Built-only headings: {', '.join(sorted(built_only))}")
    if ref_only:
        print(f"  Ref-only headings: {', '.join(sorted(ref_only))}")


def _compare_code_examples(built: dict, ref: dict):
    """Compare code examples."""
    b_text = built["code_examples.md"]
    r_text = ref["code_examples.md"]

    b_sections = set(re.findall(r"^## (.+)$", b_text, re.MULTILINE))
    r_sections = set(re.findall(r"^## (.+)$", r_text, re.MULTILINE))

    print("\n--- Code Examples ---")
    print(f"Built: {len(b_text):,} chars, {len(b_sections)} examples | "
          f"Ref: {len(r_text):,} chars, {len(r_sections)} examples")

    shared = b_sections & r_sections
    if shared:
        print(f"  Shared ({len(shared)}): {', '.join(sorted(list(shared))[:15])}")
    built_only = b_sections - r_sections
    if built_only:
        print(f"  Built-only ({len(built_only)}): {', '.join(sorted(list(built_only))[:15])}")
    ref_only = r_sections - b_sections
    if ref_only:
        print(f"  Ref-only ({len(ref_only)}): {', '.join(sorted(list(ref_only))[:15])}")


def _llm_semantic_comparison(built: dict, ref: dict, model: str):
    """Use an LLM to produce a semantic diff summary across all components."""
    from autocomp.common import LLMClient

    print("\n--- LLM Semantic Analysis ---")
    try:
        if "::" in model:
            provider, model_name = model.split("::", 1)
        else:
            provider, model_name = None, model
        llm = LLMClient(model_name, provider)
    except Exception as e:
        print(f"  (Skipped: could not initialize LLM: {e})")
        return

    def _truncate(text: str, limit: int = 6000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... [{len(text) - limit:,} chars truncated]"

    b_strats = "\n".join(f"- {s}" for s in built["strategies"])
    r_strats = "\n".join(f"- {s}" for s in ref["strategies"])

    b_rules_text = yaml.dump(built["rules"], default_flow_style=False)
    r_rules_text = yaml.dump(ref["rules"], default_flow_style=False)

    b_isa_entries = re.findall(r"^### (.+)$", built["isa_docs.md"], re.MULTILINE)
    r_isa_entries = re.findall(r"^### (.+)$", ref["isa_docs.md"], re.MULTILINE)

    prompt = f"""Compare these two agent configurations and identify the most important semantic differences. Focus on:
1. Coverage gaps: what concepts/capabilities does one have that the other lacks?
2. Quality differences: where is one agent's guidance more specific or actionable?
3. Potential issues: any rules, strategies, or docs that seem incorrect or counterproductive?

=== BUILT AGENT ===

Optimization strategies:
{b_strats}

Rules:
{_truncate(b_rules_text, 4000)}

Architecture summary ({len(built['architecture.md']):,} chars):
{_truncate(built['architecture.md'], 3000)}

ISA entries ({len(b_isa_entries)}): {', '.join(b_isa_entries[:50])}{'...' if len(b_isa_entries) > 50 else ''}

=== REFERENCE AGENT ===

Optimization strategies:
{r_strats}

Rules:
{_truncate(r_rules_text, 4000)}

Architecture summary ({len(ref['architecture.md']):,} chars):
{_truncate(ref['architecture.md'], 3000)}

ISA entries ({len(r_isa_entries)}): {', '.join(r_isa_entries[:50])}{'...' if len(r_isa_entries) > 50 else ''}

Provide a concise analysis (bullet points). End with an overall assessment of which agent would likely produce better optimization suggestions and why."""

    try:
        responses = llm.chat(prompt=prompt, num_candidates=1, temperature=0)
        if responses:
            print(responses[0])
        else:
            print("  (No response from LLM)")
    except Exception as e:
        print(f"  (LLM comparison failed: {e})")


# ------------------------------------------------------------------
# Keyword extraction for structural comparison
# ------------------------------------------------------------------

_OPT_THEMES = {
    "tiling", "tile", "fusion", "fuse", "fused", "precision", "bfloat16",
    "fp16", "fp8", "double buffering", "pipelining", "pipeline",
    "matmul", "transpose", "loop", "reorder", "hoist",
    "reduction", "scan", "softmax", "mask", "load", "store",
    "data movement", "overlap", "compute", "contiguous", "locality",
    "reuse", "accumulation", "pad", "align", "spill", "prefetch",
    "unroll", "vectorize", "parallel", "batch", "dma", "cache",
    "memory", "bandwidth", "throughput", "latency",
}


def _extract_keywords(strategies: list[str]) -> set[str]:
    keywords = set()
    for s in strategies:
        s_lower = s.lower()
        for term in _OPT_THEMES:
            if term in s_lower:
                keywords.add(term)
    return keywords


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
    parser.add_argument("--source-url", action="append", default=None, dest="source_urls",
                        help="URL to ingest (can be repeated)")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Max link-following depth for webpage sources (default: 2)")
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Max pages to fetch per webpage source (default: 50)")
    parser.add_argument("--agent-name", default="trn",
                        help="Name for the built agent")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT,
                        help="Base output directory for built agents (default: ./output/built)")
    parser.add_argument("--description", default="",
                        help="Context description for LLM prompts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test ingestion only (no LLM calls needed)")
    parser.add_argument("--model", default="aws::us.anthropic.claude-opus-4-6-v1",
                        help="LLM model for synthesis")
    parser.add_argument("--light-model", default="aws::us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        help="Optional cheaper/faster model for high-token extraction tasks")
    parser.add_argument("--inspect", metavar="CONFIG_DIR",
                        help="Skip build, just inspect an existing config directory")
    parser.add_argument("--compare-to", metavar="REF_DIR",
                        help="Compare against a reference agent config directory "
                             "(uses --model for semantic LLM comparison if available)")
    parser.add_argument("--compare-to-agent", metavar="AGENT_NAME",
                        help="Export a hand-crafted agent (e.g. 'trn') to a temp config "
                             "dir and compare against it")
    parser.add_argument("--rerun", metavar="COMPONENT",
                        choices=["rules", "optimization_menu", "isa", "architecture", "examples"],
                        help="Re-run synthesis for a single component using an existing "
                             "built config dir (requires --inspect and --source-dir)")
    args = parser.parse_args()

    # Resolve --compare-to-agent into a --compare-to dir
    if args.compare_to_agent:
        export_dir = Path.cwd() / "output" / "exported" / args.compare_to_agent
        export_agent_config(args.compare_to_agent, export_dir)
        args.compare_to = str(export_dir)

    if args.dry_run:
        if not args.source_dir and not args.source_urls:
            parser.error("--dry-run requires at least one of --source-dir or --source-url")
        dry_run(source_dir=args.source_dir, source_urls=args.source_urls,
                max_depth=args.max_depth, max_pages=args.max_pages)
        return

    if args.rerun:
        if not args.inspect:
            parser.error("--rerun requires --inspect to specify the config directory")
        config_dir = Path(args.inspect)
        if not config_dir.is_absolute():
            config_dir = REPO_ROOT / config_dir
        rerun_component(
            component=args.rerun,
            config_dir=config_dir,
            model=args.model,
            light_model=args.light_model,
            description=args.description,
            source_dir=args.source_dir,
            source_urls=args.source_urls,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
        )
        return

    if args.inspect:
        config_dir = Path(args.inspect)
        if not config_dir.is_absolute():
            config_dir = REPO_ROOT / config_dir
        inspect_built_agent(config_dir)
        if args.compare_to:
            ref_dir = Path(args.compare_to)
            if not ref_dir.is_absolute():
                ref_dir = REPO_ROOT / ref_dir
            compare_agents(config_dir, ref_dir, model=args.model)
        return

    if not args.source_dir and not args.source_urls:
        parser.error("At least one of --source-dir or --source-url is required for building")

    output_dir = str(Path(args.output_dir) / args.agent_name)
    print(f"Building {args.agent_name} agent...")
    if args.source_dir:
        print(f"  Source dir: {args.source_dir}")
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
        description=args.description,
        source_dir=args.source_dir,
        source_urls=args.source_urls,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
    )
    inspect_built_agent(config_dir)
    if args.compare_to:
        ref_dir = Path(args.compare_to)
        if not ref_dir.is_absolute():
            ref_dir = REPO_ROOT / ref_dir
        compare_agents(config_dir, ref_dir, model=args.model)


if __name__ == "__main__":
    main()
