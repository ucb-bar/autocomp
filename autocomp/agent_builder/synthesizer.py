"""
Component synthesis for the Agent Builder.

General-purpose content routing + LLM filtering + per-file extraction to
distill ingested knowledge into agent components: architecture summary,
ISA docs, optimization menu, rules, and code examples.

Stage 1   (route):   Broad path + content heuristics cast a wide net.
Stage 1.5 (filter):  LLM prunes false positives from each bucket.
Stage 2   (extract): Per-file LLM extraction into structured ISAEntry objects.
Stage 3   (merge):   LLM-based functional categorization + reassembly.
"""

import json
import re
import time
from dataclasses import dataclass

from autocomp.common import logger, LLMClient
from autocomp.agent_builder.ingestor import SourceIndex


@dataclass
class SynthesizedComponents:
    """The components produced by the synthesizer."""
    architecture_summary: str
    isa_docs: str
    optimization_menu: list[str]
    rules: dict[str, list[str]]  # keys: "general", "planning", "coding"
    code_examples: str = ""


@dataclass
class ISAEntry:
    """A single extracted API/instruction entry."""
    name: str
    description: str
    source_key: str
    markdown: str


# ---------------------------------------------------------------------------
# General-purpose routing tables (no domain-specific keywords)
# ---------------------------------------------------------------------------

_ISA_PATH_PATTERNS = [
    "api/",
    "reference/",
    "ref/",
    "isa/",
]

_ISA_EXCLUDE_PATTERNS = [
    "example",
    "tutorial",
    "sample",
    "library/",
    "demo",
    "test",
]

_ARCH_PATTERNS = [
    "arch",
    "hardware",
    "overview",
    "design",
    "about/",
    "memory",
]

_OPT_PATTERNS = [
    "perf",
    "optim",
    "tuning",
    "best-practice",
    "performance",
    "how-to",
    "library/",
]

_RULES_PATTERNS = [
    "guide",
    "constraint",
    "pitfall",
    "migration",
]

_EXAMPLES_PATTERNS = [
    "example",
    "sample",
    "tutorial",
    "demo",
]

_SKIP_PATTERNS = [
    "release-notes",
    "archive/",
    ".github/",
    "_content-types/",
    "_ext/",
    "_templates/",
    "_utilities/",
    "_static/",
    "containers/",
    "setup/",
    "devflows/",
    "benchmarks/",
]

# Min top-level defs for a .py file to be considered an API stub
_STUB_MIN_DEFS = 5

# RST directives that indicate API reference documentation
_AUTODOC_DIRECTIVES = [
    ".. autofunction::",
    ".. autoclass::",
    ".. automodule::",
    ".. currentmodule::",
    ".. module::",
    ".. function::",
    ".. class::",
    ".. method::",
    ".. attribute::",
]


def _is_python_stub(text: str) -> bool:
    """Check if a Python file looks like an API stub (many defs with docstrings, no bodies)."""
    defs = re.findall(r"^(?:def |class )\w+", text, re.MULTILINE)
    if len(defs) < _STUB_MIN_DEFS:
        return False
    docstrings = len(re.findall(r'^\s+(?:r)?"""', text, re.MULTILINE))
    ellipsis_bodies = len(re.findall(r"^\s+\.\.\.\s*$", text, re.MULTILINE))
    return docstrings >= len(defs) * 0.5 and ellipsis_bodies >= len(defs) * 0.3


def _is_autodoc_rst(text: str) -> bool:
    """Check if an RST file contains autodoc/API-reference directives."""
    return any(directive in text for directive in _AUTODOC_DIRECTIVES)


def _route_content(indices: list[SourceIndex]) -> dict[str, list[tuple[str, str]]]:
    """
    Route ingested content into component buckets using general heuristics.

    Uses a combination of path patterns and content-based signals to cast a
    wide net. An LLM filtering step later prunes false positives.

    Returns {component: [(content_key, content_text), ...]}.
    A file can appear in multiple buckets.
    """
    buckets: dict[str, list[tuple[str, str]]] = {
        "isa": [],
        "architecture": [],
        "optimization": [],
        "rules": [],
        "examples": [],
    }

    for idx in indices:
        for key, text in idx.content.items():
            if not text or not text.strip():
                continue
            key_lower = key.lower()

            if any(skip in key_lower for skip in _SKIP_PATTERNS):
                continue

            # --- ISA / API reference (path + content signals) ---
            is_excluded = any(ex in key_lower for ex in _ISA_EXCLUDE_PATTERNS)
            if not is_excluded:
                path_match = any(p in key_lower for p in _ISA_PATH_PATTERNS)
                content_match = False
                if key.endswith(".py"):
                    content_match = _is_python_stub(text)
                elif key.endswith(".rst"):
                    content_match = _is_autodoc_rst(text)
                if path_match or content_match:
                    buckets["isa"].append((key, text))

            # --- Architecture ---
            for pattern in _ARCH_PATTERNS:
                if pattern in key_lower:
                    buckets["architecture"].append((key, text))
                    break

            # --- Optimization ---
            for pattern in _OPT_PATTERNS:
                if pattern in key_lower:
                    buckets["optimization"].append((key, text))
                    break

            # --- Rules (prose guides, constraints, pitfalls) ---
            for pattern in _RULES_PATTERNS:
                if pattern in key_lower:
                    buckets["rules"].append((key, text))
                    break

            # --- Examples (code examples, tutorials, demos) ---
            if any(p in key_lower for p in _EXAMPLES_PATTERNS):
                buckets["examples"].append((key, text))

    return buckets


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

class ComponentSynthesizer:
    """
    Synthesizer that distills ingested knowledge into agent components.

    Uses heuristic path-based routing (not LLM) for content selection,
    then per-file LLM extraction for each component.
    """

    def __init__(self, llm_client: LLMClient, light_llm_client: LLMClient | None = None,
                 description: str = ""):
        self.llm = llm_client
        self.light_llm = light_llm_client or llm_client
        self._context_prefix = f"Context: {description}\n\n" if description else ""

    def _chat(self, prompt: str, **kwargs) -> list[str]:
        return self.llm.chat(prompt=self._context_prefix + prompt, **kwargs)

    def _chat_light(self, prompt: str, **kwargs) -> list[str]:
        return self.light_llm.chat(prompt=self._context_prefix + prompt, **kwargs)

    def _chat_light_async(self, prompts: list[str], **kwargs) -> list[list[str]]:
        return self.light_llm.chat_async(
            [self._context_prefix + p for p in prompts], **kwargs,
        )

    # ------------------------------------------------------------------
    # LLM bucket filter (Stage 1.5 -- precision pass)
    # ------------------------------------------------------------------

    _FILTER_DESCRIPTIONS = {
        "isa": (
            "API / instruction-set reference documentation "
            "(function signatures, parameter descriptions, instruction semantics)"
        ),
        "architecture": (
            "hardware architecture documentation "
            "(memory hierarchy, compute units, system design, chip overviews)"
        ),
        "optimization": (
            "performance optimization guidance "
            "(tuning strategies, optimization techniques, profiling advice)"
        ),
        "rules": (
            "kernel/code optimization constraints and programming model rules "
            "(correctness constraints, tile size rules, memory layout requirements, "
            "API usage pitfalls). Exclude deployment, serving, training frameworks, "
            "and environment configuration"
        ),
        "examples": (
            "code examples and tutorials that demonstrate kernel/API usage "
            "(runnable examples, sample kernels, tutorial code with actual implementations). "
            "Exclude setup scripts, configuration, build files, and test harnesses"
        ),
    }

    _FILTER_PREVIEW_CHARS = 2000

    def _llm_filter_bucket(
        self, bucket_name: str, items: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Use the LLM to prune false positives from a routed bucket.

        Makes one lightweight LLM call per file (parallelised via chat_async),
        each including a content preview so the model can make an informed
        yes/no decision.
        """
        if len(items) <= 3:
            return items

        description = self._FILTER_DESCRIPTIONS.get(bucket_name, bucket_name)

        prompts: list[str] = []
        for key, text in items:
            preview = text[:self._FILTER_PREVIEW_CHARS]
            prompts.append(
                f"Does this file contain {description}?\n"
                f"Filename: {key}\n\n"
                f"=== CONTENT PREVIEW ===\n{preview}\n=== END ===\n\n"
                "Answer YES or NO. One word only."
            )

        all_responses = self._chat_light_async(
            prompts, num_candidates=1, temperature=0,
        )

        kept: list[tuple[str, str]] = []
        rejected: list[str] = []
        for item, responses in zip(items, all_responses):
            answer = (responses[0].strip() if responses else "YES").upper()
            if answer.startswith("YES"):
                kept.append(item)
            else:
                rejected.append(item[0])

        if not kept:
            logger.warning(
                "  LLM filter %s: all %d files rejected, keeping all",
                bucket_name, len(items),
            )
            return items

        logger.info("  LLM filter %s: kept %d/%d", bucket_name, len(kept), len(items))
        logger.info("  LLM filter %s kept: %s",
                     bucket_name, [k for k, _ in kept])
        if rejected:
            logger.info("  LLM filter %s rejected: %s", bucket_name, rejected)
        return kept

    def synthesize(self, indices: list[SourceIndex]) -> SynthesizedComponents:
        """Run the full synthesis pipeline: route -> filter -> extract -> merge."""
        t0 = time.time()

        # Stage 1: Route content into buckets (wide net)
        logger.info("ComponentSynthesizer: Stage 1 -- routing content by path + content signals")
        buckets = _route_content(indices)
        for bname, items in buckets.items():
            total_chars = sum(len(t) for _, t in items)
            logger.info("  %s: %d files, %d chars", bname, len(items), total_chars)
            logger.debug("  %s files: %s", bname, [k for k, _ in items])
        t_route = time.time()
        logger.info("  Routing took %.1fs", t_route - t0)

        # Stage 1.5: LLM filter to prune false positives
        logger.info("ComponentSynthesizer: Stage 1.5 -- LLM filtering")
        for bname in buckets:
            buckets[bname] = self._llm_filter_bucket(bname, buckets[bname])
        t_filter = time.time()
        logger.info("  Filtering took %.1fs", t_filter - t_route)

        # Post-filter summary
        logger.info("ComponentSynthesizer: Post-filter summary:")
        for bname, items in buckets.items():
            total_chars = sum(len(t) for _, t in items)
            logger.info("  %s: %d files, %d chars", bname, len(items), total_chars)

        # Stage 2 + 3: Extract and merge each component
        logger.info("ComponentSynthesizer: Stage 2 -- per-file extraction")

        t_start = time.time()
        architecture = self._synthesize_architecture(buckets["architecture"])
        logger.info("  Architecture: %d chars (%.1fs)", len(architecture), time.time() - t_start)

        t_start = time.time()
        isa_docs = self._extract_isa_docs(buckets["isa"])
        logger.info("  ISA docs: %d chars (%.1fs)", len(isa_docs), time.time() - t_start)

        t_start = time.time()
        opt_menu = self._synthesize_optimization_menu(buckets["optimization"])
        logger.info("  Optimization menu: %d strategies (%.1fs)", len(opt_menu), time.time() - t_start)

        t_start = time.time()
        rules = self._synthesize_rules(buckets["rules"], architecture=architecture, isa_docs=isa_docs)
        n_rules = sum(len(v) for v in rules.values())
        logger.info("  Rules: %d rules across %d categories (%.1fs)", n_rules, len(rules), time.time() - t_start)

        t_start = time.time()
        code_examples = self._extract_code_examples(buckets["examples"])
        logger.info("  Code examples: %d chars (%.1fs)", len(code_examples), time.time() - t_start)

        logger.info("ComponentSynthesizer: total synthesis time %.1fs", time.time() - t0)

        return SynthesizedComponents(
            architecture_summary=architecture,
            isa_docs=isa_docs,
            optimization_menu=opt_menu,
            rules=rules,
            code_examples=code_examples,
        )

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def _synthesize_architecture(self, items: list[tuple[str, str]]) -> str:
        """Summarize hardware architecture from routed content."""
        content = _items_to_text(items, max_chars=100_000)
        if not content.strip():
            return "No architecture documentation found. Please add manually."

        prompt = f"""Below is documentation about the hardware target.

Write a concise but thorough hardware architecture summary covering:
- What the hardware is and its programming model
- Memory hierarchy (on-chip memories, caches, off-chip memory, sizes, bandwidths)
- Compute units (types, capabilities, throughput)
- Key constraints and characteristics that affect code optimization

This summary will be included at the top of every prompt to the agent. It should give the agent the context it needs to make good optimization decisions.

Write in clear technical prose. Do not include instructions to the reader.

=== DOCUMENTATION ===
{content}
=== END DOCUMENTATION ===

Hardware Architecture Summary:"""

        responses = self._chat(prompt=prompt, num_candidates=1, temperature=0)
        return responses[0].strip() if responses else "Architecture summary generation failed."

    # ------------------------------------------------------------------
    # ISA docs
    # ------------------------------------------------------------------

    def _extract_isa_docs(self, items: list[tuple[str, str]]) -> str:
        """Extract ISA/API documentation via LLM-identified boundaries + verbatim copy + categorization."""
        if not items:
            return "No ISA documentation found. Please add manually."

        candidates = [
            (key, text) for key, text in items
            if len(text.strip()) >= 50 and len(text) >= 200
        ]
        candidates.sort(key=lambda x: len(x[1]), reverse=True)

        if not candidates:
            return "No ISA documentation found. Please add manually."

        # Ask the LLM to identify entry boundaries (name, description, line range)
        # then copy source content verbatim -- no LLM rewriting.
        prompts: list[str] = []
        for key, text in candidates:
            prompts.append(self._build_isa_boundary_prompt(key, text))
            logger.info("  ISA boundary detection queued: %s (%d chars)", key, len(text))

        logger.info("  ISA boundary detection: sending %d files to LLM in parallel", len(prompts))
        all_responses = self._chat_light_async(
            prompts, num_candidates=1, temperature=0,
        )

        all_entries: list[ISAEntry] = []
        for (key, text), responses in zip(candidates, all_responses):
            raw = (responses[0].strip() if responses else "")
            entries = self._parse_boundary_response(raw, text, key)
            logger.info("  ISA extract: %s -> %d entries", key, len(entries))
            all_entries.extend(entries)

        if not all_entries:
            return "No ISA documentation found. Please add manually."

        logger.info("  ISA extraction complete: %d entries from %d files",
                     len(all_entries), len(candidates))

        categories = self._categorize_isa_entries(all_entries)
        logger.info("  ISA categorization: %d categories", len(categories))
        for cat_name, entry_names in categories.items():
            logger.info("    %s: %d entries", cat_name, len(entry_names))

        # Build a name -> list[ISAEntry] index for reassembly
        name_to_entries: dict[str, list[ISAEntry]] = {}
        for entry in all_entries:
            name_to_entries.setdefault(entry.name, []).append(entry)

        # Reassemble by category
        parts: list[str] = []
        categorized_names: set[str] = set()
        for cat_name, entry_names in categories.items():
            section_entries: list[str] = []
            for name in entry_names:
                for entry in name_to_entries.get(name, []):
                    section_entries.append(entry.markdown)
                categorized_names.add(name)
            if section_entries:
                parts.append(f"## {cat_name}\n\n" + "\n\n---\n\n".join(section_entries))

        # Fallback: any entries not assigned to a category go into "Other"
        other_entries = [
            e.markdown for e in all_entries if e.name not in categorized_names
        ]
        if other_entries:
            parts.append("## Other\n\n" + "\n\n---\n\n".join(other_entries))

        combined = "\n\n".join(parts)
        logger.info("  ISA docs assembled: %d chars, %d categories, %d entries",
                     len(combined), len(categories), len(all_entries))
        return combined

    def _categorize_isa_entries(
        self, entries: list[ISAEntry]
    ) -> dict[str, list[str]]:
        """Use the main LLM to group ISA entries into functional categories."""
        # Build compact index: deduplicate by name for the prompt
        seen: dict[str, str] = {}
        for entry in entries:
            if entry.name not in seen:
                seen[entry.name] = entry.description

        index_lines = [f"{name}: {desc}" for name, desc in seen.items()]
        index_text = "\n".join(index_lines)

        prompt = f"""Below is a list of API/instruction entries from the hardware SDK. Each line is "name: description".

Group these entries into functional categories (e.g., "Memory Operations", "Math and Arithmetic", "Tensor Operations", "Control Flow", "Data Types and Constants", etc.).

Return a JSON object mapping category names to lists of entry names. Order categories from fundamental to specialized. Every entry name must appear in exactly one category.

=== ENTRIES ===
{index_text}
=== END ===

Return ONLY a JSON object:"""

        responses = self._chat(prompt=prompt, num_candidates=1, temperature=0)
        raw = responses[0].strip() if responses else "{}"

        try:
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                result = json.loads(json_match.group())
                if isinstance(result, dict) and all(
                    isinstance(v, list) for v in result.values()
                ):
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

        logger.warning("Failed to parse categorization JSON, using single category")
        return {"API Reference": list(seen.keys())}

    def _build_isa_boundary_prompt(self, key: str, text: str) -> str:
        """Build a prompt that asks the LLM to identify entry boundaries, not rewrite content."""
        # Number lines so the LLM can reference them
        lines = text.split("\n")
        numbered = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
        numbered = _truncate(numbered, max_chars=120_000)
        return f"""Below is a numbered source file "{key}". Identify ALL API/instruction reference entries (functions, classes, methods, enums, instructions).

For each entry, return a JSON object with:
- "name": the entry name (function, class, instruction, etc.)
- "description": a one-line summary
- "start_line": the line number where this entry begins (inclusive)
- "end_line": the line number where this entry ends (inclusive)

RULES:
- Include the FULL definition and documentation for each entry (signature, docstring, all parameters, examples)
- Do NOT overlap line ranges between entries
- SKIP imports, module-level comments, tutorials, and narrative text
- If no API reference content is found, return an empty array []

Return ONLY a JSON array:

=== FILE (numbered lines) ===
{numbered}
=== END ===

JSON array:"""

    def _parse_boundary_response(
        self, raw: str, source_text: str, source_key: str
    ) -> list[ISAEntry]:
        """Parse LLM boundary response and extract content verbatim from source."""
        if not raw:
            return []

        lines = source_text.split("\n")

        try:
            arr_match = re.search(r"\[[\s\S]*\]", raw)
            if not arr_match:
                return _parse_markdown_entries(raw, source_key)
            items = json.loads(arr_match.group())
            entries: list[ISAEntry] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                desc = str(item.get("description", "")).strip()
                start = int(item.get("start_line", 0))
                end = int(item.get("end_line", 0))
                if start < 1 or end < start:
                    continue
                # Copy verbatim from source (1-indexed to 0-indexed)
                content = "\n".join(lines[start - 1:end])
                md = f"### {name}\n\n{content}"
                entries.append(ISAEntry(
                    name=name, description=desc,
                    source_key=source_key, markdown=md,
                ))
            if entries:
                return entries
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return _parse_markdown_entries(raw, source_key)

    def _extract_code_examples(self, items: list[tuple[str, str]]) -> str:
        """Extract code examples from the examples bucket in parallel (language-agnostic)."""
        candidates = [
            (k, t) for k, t in items if len(t.strip()) > 200
        ]
        candidates.sort(key=lambda x: len(x[1]), reverse=True)

        if not candidates:
            logger.info("  Code examples: no candidates with >200 chars (%d total in bucket)", len(items))
            return ""

        logger.info("  Code examples: processing %d candidates in parallel", len(candidates))

        prompts: list[str] = []
        basenames: list[str] = []
        for key, text in candidates:
            basename = key.rsplit("/", 1)[-1] if "/" in key else key
            basenames.append(basename)
            text_trunc = _truncate(text, max_chars=40_000)
            prompts.append(
                f'Below is code from example file "{basename}". '
                "Extract the core code examples that demonstrate API usage.\n\n"
                "RULES:\n"
                "- Keep the full function bodies including decorators and type hints\n"
                "- Include necessary imports at the top\n"
                "- REMOVE test harnesses, benchmarking code, main blocks, "
                "and setup boilerplate\n"
                "- Wrap each example in a fenced code block with the appropriate language\n"
                '- If no meaningful code examples are found, output "NO_EXAMPLE"\n\n'
                f"=== FILE ===\n{text_trunc}\n=== END ===\n\n"
                "Extracted code examples:"
            )

        all_responses = self._chat_light_async(
            prompts, num_candidates=1, temperature=0,
        )

        parts: list[str] = []
        for basename, (key, _), responses in zip(basenames, candidates, all_responses):
            extracted = (responses[0].strip() if responses else "")
            if extracted and "NO_EXAMPLE" not in extracted:
                parts.append(f"## {basename}\n\n{extracted}")
                logger.info("    Example: %s -> %d chars", key, len(extracted))
            else:
                logger.info("    Example: %s -> no usable examples", key)

        logger.info("  Code examples: extracted %d/%d files", len(parts), len(candidates))
        return "\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Optimization menu
    # ------------------------------------------------------------------

    def _synthesize_optimization_menu(self, items: list[tuple[str, str]]) -> list[str]:
        """Synthesize optimization strategies from routed documentation."""
        defaults = [
            "reduce data movement",
            "overlap data movement and compute",
            "loop tiling",
            "loop reordering and restructuring",
            "fuse operations",
            "use lower precision",
            "double buffering",
            "software pipelining",
            "hoist redundant operations out of loops",
            "simplify or remove unnecessary code",
        ]

        content = _items_to_text(items, max_chars=100_000)
        if not content.strip():
            defaults.append("Other methods not listed here.")
            return defaults

        prompt = f"""Below is documentation about the hardware target.

The agent uses an "optimization menu" -- a list of strategies it picks from when planning how to optimize a kernel. Here are the generic strategies already included:

{chr(10).join("- " + d for d in defaults)}

Generate 5-15 ADDITIONAL hardware-specific optimization strategies from the documentation.

EXAMPLES:
- "keep reusable data in local memory across outer loop iterations"
- "fuse dependent operations into a single loop to avoid HBM round-trips"
- "multi-tile grouping"
- "Add additional loop levels so larger blocks of data can be loaded"
- "supertile fuse-and-reuse"
- "fuse instructions"
- "stronger tiling for contraction / moving-free split"
- "delay division until after all reductions are complete"
- "use the streaming softmax with running max and scaling trick"

RULES:
- Avoid making the strategies overly specific
- Each strategy MUST be under 15 words.
- Write as a short action phrase, not a full sentence
- ONLY kernel-level and algorithmic optimizations
- NO deployment, serving, threading, batching, environment, profiling, or configuration advice
- Do NOT repeat the generic strategies above

Return each strategy on its own line, prefixed with "- ".

=== DOCUMENTATION ===
{content}
=== END DOCUMENTATION ===

Hardware-specific optimization strategies:"""

        responses = self._chat(prompt=prompt, num_candidates=1, temperature=0)
        raw = responses[0].strip() if responses else ""

        hw_specific: list[str] = []
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                line = line[2:].strip()
            elif line.startswith("* "):
                line = line[2:].strip()
            elif re.match(r"^\d+\.\s", line):
                line = re.sub(r"^\d+\.\s*", "", line).strip()
            else:
                continue
            if line and len(line) > 10:
                hw_specific.append(line)

        combined = defaults + hw_specific
        combined.append("Other methods not listed here.")
        return combined

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _synthesize_rules(self, items: list[tuple[str, str]],
                          architecture: str = "", isa_docs: str = "") -> dict[str, list[str]]:
        """Extract rules and constraints from routed documentation."""
        base_rules: dict[str, list[str]] = {
            "general": [
                "The rewritten program should be semantically equivalent to the original program, within a small numerical tolerance.",
                "Keep the same function name and signature as the original program (helper functions can be renamed or deleted).",
            ],
            "planning": [
                "Limit the scope of the plan to the selected optimization.",
                "Do not count out any of the optimizations unless they are clearly irrelevant to the code.",
            ],
            "coding": [
                "Wrap the generated code with ```python at the beginning and ``` at the end.",
            ],
        }

        content = _items_to_text(items, max_chars=100_000)
        if not content.strip():
            return base_rules

        # Provide architecture and ISA as additional context
        context_parts: list[str] = []
        if architecture:
            context_parts.append(f"=== ARCHITECTURE SUMMARY ===\n{architecture}\n=== END ===\n")
        if isa_docs:
            # Include just the section headers and first line of each to keep it compact
            isa_summary_lines: list[str] = []
            for line in isa_docs.split("\n"):
                if line.startswith("## ") or line.startswith("### "):
                    isa_summary_lines.append(line)
            if isa_summary_lines:
                context_parts.append(
                    f"=== ISA API NAMES (for reference) ===\n"
                    + "\n".join(isa_summary_lines[:200])
                    + "\n=== END ===\n"
                )
        extra_context = "\n".join(context_parts)

        prompt = f"""Below is documentation about the hardware target's programming model.

{extra_context}

Extract rules and constraints that the agent MUST follow when generating optimized kernel code. Focus on:
- Programming model constraints (e.g., "the partition dimension is always 128", "free dimensions must be multiples of 512 for the Tensor Engine")
- Memory layout rules (e.g., "tiles in scratchpad must not exceed X KB", "data must be contiguous along dimension Y")
- Correctness constraints (e.g., "loop variables cannot be used for indexing in this context")
- API usage pitfalls (e.g., "reduction output must be stored before reuse")
- Try not to generate too many rules. Only include rules that are truly critical to generating correct code and relevant to the context provided above.

Categorize each rule into one of:
- "general" -- applies to both planning and coding phases
- "planning" -- applies only when generating optimization plans
- "coding" -- applies only when generating code

Return as a JSON object with keys "general", "planning", "coding", each mapping to a list of rule strings.

=== DOCUMENTATION ===
{content}
=== END DOCUMENTATION ===

Return ONLY a JSON object:"""

        responses = self._chat_light(prompt=prompt, num_candidates=1, temperature=0)
        raw = responses[0].strip() if responses else "{}"

        try:
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                extracted = json.loads(json_match.group())
                for key in ("general", "planning", "coding"):
                    if key in extracted and isinstance(extracted[key], list):
                        base_rules[key].extend(extracted[key])
                        logger.info("  Rules extracted (%s): %d new rules", key, len(extracted[key]))
        except json.JSONDecodeError:
            logger.warning("Failed to parse rules JSON, using defaults only")

        return base_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_markdown_entries(text: str, source_key: str) -> list[ISAEntry]:
    """Split markdown text on ### boundaries into ISAEntry objects (fallback parser)."""
    entries: list[ISAEntry] = []
    parts = re.split(r"(?=^### )", text, flags=re.MULTILINE)
    for part in parts:
        part = part.strip()
        if not part.startswith("### "):
            continue
        first_line = part.split("\n", 1)[0]
        name = first_line[4:].strip()
        rest = part.split("\n", 1)[1].strip() if "\n" in part else ""
        desc_line = ""
        for line in rest.split("\n"):
            line = line.strip()
            if line and not line.startswith("```") and not line.startswith("|"):
                desc_line = line
                break
        entries.append(ISAEntry(
            name=name, description=desc_line,
            source_key=source_key, markdown=part,
        ))
    return entries


def _items_to_text(items: list[tuple[str, str]], max_chars: int) -> str:
    """Concatenate (key, text) pairs into a single string, respecting a size budget."""
    parts: list[str] = []
    total = 0
    for key, text in items:
        chunk = f"--- {key} ---\n{text}\n"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 500:
                parts.append(chunk[:remaining] + "\n[... truncated ...]")
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated ...]"
