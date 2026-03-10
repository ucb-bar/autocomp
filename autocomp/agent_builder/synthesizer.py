"""
Component synthesis for the Agent Builder.

LLM-based content routing + per-file extraction to distill ingested knowledge
into agent components: architecture summary, ISA docs, optimization menu,
rules, and code examples.

Stage 1   (route):   LLM classifies each content item into component buckets.
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


_ROUTE_PREVIEW_CHARS = 6000

_BUCKET_NAMES = ["isa", "architecture", "optimization", "rules", "examples"]

_BUCKET_DESCRIPTIONS = {
    "isa": "API / instruction-set reference (function signatures, parameter descriptions, class definitions, instruction semantics, and their inline usage examples/snippets). NOT standalone tutorials or sample programs.",
    "architecture": "hardware architecture (memory hierarchy, compute units, system design, chip overview, programming model)",
    "optimization": "performance optimization guidance (tuning strategies, optimization techniques, pipelining, tiling, matrix multiplication patterns)",
    "rules": "programming model constraints and rules (correctness constraints, tile size rules, memory layout requirements, API usage pitfalls, changelogs)",
    "examples": "code examples, tutorials, sample kernels, example implementations, operator support lists, framework integration docs",
    "skip": "not relevant to any of the above (release notes, navigation pages, setup/config, unrelated docs)",
}


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

class ComponentSynthesizer:
    """
    Synthesizer that distills ingested knowledge into agent components.

    Uses LLM-based content routing followed by per-file LLM extraction
    for each component.
    """

    def __init__(self, llm_client: LLMClient, light_llm_client: LLMClient | None = None,
                 description: str = ""):
        self.llm = llm_client
        self.light_llm = light_llm_client or llm_client
        self._context_prefix = f"Context: {description}\n\n" if description else ""

    def _chat(self, prompt: str, **kwargs) -> list[str]:
        results = self.llm.chat_async(
            [prompt], **kwargs,
        )
        return results[0] if results else []

    def _chat_light(self, prompt: str, **kwargs) -> list[str]:
        results = self.light_llm.chat_async(
            [prompt], **kwargs,
        )
        return results[0] if results else []

    def _chat_light_async(self, prompts: list[str], **kwargs) -> list[list[str]]:
        return self.light_llm.chat_async(
            prompts, **kwargs,
        )

    # ------------------------------------------------------------------
    # LLM-based content routing (replaces heuristic routing + filter)
    # ------------------------------------------------------------------

    def _pre_filter(
        self, items: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Binary relevance filter: remove items clearly outside the agent's scope.

        Runs one light LLM call per item in parallel with a simple yes/no prompt.
        Skipped when no user context is provided.
        """
        if not self._context_prefix or not items:
            return items

        prompts: list[str] = []
        for key, text in items:
            preview = text[:_ROUTE_PREVIEW_CHARS]
            prompts.append(
                f"Source: {key}\n\n"
                f"=== CONTENT PREVIEW ===\n{preview}\n=== END ===\n\n"
                f"{self._context_prefix}"
                f"Is this content relevant to the agent's task described above?\n\n"
                f"Answer YES or NO."
            )

        logger.info("Pre-filter: checking relevance of %d items", len(prompts))
        all_responses = self._chat_light_async(
            prompts, num_candidates=1, temperature=0,
        )

        filtered: list[tuple[str, str]] = []
        removed = 0
        for (key, text), responses in zip(items, all_responses):
            raw = (responses[0].strip() if responses else "").lower()
            if "no" in raw and "yes" not in raw:
                removed += 1
            else:
                filtered.append((key, text))

        logger.info("Pre-filter: %d -> %d items (%d removed)",
                     len(items), len(filtered), removed)
        return filtered

    def _llm_route_content(
        self, indices: list[SourceIndex]
    ) -> dict[str, list[tuple[str, str]]]:
        """Classify each content item into component buckets via LLM.

        One lightweight LLM call per item (parallelised via chat_async),
        examining both the key (filename/URL) and a content preview.
        Each item can be assigned to multiple buckets.
        """
        buckets: dict[str, list[tuple[str, str]]] = {b: [] for b in _BUCKET_NAMES}

        all_items: list[tuple[str, str]] = []
        for idx in indices:
            for key, text in idx.content.items():
                if text and text.strip():
                    all_items.append((key, text))

        if not all_items:
            return buckets

        all_items = self._pre_filter(all_items)

        if not all_items:
            return buckets

        bucket_list = "\n".join(
            f"- {name}: {_BUCKET_DESCRIPTIONS[name]}" for name in _BUCKET_NAMES
        )
        skip_desc = _BUCKET_DESCRIPTIONS["skip"]

        prompts: list[str] = []
        context_note = ""
        if self._context_prefix:
            context_note = (
                f"\n{self._context_prefix}"
                f"SCOPE: Only classify content into a category if it is directly relevant "
                f"to the agent's task described above. Content about topics outside that "
                f"scope should be classified as 'skip', even if it technically matches a "
                f"category description.\n\n"
            )
        for key, text in all_items:
            preview = text[:_ROUTE_PREVIEW_CHARS]
            prompts.append(
                f"Source: {key}\n\n"
                f"=== CONTENT PREVIEW ===\n{preview}\n=== END ===\n\n"
                f"Classify this content into one or more categories.\n\n"
                f"Categories:\n{bucket_list}\n- skip: {skip_desc}\n\n"
                f"IMPORTANT DISTINCTIONS:\n"
                f"- 'isa' is ONLY for API/instruction reference docs (function signatures, parameter tables, class definitions, and their inline usage examples) "
                f"that the agent will directly use when writing code. "
                f"API docs for tools, services, or libraries outside the agent's scope are NOT 'isa'.\n"
                f"- Files that are primarily standalone runnable programs/tutorials (not API reference) are 'examples', not 'isa'. "
                f"However, API reference files that contain inline code snippets within docstrings are still 'isa'.\n"
                f"- Operator support tables (lists of supported ops for a framework) are 'examples', not 'isa'.\n"
                f"- 'optimization' is ONLY for guidance on optimizing code that the agent writes. "
                f"Optimization advice for things outside the agent's scope is 'skip'.\n\n"
                f"{context_note}"
                f"Return ONLY a comma-separated list of category names "
                f"(from: {', '.join(_BUCKET_NAMES)}, skip). "
                f"Example: isa, rules"
            )

        logger.info("LLM routing: classifying %d items", len(prompts))
        all_responses = self._chat_light_async(
            prompts, num_candidates=1, temperature=0,
        )

        skipped = 0
        for (key, text), responses in zip(all_items, all_responses):
            raw = (responses[0].strip() if responses else "").lower()
            assigned = False
            for bucket_name in _BUCKET_NAMES:
                if bucket_name in raw:
                    buckets[bucket_name].append((key, text))
                    assigned = True
            if not assigned:
                skipped += 1

        for bname, items in buckets.items():
            total_chars = sum(len(t) for _, t in items)
            logger.info("  %s: %d files, %d chars", bname, len(items), total_chars)
        if skipped:
            logger.info("  skipped: %d files", skipped)

        return buckets

    def synthesize(self, indices: list[SourceIndex]) -> SynthesizedComponents:
        """Run the full synthesis pipeline: route -> extract -> merge."""
        t0 = time.time()

        # Stage 1: LLM-based content routing
        logger.info("ComponentSynthesizer: Stage 1 -- LLM content routing")
        buckets = self._llm_route_content(indices)
        t_route = time.time()
        logger.info("  Routing took %.1fs", t_route - t0)

        # Stage 2 + 3: Extract and merge each component
        logger.info("ComponentSynthesizer: Stage 2 -- extract, merge, synthesize")

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

    _ARCH_SINGLE_DOC_CHARS = 30_000
    _ARCH_MERGE_CHARS = 60_000

    def _synthesize_architecture(self, items: list[tuple[str, str]]) -> str:
        """Summarize hardware architecture from routed content.

        For small inputs, sends everything in a single LLM call.
        For larger inputs, summarizes each document individually in parallel,
        then merges the summaries.
        """
        if not items:
            return "No architecture documentation found. Please add manually."

        total_chars = sum(len(t) for _, t in items)
        if total_chars <= self._ARCH_SINGLE_DOC_CHARS:
            return self._arch_single_pass(items)

        return self._arch_map_reduce(items)

    _ARCH_PROMPT = (
        "Write a concise but thorough hardware architecture summary covering:\n"
        "- What the hardware is and its programming model\n"
        "- Memory hierarchy (on-chip memories, caches, off-chip memory, sizes, bandwidths)\n"
        "- Compute units (types, capabilities, throughput)\n"
        "- Key constraints and characteristics that affect code optimization\n\n"
        "This summary will be included at the top of every prompt to the agent. "
        "It should give the agent the context it needs to make good optimization decisions.\n\n"
        "Write in clear technical prose. Do not include instructions to the reader."
    )

    def _arch_single_pass(self, items: list[tuple[str, str]]) -> str:
        content = _items_to_text(items, max_chars=self._ARCH_SINGLE_DOC_CHARS)
        prompt = (
            f"=== DOCUMENTATION ===\n{content}\n=== END DOCUMENTATION ===\n\n"
            f"{self._context_prefix}"
            f"{self._ARCH_PROMPT}\n\n"
            f"Hardware Architecture Summary:"
        )
        responses = self._chat(prompt=prompt, num_candidates=1, temperature=0)
        return responses[0].strip() if responses else "Architecture summary generation failed."

    def _arch_map_reduce(self, items: list[tuple[str, str]]) -> str:
        # Map: summarize each document individually in parallel
        logger.info("Architecture: map - summarize each document individually in parallel")
        prompts: list[str] = []
        for key, text in items:
            truncated = _truncate(text, max_chars=self._ARCH_SINGLE_DOC_CHARS)
            prompts.append(
                f"Source: {key}\n\n"
                f"=== DOCUMENT ===\n{truncated}\n=== END ===\n\n"
                f"{self._context_prefix}"
                f"Extract architecture-relevant information from this document.\n"
                f"Focus on: hardware capabilities, memory hierarchy, compute units, "
                f"constraints, and performance characteristics.\n"
                f"Be thorough -- include specific numbers (sizes, bandwidths, limits).\n\n"
                f"Architecture-relevant notes:"
            )

        all_responses = self._chat(
            prompt=prompts, num_candidates=1, temperature=0,
        ) if len(prompts) == 1 else [
            r[0] if r else "" for r in
            self.llm.chat_async(
                prompts,
                num_candidates=1, temperature=0,
            )
        ]

        logger.info("Architecture: reduce - merge summaries into final architecture summary")
        summaries = "\n\n".join(
            f"--- {items[i][0]} ---\n{all_responses[i]}"
            for i in range(len(items)) if i < len(all_responses) and all_responses[i]
        )
        summaries = _truncate(summaries, max_chars=self._ARCH_MERGE_CHARS)

        # Reduce: merge summaries into final architecture summary
        prompt = (
            f"=== EXTRACTED NOTES ===\n{summaries}\n=== END ===\n\n"
            f"Above are architecture notes extracted from multiple documents "
            f"about the same hardware target.\n\n"
            f"{self._context_prefix}"
            f"{self._ARCH_PROMPT}\n\n"
            f"Hardware Architecture Summary:"
        )
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
        logger.info("ISA docs: extract - identify entry/exit boundaries and copy content verbatim")
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
            entries = [e for e in entries if self._is_valid_isa_entry(e)]
            logger.info("  ISA extract: %s -> %d entries", key, len(entries))
            all_entries.extend(entries)

        if not all_entries:
            return "No ISA documentation found. Please add manually."

        logger.info("  ISA extraction complete: %d entries from %d files",
                     len(all_entries), len(candidates))

        all_entries = self._filter_isa_entries(all_entries)

        if not all_entries:
            return "No ISA documentation found. Please add manually."

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

    def _filter_isa_entries(self, entries: list[ISAEntry]) -> list[ISAEntry]:
        """Use the LLM to filter ISA entries down to those directly used in kernel code."""
        seen: dict[str, str] = {}
        for entry in entries:
            if entry.name not in seen:
                seen[entry.name] = entry.description

        index_lines = [f"{name}: {desc}" for name, desc in seen.items()]
        index_text = "\n".join(index_lines)

        prompt = f"""{self._context_prefix}Below are API entries extracted from the SDK. Each line is "name: description".

=== ENTRIES ({len(seen)} total) ===
{index_text}
=== END ===

Keep ONLY entries the agent would directly call or reference when writing optimized code. Remove APIs that are outside the agent's scope as described above.

Return ONLY a JSON array of entry names to keep.

JSON array:"""

        responses = self._chat(prompt=prompt, num_candidates=1, temperature=0)
        raw = responses[0].strip() if responses else "[]"

        try:
            arr_match = re.search(r"\[[\s\S]*\]", raw)
            if arr_match:
                keep_names = set(json.loads(arr_match.group()))
                filtered = [e for e in entries if e.name in keep_names]
                logger.info("  ISA filtering: %d -> %d entries (%d removed)",
                            len(entries), len(filtered), len(entries) - len(filtered))
                return filtered
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        logger.warning("  ISA filtering: failed to parse LLM response, keeping all entries")
        return entries

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

=== ENTRIES ===
{index_text}
=== END ===

{self._context_prefix}Group these entries into functional categories (e.g., "Memory Operations", "Math and Arithmetic", "Tensor Operations", "Control Flow", "Data Types and Constants", etc.).

Return a JSON object mapping category names to lists of entry names. Order categories from fundamental to specialized. Every entry name must appear in exactly one category.

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
        lines = text.split("\n")
        numbered = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
        numbered = _truncate(numbered, max_chars=120_000)
        return f"""Below is a numbered source file "{key}". Identify ALL API/instruction reference entries (functions, classes, methods, enums, instructions) that have proper documentation.

=== FILE (numbered lines) ===
{numbered}
=== END ===

{self._context_prefix}For each entry, return a JSON object with:
- "name": the entry name (function, class, instruction, etc.)
- "description": a one-line summary
- "start_line": the line number where this entry begins (inclusive)
- "end_line": the line number where this entry ends (inclusive)

RULES:
- Include the FULL definition and documentation for each entry (signature, description, docstring, all parameters, return type, AND any "Example:" or "Examples:" sections with code blocks that follow)
- KEEP inline code examples/snippets that appear inside an API entry's docstring — these show correct usage patterns and are essential
- Do NOT overlap line ranges between entries
- ONLY include entries that are API/library reference: function signatures with documented parameters, descriptions, class definitions, enum definitions, instruction specifications, usage examples
- If no API reference content is found, return an empty array []

Return ONLY a JSON array:

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

    @staticmethod
    def _is_valid_isa_entry(entry: ISAEntry) -> bool:
        """Filter out entries that are clearly not ISA reference material."""
        md = entry.markdown
        lines = [l for l in md.split("\n") if l.strip() and not l.startswith("### ")]
        if len(lines) <= 1:
            return False
        # If the entire body is just table rows, it's a support list entry
        non_table_lines = [l for l in lines if not l.strip().startswith("|") and l.strip() != "---"]
        if not non_table_lines:
            return False
        return True

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
                f"Source: {basename}\n\n"
                f"=== FILE ===\n{text_trunc}\n=== END ===\n\n"
                f"{self._context_prefix}"
                "Extract the core code examples that demonstrate API usage.\n\n"
                "RULES:\n"
                "- Keep the full function bodies including decorators and type hints\n"
                "- Include necessary imports at the top\n"
                "- REMOVE test harnesses, benchmarking code, main blocks, "
                "and setup boilerplate\n"
                "- Wrap each example in a fenced code block with the appropriate language\n"
                '- If no meaningful code examples are found, output "NO_EXAMPLE"\n\n'
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

=== DOCUMENTATION ===
{content}
=== END DOCUMENTATION ===

The agent uses an "optimization menu" -- a list of strategies it picks from when planning how to optimize a single kernel's code. Here are the generic strategies already included:

{chr(10).join("- " + d for d in defaults)}

Generate 5-15 ADDITIONAL hardware-specific performance optimization strategies from the documentation.

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

{self._context_prefix}RULES:
- Avoid making the strategies overly specific
- Each strategy MUST be under 15 words.
- Write as a short action phrase, not a full sentence
- ONLY optimizations within the agent's scope as described above
- NOT optimizations outside that scope, even if mentioned in the documentation
- Do NOT repeat the generic strategies above

Return each strategy on its own line, prefixed with "- ".

Performance optimization strategies:"""

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

=== DOCUMENTATION ===
{content}
=== END DOCUMENTATION ===

{extra_context}

{self._context_prefix}Extract critical rules that the agent MUST follow when generating optimized kernel code. Focus on:
- Programming model constraints (e.g., "the partition dimension is always 128", "free dimensions must be multiples of 512 for the Tensor Engine")
- Memory layout rules (e.g., "tiles in scratchpad must not exceed X KB", "data must be contiguous along dimension Y")
- Correctness constraints (e.g., "loop variables cannot be used for indexing in this context")
- API usage pitfalls (e.g., "reduction output must be stored before reuse")
Only include rules that are critical to generating CORRECT code. Do NOT include optimization tips, performance advice, or general best practices -- those belong in the optimization menu. Return AT MOST 5 rules per category — pick the most critical ones only.

Categorize each rule into one of:
- "general" -- applies to both planning and coding phases
- "planning" -- applies only when generating optimization plans
- "coding" -- applies only when generating code

Return as a JSON object with keys "general", "planning", "coding", each mapping to a list of rule strings.

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
