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
    translate_menu: list[str]
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
    "rules": "programming model constraints and rules (correctness constraints, tile size rules, memory layout requirements, API usage pitfalls)",
    "examples": "code examples, tutorials, sample kernels, example implementations, framework integration docs",
    "skip": "not relevant to any of the above (release notes, changelogs, navigation pages, setup/config, unrelated docs)",
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

    # Default context budget: ~150K chars leaves generous room for prompt
    # instructions within a 200K-token context window.
    DEFAULT_CONTEXT_BUDGET = 150_000

    def __init__(self, llm_client: LLMClient, light_llm_client: LLMClient | None = None,
                 agent_scope: str = "", context_budget: int = DEFAULT_CONTEXT_BUDGET):
        self.llm = llm_client
        self.light_llm = light_llm_client or llm_client
        self._context_prefix = f"Agent scope: {agent_scope}\n\n" if agent_scope else ""
        self.context_budget = context_budget

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
                f"Is this content directly relevant to the agent's task described above?\n\n"
                f"Answer YES or NO."
            )

        logger.info("Pre-filter: checking relevance of %d items", len(prompts))
        all_responses = self._chat_light_async(
            prompts, num_samples=1, temperature=0,
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
        seen_keys: set[str] = set()
        for idx in indices:
            for key, text in idx.content.items():
                if text and text.strip() and key not in seen_keys:
                    seen_keys.add(key)
                    all_items.append((key, text))

        n_total = sum(len(idx.content) for idx in indices)
        if len(all_items) < n_total:
            logger.info("LLM routing: deduplicated %d -> %d items across %d sources",
                        n_total, len(all_items), len(indices))

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
                f"- 'isa' is ONLY for API/instruction reference docs (function signatures, parameter tables, class definitions, inline usage examples, and other closely related documentation) "
                f"that the agent will directly use when writing code. "
                f"API docs for tools, services, or libraries outside the agent's scope are NOT 'isa'.\n"
                f"- Files that are primarily standalone runnable programs/tutorials (not API reference) are 'examples', not 'isa'. "
                f"However, API reference files that contain inline code snippets within docstrings are still 'isa'.\n"
                f"- 'optimization' is ONLY for guidance on optimizing code that the agent writes. "
                f"Optimization advice for things outside the agent's scope is 'skip'.\n\n"
                f"{context_note}"
                f"Return ONLY a comma-separated list of category names "
                f"(from: {', '.join(_BUCKET_NAMES)}, skip). "
                f"Example: isa, rules"
            )

        logger.info("LLM routing: classifying %d items", len(prompts))
        all_responses = self._chat_light_async(
            prompts, num_samples=1, temperature=0,
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
        translate_menu = self._synthesize_translate_menu(
            buckets["optimization"], architecture=architecture, isa_docs=isa_docs,
            code_examples_raw=buckets["examples"],
        )
        logger.info("  Translate menu: %d strategies (%.1fs)", len(translate_menu), time.time() - t_start)

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
            translate_menu=translate_menu,
            rules=rules,
            code_examples=code_examples,
        )

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def _synthesize_architecture(self, items: list[tuple[str, str]]) -> str:
        """Summarize hardware architecture from routed content.

        For small inputs, sends everything in a single LLM call.
        For larger inputs, summarizes each document individually in parallel,
        then merges the summaries.
        """
        if not items:
            return "No architecture documentation found. Please add manually."

        items = _chunk_items(items, self.context_budget)
        total_chars = sum(len(t) for _, t in items)
        if total_chars <= self.context_budget:
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
        content = _items_to_text(items, max_chars=self.context_budget)
        prompt = (
            f"=== DOCUMENTATION ===\n{content}\n=== END DOCUMENTATION ===\n\n"
            f"{self._context_prefix}"
            f"{self._ARCH_PROMPT}\n\n"
            f"Hardware Architecture Summary:"
        )
        responses = self._chat(prompt=prompt, num_samples=1, temperature=0)
        return responses[0].strip() if responses else "Architecture summary generation failed."

    def _arch_map_reduce(self, items: list[tuple[str, str]]) -> str:
        # Map: summarize each document individually in parallel
        logger.info("Architecture: map - summarize each document individually in parallel")
        prompts: list[str] = []
        for key, text in items:
            truncated = _truncate(text, max_chars=self.context_budget)
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

        all_responses = [
            r[0] if r else "" for r in
            self.llm.chat_async(
                prompts, num_samples=1, temperature=0,
            )
        ]

        logger.info("Architecture: reduce - merge summaries into final architecture summary")
        summaries = "\n\n".join(
            f"--- {items[i][0]} ---\n{all_responses[i]}"
            for i in range(len(items)) if i < len(all_responses) and all_responses[i]
        )
        summaries = _truncate(summaries, max_chars=self.context_budget)

        # Reduce: merge summaries into final architecture summary
        prompt = (
            f"=== EXTRACTED NOTES ===\n{summaries}\n=== END ===\n\n"
            f"Above are architecture notes extracted from multiple documents "
            f"about the same hardware target.\n\n"
            f"{self._context_prefix}"
            f"{self._ARCH_PROMPT}\n\n"
            f"Hardware Architecture Summary:"
        )
        responses = self._chat(prompt=prompt, num_samples=1, temperature=0)
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
            prompts, num_samples=1, temperature=0,
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

        all_entries = self._filter_isa_entries(all_entries)

        if not all_entries:
            return "No ISA documentation found. Please add manually."

        # Merge entries with the same name: combine content, skip near-duplicates
        grouped: dict[str, list[ISAEntry]] = {}
        for entry in all_entries:
            grouped.setdefault(entry.name, []).append(entry)

        merged: list[ISAEntry] = []
        for name, entries in grouped.items():
            if len(entries) == 1:
                merged.append(entries[0])
                continue
            # Sort longest first, skip entries whose content is a substring of a longer one
            entries.sort(key=lambda e: len(e.markdown), reverse=True)
            unique_parts: list[str] = []
            for e in entries:
                body = e.markdown.split("\n", 2)[-1].strip() if "\n" in e.markdown else e.markdown
                if not any(body in existing for existing in unique_parts):
                    unique_parts.append(e.markdown)
            combined_md = "\n\n".join(unique_parts)
            merged.append(ISAEntry(
                name=name,
                description=entries[0].description,
                source_key=entries[0].source_key,
                markdown=combined_md,
            ))
        if len(merged) < len(all_entries):
            logger.info("  ISA merge: %d -> %d entries (%d duplicates merged)",
                         len(all_entries), len(merged), len(all_entries) - len(merged))
        all_entries = merged

        # Content-based dedup: remove entries whose body is a near-duplicate
        # of an earlier entry (catches same content extracted under different names).
        deduped: list[ISAEntry] = []
        seen_bodies: list[str] = []
        for entry in all_entries:
            body = entry.markdown.split("\n", 2)[-1].strip() if "\n" in entry.markdown else entry.markdown
            if any(body in prev or prev in body for prev in seen_bodies):
                logger.info("  ISA dedup: dropping '%s' (content duplicate)", entry.name)
                continue
            seen_bodies.append(body)
            deduped.append(entry)
        if len(deduped) < len(all_entries):
            logger.info("  ISA content dedup: %d -> %d entries",
                         len(all_entries), len(deduped))
        all_entries = deduped

        categories = self._categorize_isa_entries(all_entries)
        logger.info("  ISA categorization: %d categories", len(categories))
        for cat_name, entry_names in categories.items():
            logger.info("    %s: %d entries", cat_name, len(entry_names))

        # Build a name -> entry index for reassembly
        name_to_entry: dict[str, ISAEntry] = {e.name: e for e in all_entries}

        # Reassemble by category
        parts: list[str] = []
        categorized_names: set[str] = set()
        for cat_name, entry_names in categories.items():
            section_entries: list[str] = []
            for name in entry_names:
                entry = name_to_entry.get(name)
                if entry:
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

        prompt = f"""Below are API entries extracted from the SDK. Each line is "name: description".

=== ENTRIES ({len(seen)} total) ===
{index_text}
=== END ===

{self._context_prefix}Keep ONLY entries the agent would require to write optimized code, such as APIs/instructions it will call, conceptual references and explanations, or other closely related documentation. Remove APIs that are outside the agent's scope as described above.
Do not include standalone tutorials or sample programs, as those belong in the "examples" bucket. Skip release notes, changelogs, and other non-API/instruction documentation.

Return ONLY a JSON array of entry names to keep.

JSON array:"""

        responses = self._chat(prompt=prompt, num_samples=1, temperature=0)
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

        responses = self._chat(prompt=prompt, num_samples=1, temperature=0)
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
        numbered = _truncate(numbered, max_chars=self.context_budget)
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
- Include entries that are API/library reference: function signatures with documented parameters, descriptions, class definitions, enum definitions, instruction specifications, usage examples
- Also include conceptual references and explanations that document how APIs behave or interact
- Closely related documentation can also be included, such as additional information about particular parts of the API/instruction set.
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
                if len(content.strip()) < 20:
                    continue
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
            text_trunc = _truncate(text, max_chars=self.context_budget)
            prompts.append(
                f"Source: {basename}\n\n"
                f"=== FILE ===\n{text_trunc}\n=== END ===\n\n"
                f"{self._context_prefix}"
                "Extract the core code examples that demonstrate API usage.\n\n"
                "RULES:\n"
                "- Start with a single line: SUMMARY: followed by a 1-2 sentence "
                "description of what this document covers and what key concepts, "
                "APIs, or techniques it demonstrates\n"
                "- Then include the code examples as fenced code blocks\n"
                "- Keep the full function bodies including decorators and type hints\n"
                "- Include necessary imports at the top\n"
                "- REMOVE test harnesses, benchmarking code, main blocks, "
                "and setup boilerplate\n"
                "- Wrap each example in a fenced code block with the appropriate language\n"
                '- If no meaningful code examples are found, output "NO_EXAMPLE"\n\n'
                "Extracted code examples:"
            )

        all_responses = self._chat_light_async(
            prompts, num_samples=1, temperature=0,
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
        """Synthesize optimization strategies from routed documentation via map-reduce."""
        defaults = [
            "reduce data movement",
            "overlap data movement and compute",
            "cache reused data in local memory instead of reloading from main memory",
            "loop tiling",
            "loop reordering and restructuring",
            "loop unrolling",
            "fuse operations",
            "use lower precision",
            "double buffering",
            "software pipelining",
            "hoist redundant operations out of loops",
            "eliminate redundant computation",
            "simplify or remove unnecessary code",
            "try new parameter values",
            "rewrite the algorithm to reduce total work",
        ]

        if not items:
            return defaults

        items = _chunk_items(items, self.context_budget)
        total_chars = sum(len(t) for _, t in items)
        if total_chars <= self.context_budget:
            return self._opt_single_pass(items, defaults)

        return self._opt_map_reduce(items, defaults)

    def _opt_single_pass(self, items: list[tuple[str, str]], defaults: list[str]) -> list[str]:
        """Single-pass optimization menu when content fits in one call."""
        content = _items_to_text(items, max_chars=self.context_budget)
        raw = self._opt_reduce(content, defaults)
        return defaults + self._parse_bullet_lines(raw)

    def _opt_map_reduce(self, items: list[tuple[str, str]], defaults: list[str]) -> list[str]:
        """Map-reduce optimization menu for large content."""
        # Map: extract candidate strategies from each document in parallel
        logger.info("Optimization menu: map - extract strategies from %d documents", len(items))
        defaults_text = chr(10).join("- " + d for d in defaults)
        prompts: list[str] = []
        for key, text in items:
            truncated = _truncate(text, max_chars=self.context_budget)
            prompts.append(
                f"Source: {key}\n\n"
                f"=== DOCUMENT ===\n{truncated}\n=== END ===\n\n"
                f"{self._context_prefix}"
                f"Extract performance optimization strategies taught or demonstrated "
                f"in this document. Techniques demonstrated in tutorials and examples "
                f"have strong potential to be useful optimizations.\n\n"
                f"Most strategies should be general optimization ideas. "
                f"When a specific API or function is the key enabler of an optimization, "
                f"it is okay to name it, but focus on the fundamental optimization idea.\n\n"
                f"These generic strategies are already known (do NOT repeat them):\n"
                f"{defaults_text}\n\n"
                f"Return each NEW strategy on its own line, prefixed with \"- \". "
                f"If no new strategies are found, return \"- none\".\n\n"
                f"Optimization strategies:"
            )

        all_responses = [
            r[0] if r else "" for r in
            self.llm.chat_async(
                prompts, num_samples=1, temperature=0,
            )
        ]

        # Collect all candidate strategies from map step
        candidates: list[str] = []
        for i, resp in enumerate(all_responses):
            if not resp:
                continue
            lines = self._parse_bullet_lines(resp)
            if lines:
                logger.info("  %s: %d strategies", items[i][0], len(lines))
            candidates.extend(lines)

        logger.info("Optimization menu: reduce - merge %d candidates into final list", len(candidates))
        if not candidates:
            return defaults

        candidate_text = chr(10).join(f"- {c}" for c in candidates)
        candidate_text = _truncate(candidate_text, max_chars=self.context_budget)

        # Reduce: merge, deduplicate, and curate
        raw = self._opt_reduce(candidate_text, defaults, is_reduce=True)
        return defaults + self._parse_bullet_lines(raw)

    def _opt_reduce(self, content: str, defaults: list[str], is_reduce: bool = False) -> str:
        """Run the reduce/synthesis prompt for optimization menu."""
        defaults_text = chr(10).join("- " + d for d in defaults)
        if is_reduce:
            header = (
                "Below are candidate optimization strategies extracted from "
                "multiple documents about the hardware target."
            )
            content_label = "CANDIDATE STRATEGIES"
            instruction = (
                "Merge, deduplicate, and curate these into a final list of "
                "10-20 hardware-specific optimization strategies. "
            )
        else:
            header = "Below is documentation about the hardware target."
            content_label = "DOCUMENTATION"
            instruction = (
                "Generate 10-20 ADDITIONAL hardware-specific performance optimization "
                "strategies from the documentation."
            )

        prompt = f"""{header}

=== {content_label} ===
{content}
=== END {content_label} ===

The agent uses an "optimization menu" -- a list of strategies it picks from when planning how to optimize a single kernel's code. Here are the generic strategies already included:

{defaults_text}

{instruction}

EXAMPLES of good strategy descriptions:
- "multi-tile grouping"
- "supertile fuse-and-reuse"
- "keep reusable data in local memory across outer loop iterations"
- "fuse dependent operations into a single loop to avoid HBM round-trips"
- "use the streaming softmax with running max and scaling trick"
- "delay division until after all reductions are complete"
- "Use `scratch_shapes=[pltpu.VMEM(...)]` as persistent VMEM accumulator to avoid repeated HBM read-modify-write"

{self._context_prefix}RULES:
- Most strategies should be general optimization ideas (under 15 words)
- When a specific API or function is the key enabler of an optimization, it is okay to name it (up to 25 words)
- Identify both general ideas and API-specific ones, whichever is more important
- Write as a short action phrase, not a full sentence
- ONLY optimizations within the agent's scope as described above
- NOT optimizations outside that scope, even if mentioned in the documentation
- Do NOT repeat the generic strategies above

Return each strategy on its own line, prefixed with "- ".

Performance optimization strategies:"""

        responses = self._chat(prompt=prompt, num_samples=1, temperature=0)
        return responses[0].strip() if responses else ""

    @staticmethod
    def _parse_bullet_lines(raw: str) -> list[str]:
        """Parse bullet-pointed optimization lines from LLM output."""
        results: list[str] = []
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
            if line and len(line) > 10 and line.lower() != "none":
                results.append(line)
        return results

    # ------------------------------------------------------------------
    # Translate menu
    # ------------------------------------------------------------------

    _DEFAULT_TRANSLATE_MENU = [
        "Convert high-level code to hardware-specific kernel code",
        "Convert a small amount of high-level code to hardware-specific kernel code",
    ]

    def _synthesize_translate_menu(
        self,
        items: list[tuple[str, str]],
        architecture: str = "",
        isa_docs: str = "",
        code_examples_raw: list[tuple[str, str]] | None = None,
    ) -> list[str]:
        """Synthesize code translation strategies from architecture, ISA, and examples.

        Translation strategies guide the LLM in converting standard-library code
        (e.g. NumPy, PyTorch, vanilla JAX) into hardware-specific kernel code
        (e.g. CUDA kernels, NKI kernels, Pallas kernels).  They are generic across
        hardware targets -- the prompt infers the right patterns from the docs.
        """
        defaults = list[str](self._DEFAULT_TRANSLATE_MENU)
        context_parts: list[str] = []

        if architecture:
            context_parts.append(f"=== ARCHITECTURE ===\n{architecture}\n")
        if isa_docs:
            context_parts.append(f"=== ISA / API REFERENCE ===\n{isa_docs}\n")
        if code_examples_raw:
            context_parts.append(
                f"=== CODE EXAMPLES ===\n{_items_to_text(code_examples_raw, max_chars=self.context_budget)}\n"
            )

        if not context_parts:
            return defaults

        context = "\n".join(context_parts)
        defaults_text = chr(10).join("- " + d for d in defaults)

        prompt = f"""{context}

{self._context_prefix}The agent translates standard-library code (e.g. NumPy, PyTorch, JAX, \
or other high-level frameworks) into optimized hardware-specific kernel code for the target \
described above.

These generic strategies are already included:
{defaults_text}

Generate 3-5 ADDITIONAL translation strategies. Each strategy should describe a pattern for converting \
high-level code into the target hardware's kernel API, referencing approaches specific to the target hardware.

RULES:
- Each strategy should be a concise action phrase (1 sentence)
- Focus on HOW to map the computation to the hardware primitives shown in the docs
- Reference specific API calls, memory spaces, or tiling constructs when they are central
- Generic strategies are allowed if they add value beyond the generic strategies already included

Return each strategy on its own line, prefixed with "- ".

Translation strategies:"""

        responses = self._chat(prompt=prompt, num_samples=1, temperature=0)
        raw = responses[0].strip() if responses else ""
        strategies = self._parse_bullet_lines(raw)
        if not strategies:
            logger.warning("Translate menu: LLM returned no strategies")
        return defaults + strategies

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _synthesize_rules(self, items: list[tuple[str, str]],
                          architecture: str = "", isa_docs: str = "") -> dict[str, list[str]]:
        """Extract rules and constraints from routed documentation via map-reduce."""
        base_rules: dict[str, list[str]] = {
            "general": [
                "The rewritten program should be semantically equivalent to the original program, within a small numerical tolerance.",
                "Keep the same function name and signature as the original program (helper functions can be renamed or deleted).",
            ],
            "planning": [
                "Limit the scope of the plan to the selected strategy.",
                "Do not count out any of the strategies unless they are clearly irrelevant to the code.",
            ],
            "coding": [
                "Wrap the generated code with ```python at the beginning and ``` at the end.",
            ],
        }

        if not items:
            return base_rules

        extra_context = self._rules_extra_context(architecture, isa_docs)

        items = _chunk_items(items, self.context_budget)
        total_chars = sum(len(t) for _, t in items)
        if total_chars <= self.context_budget:
            return self._rules_single_pass(items, base_rules, extra_context)

        return self._rules_map_reduce(items, base_rules, extra_context)

    def _rules_extra_context(self, architecture: str, isa_docs: str) -> str:
        context_parts: list[str] = []
        if architecture:
            context_parts.append(f"=== ARCHITECTURE SUMMARY ===\n{architecture}\n=== END ===\n")
        if isa_docs:
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
        return "\n".join(context_parts)

    _RULES_INSTRUCTION = (
        "Extract critical rules that the agent MUST follow when generating optimized kernel code. Focus on:\n"
        "- Programming model constraints (e.g., \"the partition dimension is always 128\")\n"
        "- Memory layout rules (e.g., \"tiles in scratchpad must not exceed X KB\")\n"
        "- Correctness constraints (e.g., \"loop variables cannot be used for indexing in this context\")\n"
        "- Known/common API usage pitfalls (e.g., \"reduction output must be stored before reuse\")\n"
        "Only include rules that are critical to generating CORRECT code. Do NOT include optimization tips, "
        "performance advice, or general best practices -- those belong in the optimization menu."
    )

    def _rules_single_pass(self, items: list[tuple[str, str]],
                           base_rules: dict[str, list[str]], extra_context: str) -> dict[str, list[str]]:
        content = _items_to_text(items, max_chars=self.context_budget)
        raw = self._rules_reduce(content, extra_context, is_reduce=False)
        return self._merge_rules(base_rules, raw)

    def _rules_map_reduce(self, items: list[tuple[str, str]],
                          base_rules: dict[str, list[str]], extra_context: str) -> dict[str, list[str]]:
        logger.info("Rules: map - extract rules from %d documents", len(items))
        prompts: list[str] = []
        for key, text in items:
            truncated = _truncate(text, max_chars=self.context_budget)
            prompts.append(
                f"Source: {key}\n\n"
                f"=== DOCUMENT ===\n{truncated}\n=== END ===\n\n"
                f"{self._context_prefix}"
                f"{self._RULES_INSTRUCTION}\n\n"
                f"Return each rule on its own line, prefixed with \"- \". "
                f"If no rules are found, return \"- none\".\n\n"
                f"Rules:"
            )

        all_responses = [
            r[0] if r else "" for r in
            self.llm.chat_async(
                prompts, num_samples=1, temperature=0,
            )
        ]

        candidates: list[str] = []
        for i, resp in enumerate(all_responses):
            if not resp:
                continue
            lines = self._parse_bullet_lines(resp)
            if lines:
                logger.info("  %s: %d rules", items[i][0], len(lines))
            candidates.extend(lines)

        logger.info("Rules: reduce - merge %d candidates into final list", len(candidates))
        if not candidates:
            return base_rules

        candidate_text = "\n".join(f"- {c}" for c in candidates)
        candidate_text = _truncate(candidate_text, max_chars=self.context_budget)

        raw = self._rules_reduce(candidate_text, extra_context, is_reduce=True)
        return self._merge_rules(base_rules, raw)

    def _rules_reduce(self, content: str, extra_context: str, is_reduce: bool = False) -> str:
        if is_reduce:
            header = ("Below are candidate correctness rules extracted from "
                      "multiple documents about the hardware target's programming model.")
            content_label = "CANDIDATE RULES"
            task = ("Merge, deduplicate, and curate these into a final list of critical "
                    "correctness rules. Return AT MOST 5 rules per category -- pick the most critical ones only.")
        else:
            header = "Below is documentation about the hardware target's programming model."
            content_label = "DOCUMENTATION"
            task = ("Return AT MOST 5 rules per category -- pick the most critical ones only.")

        prompt = f"""{header}

=== {content_label} ===
{content}
=== END {content_label} ===

{extra_context}

{self._context_prefix}{self._RULES_INSTRUCTION}
{task}

Categorize each rule into one of:
- "general" -- applies to both planning and coding phases
- "planning" -- applies only when generating plans
- "coding" -- applies only when generating code

Return as a JSON object with keys "general", "planning", "coding", each mapping to a list of rule strings.

Return ONLY a JSON object:"""

        responses = self._chat(prompt=prompt, num_samples=1, temperature=0)
        return responses[0].strip() if responses else "{}"

    @staticmethod
    def _merge_rules(base_rules: dict[str, list[str]], raw: str) -> dict[str, list[str]]:
        merged = {k: list(v) for k, v in base_rules.items()}
        try:
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                extracted = json.loads(json_match.group())
                for key in ("general", "planning", "coding"):
                    if key in extracted and isinstance(extracted[key], list):
                        merged[key].extend(extracted[key])
                        logger.info("  Rules extracted (%s): %d new rules", key, len(extracted[key]))
        except json.JSONDecodeError:
            logger.warning("Failed to parse rules JSON, using defaults only")
        return merged


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


def _chunk_items(items: list[tuple[str, str]], budget: int) -> list[tuple[str, str]]:
    """Split items that exceed *budget* chars into smaller chunks.

    Chunks are split on paragraph boundaries (double newline) when possible,
    falling back to single newlines.  Each chunk is labelled
    ``"<key> [part N/M]"`` so the LLM knows it is seeing a fragment.
    """
    result: list[tuple[str, str]] = []
    for key, text in items:
        if len(text) <= budget:
            result.append((key, text))
            continue

        chunks: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= budget:
                chunks.append(remaining)
                break
            # Try to split on a paragraph boundary
            cut = remaining.rfind("\n\n", 0, budget)
            if cut < budget // 2:
                cut = remaining.rfind("\n", 0, budget)
            if cut < budget // 2:
                cut = budget
            chunks.append(remaining[:cut])
            remaining = remaining[cut:].lstrip("\n")

        total = len(chunks)
        logger.info("Splitting '%s' (%d chars) into %d chunks", key, len(text), total)
        for i, chunk in enumerate(chunks, 1):
            result.append((f"{key} [part {i}/{total}]", chunk))
    return result
