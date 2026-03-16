"""
Runtime agent that loads its prompt components from config files
produced by the AgentBuilder.
"""

import random
import yaml
from pathlib import Path

from autocomp.common import logger
from autocomp.agents.llm_agent import LLMAgent
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate
from autocomp.hw_config.hardware_config import HardwareConfig
from autocomp.backend.eval_backend import EvalBackend


class BuiltLLMAgent(LLMAgent):
    """
    An LLMAgent subclass that loads its prompt components from an agent
    config directory (produced by AgentBuilder) instead of hardcoding them.
    """

    _DEFAULT_TRANSLATE_MENU = [
        "convert high-level code to target kernel code",
    ]

    def __init__(self, model: str, config_dir: str | Path,
                 hw_config: HardwareConfig, eval_backend: EvalBackend,
                 menu_strategy: str = "static",
                 fine_grained_isa: bool = False,
                 give_examples_feedback: float = 0.0):
        super().__init__(model)
        self.hw_config = hw_config
        self.eval_backend = eval_backend
        self.config_dir = Path(config_dir)
        self.menu_strategy = menu_strategy # choose from: [static, one-shot]
        self.fine_grained_isa = fine_grained_isa
        self.give_examples_feedback = give_examples_feedback

        # Load all config files
        self._architecture = self._load_text("architecture.md")
        self._isa_docs_raw = self._load_text("isa_docs.md")
        self._isa_sections = self._parse_isa_sections(self._isa_docs_raw)
        self._isa_subsections = self._parse_isa_subsections(self._isa_sections)
        self._optimization_menu = self._load_optimization_menu()
        self._translate_menu = self._load_translate_menu()
        self._rules = self._load_rules()
        self._code_example_sections = self._load_code_example_sections()

        # Cache for ISA selection per problem (stores final assembled text)
        self._isa_selection_cache: dict[str, str] = {}

        # Cache for code example selection per problem (stores list of section names)
        self._code_example_cache: dict[str, list[str]] = {}

        # Cache for new menu options per candidate
        self._new_menu_cache: dict[str, list[str]] = {}
        self._translate_menu_warned = False

        logger.info(
            "BuiltLLMAgent loaded from %s: %d ISA sections, %d optimizations, "
            "%d translate strategies, %d code examples, fine_grained_isa=%s, "
            "give_examples_feedback=%.2f",
            self.config_dir, len(self._isa_sections), len(self._optimization_menu),
            len(self._translate_menu), len(self._code_example_sections),
            self.fine_grained_isa, self.give_examples_feedback,
        )

    def __repr__(self):
        return f"BuiltLLMAgent({self.llm_client.model}, {self.config_dir.name})"

    def _load_text(self, filename: str) -> str:
        path = self.config_dir / filename
        if path.exists():
            return path.read_text()
        logger.warning("Config file not found: %s", path)
        return ""

    def _load_optimization_menu(self) -> list[str]:
        path = self.config_dir / "optimization_menu.yaml"
        if not path.exists():
            return []
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        items = data.get("optimizations", [])
        return [item["strategy"] if isinstance(item, dict) else str(item) for item in items]

    def _load_translate_menu(self) -> list[str]:
        path = self.config_dir / "translate_menu.yaml"
        if not path.exists():
            return []
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        items = data.get("strategies", [])
        return [item["strategy"] if isinstance(item, dict) else str(item) for item in items]

    def _load_rules(self) -> dict[str, list[str]]:
        path = self.config_dir / "rules.yaml"
        if not path.exists():
            return {"general": [], "planning": [], "coding": []}
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return {
            "general": data.get("general", []),
            "planning": data.get("planning", []),
            "coding": data.get("coding", []),
        }

    def _load_code_example_sections(self) -> list[tuple[str, str, str]]:
        """Parse code_examples.md into (name, summary, code_body) tuples.

        Expected format per section:
            ## <filename>
            SUMMARY: <1-2 sentence description>
            <code blocks>
        Falls back to "(no summary available)" if no SUMMARY: line is found.
        """
        raw = self._load_text("code_examples.md")
        if not raw.strip():
            return []

        sections: list[tuple[str, str, str]] = []
        current_name: str | None = None
        current_lines: list[str] = []

        def _flush():
            if current_name is None:
                return
            body = "\n".join(current_lines).strip()
            summary = "(no summary available)"
            code_body = body
            for i, line in enumerate(current_lines):
                if line.strip().startswith("SUMMARY:"):
                    summary = line.strip()[len("SUMMARY:"):].strip()
                    code_body = "\n".join(current_lines[i + 1:]).strip()
                    break
            if code_body:
                sections.append((current_name, summary, code_body))

        for line in raw.split("\n"):
            if line.startswith("## "):
                _flush()
                current_name = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)

        _flush()
        return sections

    @staticmethod
    def _parse_isa_sections(isa_text: str) -> dict[str, str]:
        """Parse ISA docs markdown into sections keyed by ## header."""
        sections: dict[str, str] = {}
        current_header = "General"
        current_lines: list[str] = []

        for line in isa_text.split("\n"):
            if line.startswith("## "):
                if current_lines:
                    sections[current_header] = "\n".join(current_lines).strip()
                current_header = line[3:].strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections[current_header] = "\n".join(current_lines).strip()

        return sections

    @staticmethod
    def _parse_isa_subsections(sections: dict[str, str]) -> dict[str, dict[str, str]]:
        """Parse ### subsections within each ## section.

        Returns {section_name: {subsection_name: content}}.
        """
        result: dict[str, dict[str, str]] = {}
        for sec_name, sec_text in sections.items():
            subs: dict[str, str] = {}
            cur_sub: str | None = None
            cur_lines: list[str] = []
            preamble_lines: list[str] = []

            for line in sec_text.split("\n"):
                if line.startswith("### "):
                    if cur_sub is not None:
                        block = "\n".join(cur_lines).strip()
                        if cur_sub in subs:
                            subs[cur_sub] += "\n\n" + block
                        else:
                            subs[cur_sub] = block
                    cur_sub = line[4:].strip()
                    cur_lines = [line]
                elif cur_sub is not None:
                    cur_lines.append(line)
                else:
                    preamble_lines.append(line)

            if cur_sub is not None:
                block = "\n".join(cur_lines).strip()
                if cur_sub in subs:
                    subs[cur_sub] += "\n\n" + block
                else:
                    subs[cur_sub] = block
            if preamble_lines:
                preamble = "\n".join(preamble_lines).strip()
                if preamble:
                    subs["_preamble"] = preamble

            result[sec_name] = subs
        return result

    @staticmethod
    def _subsection_summary(content: str, max_chars: int = 300) -> str:
        """Extract a brief description from a ### subsection's content."""
        lines = content.split("\n")
        body = []
        for line in lines:
            if line.startswith("### "):
                continue
            stripped = line.strip()
            if not stripped or stripped == "---":
                continue
            if stripped.startswith("*"):
                continue
            body.append(stripped)
            if len(" ".join(body)) >= max_chars:
                break
        summary = " ".join(body)
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "…"
        return summary

    def _get_isa_for_problem(self, prob: Prob, code: str) -> str:
        """Select relevant ISA sections (and optionally subsections) for a problem.

        Two-level filtering:
        - L1: parallel yes/no per ## section
        - Fine-grained mode additionally runs L2 on large sections that pass L1
        """
        cache_key = f"{prob.prob_type}:{prob.prob_id}"

        if cache_key in self._isa_selection_cache:
            return self._isa_selection_cache[cache_key]

        if len(self._isa_docs_raw) < 30_000:
            self._isa_selection_cache[cache_key] = self._isa_docs_raw
            return self._isa_docs_raw

        # L1: per-section yes/no
        selected_sections = self._select_sections(prob, code)
        logger.info(
            "%s BuiltLLMAgent: ISA L1 for %s: %d/%d sections selected",
            self.llm_client.model, cache_key,
            len(selected_sections), len(self._isa_sections),
        )
        logger.debug(
            "%s BuiltLLMAgent: Selected ISA sections: %s",
            self.llm_client.model, selected_sections,
        )

        if not self.fine_grained_isa:
            text = self._assemble_isa_sections(selected_sections)
            self._isa_selection_cache[cache_key] = text
            return text

        # Fine-grained: for sections that passed L1, include small sections
        # in full and run L2 subsection filtering on large sections.
        parts: list[str] = []
        sections_to_filter: list[str] = []
        for name in selected_sections:
            subs = self._isa_subsections.get(name, {})
            real_subs = {k: v for k, v in subs.items() if k != "_preamble"}
            if real_subs:
                sections_to_filter.append(name)
            else:
                parts.append(self._isa_sections[name])

        if sections_to_filter:
            filtered = self._select_subsections(prob, code, sections_to_filter)
            for sec_name, kept_subs in filtered.items():
                subs = self._isa_subsections[sec_name]
                sec_parts: list[str] = []
                if "_preamble" in subs:
                    sec_parts.append(subs["_preamble"])
                for sub_name in kept_subs:
                    if sub_name in subs:
                        sec_parts.append(subs[sub_name])
                if sec_parts:
                    parts.append("\n\n".join(sec_parts))

        text = "\n\n".join(parts)
        self._isa_selection_cache[cache_key] = text
        logger.info(
            "%s BuiltLLMAgent: Fine-grained ISA for %s: %d chars "
            "(%d small sections included, %d sections filtered at L2)",
            self.llm_client.model, cache_key, len(text),
            len(selected_sections) - len(sections_to_filter),
            len(sections_to_filter),
        )
        return text

    def _select_sections(self, prob: Prob, code: str) -> list[str]:
        """Level-1: parallel yes/no per ## section."""
        prob_context = getattr(prob, "context", "")
        code_block = f"Code:\n```\n{code}\n```"

        section_names = list(self._isa_sections.keys())
        prompts: list[str] = []
        for name in section_names:
            subs = self._isa_subsections.get(name, {})
            # Build a content preview: preamble + subsection summaries
            preview_parts: list[str] = [f"## {name}"]
            preamble = subs.get("_preamble", "")
            if preamble:
                preamble_text = preamble[:500]
                if len(preamble) > 500:
                    preamble_text += "…"
                preview_parts.append(preamble_text)
            for sub_name, content in subs.items():
                if sub_name == "_preamble":
                    continue
                summary = self._subsection_summary(content)
                if summary:
                    preview_parts.append(f"- {sub_name}: {summary}")
                else:
                    preview_parts.append(f"- {sub_name}")
            preview = "\n".join(preview_parts)

            prompt = (
                f"Problem type: {prob.prob_type}\n"
                f"{f'Problem context: {prob_context}' if prob_context else ''}\n\n"
                f"{code_block}\n\n"
                f"ISA documentation section:\n{preview}\n\n"
                "Could this documentation section be relevant for understanding, "
                "implementing, or optimizing this code? Answer Yes or No."
            )
            prompts.append(prompt)

        all_responses = self.llm_client.chat_async(
            prompts, num_candidates=1, temperature=0,
        )

        selected: list[str] = []
        for name, responses in zip(section_names, all_responses):
            answer = responses[0].strip().lower() if responses and responses[0] else ""
            is_yes = "yes" in answer.split()[:5] or answer.startswith("yes")
            logger.debug(
                "%s BuiltLLMAgent: L1 %s -> %s (%s)",
                self.llm_client.model, name,
                "YES" if is_yes else "NO",
                answer[:80],
            )
            if is_yes:
                selected.append(name)

        # If nothing was selected, fall back to including everything
        return selected if selected else section_names

    def _select_subsections(
        self, prob: Prob, code: str, section_names: list[str],
    ) -> dict[str, list[str]]:
        """Level-2: parallel yes/no per ### subsection within selected sections."""
        prob_context = getattr(prob, "context", "")
        code_block = f"Code:\n```\n{code}\n```"

        all_sub_names: list[str] = []
        sec_for_sub: list[str] = []
        prompts: list[str] = []

        for sec_name in section_names:
            for sub_name, content in self._isa_subsections[sec_name].items():
                if sub_name == "_preamble":
                    continue
                all_sub_names.append(sub_name)
                sec_for_sub.append(sec_name)

                summary = self._subsection_summary(content)
                prompt = (
                    f"Problem type: {prob.prob_type}\n"
                    f"{f'Problem context: {prob_context}' if prob_context else ''}\n\n"
                    f"{code_block}\n\n"
                    f"API documentation (from section \"{sec_name}\"):\n"
                    f"### {sub_name}\n{summary}\n\n"
                    "Could this API be relevant for understanding, implementing, "
                    "or optimizing this code? Answer Yes or No."
                )
                prompts.append(prompt)

        if not prompts:
            return {name: [] for name in section_names}

        all_responses = self.llm_client.chat_async(
            prompts, num_candidates=1, temperature=0,
        )

        result: dict[str, list[str]] = {name: [] for name in section_names}
        for sub_name, sec_name, responses in zip(
            all_sub_names, sec_for_sub, all_responses,
        ):
            answer = responses[0].strip().lower() if responses and responses[0] else ""
            is_yes = "yes" in answer.split()[:5] or answer.startswith("yes")
            if is_yes:
                result[sec_name].append(sub_name)

        for sec_name in section_names:
            total = len([k for k in self._isa_subsections[sec_name] if k != "_preamble"])
            logger.debug(
                "%s BuiltLLMAgent: Fine-grained %s: %d/%d subsections: %s",
                self.llm_client.model, sec_name,
                len(result[sec_name]), total, result[sec_name],
            )
        return result

    def _assemble_isa_sections(self, section_names: list[str]) -> str:
        parts = []
        for name in section_names:
            if name in self._isa_sections:
                parts.append(self._isa_sections[name])
        return "\n\n".join(parts)

    def _select_code_examples(self, prob: Prob, code: str, isa_text: str = "") -> list[str]:
        """LLM-select relevant code example sections using parallel yes/no prompts.

        Returns a list of section names. Results are cached per problem.
        """
        cache_key = f"{prob.prob_type}:{prob.prob_id}"
        if cache_key in self._code_example_cache:
            return self._code_example_cache[cache_key]

        if not self._code_example_sections:
            self._code_example_cache[cache_key] = []
            return []

        prob_context = getattr(prob, "context", "")
        code_block = f"Code:\n```\n{code}\n```"
        isa_hint = ""
        if isa_text:
            headers = [ln for ln in isa_text.splitlines() if ln.startswith("## ") or ln.startswith("### ")]
            if headers:
                isa_hint = "Selected ISA APIs:\n" + "\n".join(headers) + "\n\n"

        names: list[str] = []
        prompts: list[str] = []
        for name, summary, _ in self._code_example_sections:
            names.append(name)
            prompt = (
                f"{isa_hint}"
                f"Problem type: {prob.prob_type}\n"
                f"{f'Problem context: {prob_context}' if prob_context else ''}\n\n"
                f"{code_block}\n\n"
                f"Code example: {name}\nSummary: {summary}\n\n"
                "Is this code example relevant for learning optimization patterns "
                "applicable to the code above? Answer Yes or No."
            )
            prompts.append(prompt)

        all_responses = self.llm_client.chat_async(
            prompts, num_candidates=1, temperature=0,
        )

        selected: list[str] = []
        for name, responses in zip(names, all_responses):
            answer = responses[0].strip().lower() if responses and responses[0] else ""
            is_yes = "yes" in answer.split()[:5] or answer.startswith("yes")
            if is_yes:
                selected.append(name)

        self._code_example_cache[cache_key] = selected
        logger.info(
            "%s BuiltLLMAgent: code example selection for %s: %d/%d examples",
            self.llm_client.model, cache_key, len(selected), len(names),
        )
        logger.debug(
            "%s BuiltLLMAgent: Selected code examples: %s",
            self.llm_client.model, selected,
        )
        return selected

    def _get_code_example_bodies(self, names: list[str]) -> dict[str, str]:
        """Return {name: code_body} for the given section names."""
        lookup = {name: body for name, _, body in self._code_example_sections}
        return {n: lookup[n] for n in names if n in lookup}

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def get_opt_menu_options(self, prob: Prob, candidate: CodeCandidate = None) -> list[str]:
        opt_lst = list(self._optimization_menu)
        if candidate is not None:
            opt_lst = opt_lst + self._new_menu_cache.get(candidate.code, [])
        opt_lst.append("Other methods not listed here.")
        return opt_lst

    def update_new_menu_cache(self, new_menu: dict[str, list[str]]):
        self._new_menu_cache = new_menu

    def _get_prompt_rules(self, planning: bool, coding: bool, prob: Prob = None) -> str:
        rules: list[str] = []
        rules.extend(self.hw_config.get_hw_config_specific_rules())
        rules.extend(self.eval_backend.get_backend_specific_rules())
        rules.extend(self._rules.get("general", []))

        if planning:
            rules.extend(self._rules.get("planning", []))
        if coding:
            rules.extend(self._rules.get("coding", []))

        # Include problem-specific context if available
        if prob and hasattr(prob, "context") and prob.context:
            rules.append(prob.context)

        prompt_text = ""
        for i, rule in enumerate(rules):
            prompt_text += f"{i + 1}. {rule}\n"
        return prompt_text

    def _build_prompt_scaffold(
        self, candidate: CodeCandidate, prob: Prob, analysis,
        give_score_feedback, give_hw_feedback, include_ancestors,
    ) -> str:
        """Build the shared prefix: architecture + ISA + parent history."""
        include_score_feedback = random.random() < give_score_feedback
        include_hw_feedback_flag = random.random() < give_hw_feedback

        parents_prompt = ""
        cur_cand = candidate
        while cur_cand is not None:
            if include_score_feedback and cur_cand.score is not None:
                parents_prompt = f"The latency of this code was {cur_cand.score}.\n" + parents_prompt
            if include_hw_feedback_flag and hasattr(cur_cand, "hw_feedback") and cur_cand.hw_feedback:
                parents_prompt = "\n".join(cur_cand.hw_feedback) + "\n" + parents_prompt
            if not include_ancestors:
                parents_prompt = "\nThe original unoptimized code was:\n```\n" + cur_cand.code + "\n```\n" + parents_prompt
                break
            elif cur_cand.plan is not None:
                parents_prompt = (
                    "\nNext, we applied this plan to the code:\n" + cur_cand.plan
                    + "\nThe generated code was:\n" + cur_cand.code + "\n" + parents_prompt
                )
            else:
                parents_prompt = "\nThe original unoptimized code was:\n```\n" + cur_cand.code + "\n```\n" + parents_prompt
            cur_cand = cur_cand.parent

        if analysis:
            parents_prompt += "\n" + analysis

        prompt_text = self._architecture + "\n"
        prompt_text += self._get_isa_for_problem(prob, candidate.code) + "\n"
        prompt_text += parents_prompt
        return prompt_text

    def _get_propose_optimizations_prompt(
        self, candidate: CodeCandidate, prob: Prob,
        force_opt_menu, prompt_end, analysis, shuffle_opts,
        give_score_feedback, give_util_feedback, give_hw_feedback,
        include_ancestors, plan_icl_examples, cur_iter, num_iters,
        dropout_menu_options, translate,
    ) -> str:
        if translate:
            return self._get_translate_prompt(
                candidate, prob, prompt_end, analysis, shuffle_opts,
                give_score_feedback, give_hw_feedback, include_ancestors,
            )

        opt_lst = self.get_opt_menu_options(prob, candidate)
        if dropout_menu_options < 1 and not force_opt_menu:
            opt_lst = [opt for opt in opt_lst if random.random() < dropout_menu_options]
        if shuffle_opts:
            random.shuffle(opt_lst)

        # Run ISA selection first (cached) so we can inform code example selection
        isa_text = self._get_isa_for_problem(prob, candidate.code)

        # Stochastically prepend code examples at the top (deprioritized by position)
        examples_prefix = ""
        if self.give_examples_feedback > 0 and self._code_example_sections:
            selected_names = self._select_code_examples(prob, candidate.code, isa_text)
            if selected_names:
                sampled = [n for n in selected_names if random.random() < self.give_examples_feedback]
                if sampled:
                    bodies = self._get_code_example_bodies(sampled)
                    if bodies:
                        parts = [f"### {name}\n{body}" for name, body in bodies.items()]
                        framing = (
                            "Use these reference patterns to inform your optimization plan.\n\n"
                            if random.random() < 0.75 else
                            "Use these reference patterns, but don't copy them unless they are directly applicable to the target code.\n\n"
                        )
                        examples_prefix = (
                            "Reference patterns:\n\n" + "\n\n".join(parts) + "\n\n" + framing
                        )

        prompt_text = examples_prefix + self._build_prompt_scaffold(
            candidate, prob, analysis,
            give_score_feedback, give_hw_feedback, include_ancestors,
        )

        menu_text = ""
        for i, opt in enumerate(opt_lst):
            menu_text += f"{i + 1}. {opt}\n"

        prompt_text += "Please carefully review the code to identify any inefficiencies. "
        prompt_text += "Performance can be improved by using the following optimizations:\n"
        prompt_text += "<optimizations>:\n" + menu_text + "\n"

        if force_opt_menu:
            prompt_text += (
                f"Explain how to apply <optimization> {force_opt_menu}: "
                f"'{opt_lst[force_opt_menu - 1]}' to the above code to reduce "
                "execution time, and explain how it will improve performance."
            )
        else:
            prompt_text += "You are an expert performance engineer generating high-performance code for this hardware target. "
            choose_or_invent = random.random()
            if choose_or_invent < 0.1:
                prompt_text += "Invent a new optimization inspired by the <optimizations> to apply to the above code to reduce execution time, and explain how it will improve performance."
            elif choose_or_invent < 0.2:
                prompt_text += "Think of a new optimization different from the <optimizations> to apply to the above code to reduce execution time, and explain how it will improve performance."
            else:
                prompt_text += "Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its execution time."

        prompt_text += " The plan should be specific to this code and explain how to change it."
        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=True, coding=False, prob=prob)

        if prompt_end:
            prompt_text += "\n" + prompt_end
        return prompt_text

    def _get_translate_prompt(
        self, candidate: CodeCandidate, prob: Prob,
        prompt_end, analysis, shuffle_opts,
        give_score_feedback, give_hw_feedback, include_ancestors,
    ) -> str:
        """Prompt for translation iterations: convert code to the target representation."""
        if self._translate_menu:
            opt_lst = list(self._translate_menu)
        else:
            if not self._translate_menu_warned:
                logger.warning(
                    "translate_iters > 0 but no translate_menu.yaml found in %s. "
                    "Using generic default. Create a translate_menu.yaml with "
                    "target-specific strategies for better results.",
                    self.config_dir,
                )
                self._translate_menu_warned = True
            opt_lst = list(self._DEFAULT_TRANSLATE_MENU)
        if shuffle_opts:
            random.shuffle(opt_lst)

        prompt_text = self._build_prompt_scaffold(
            candidate, prob, analysis,
            give_score_feedback, give_hw_feedback, include_ancestors,
        )

        menu_text = ""
        for i, opt in enumerate(opt_lst):
            menu_text += f"{i + 1}. {opt}\n"

        prompt_text += "Please review the code and identify parts that should be converted to the target hardware representation. "
        prompt_text += "The following conversion strategies are available:\n"
        prompt_text += "<strategies>:\n" + menu_text + "\n"

        prompt_text += (
            "You are an expert at translating code to this hardware target. "
            "Come up with a plan to apply exactly one of the <strategies> to convert "
            "the above code to the target representation."
        )

        prompt_text += " The plan should be specific to this code and explain how to change it."
        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=True, coding=False, prob=prob)

        if prompt_end:
            prompt_text += "\n" + prompt_end
        return prompt_text

    def _get_propose_new_menu_prompt(self, candidate: CodeCandidate, prob: Prob):
        prompt_text = ""

        isa_text = self._get_isa_for_problem(prob, candidate.code)

        # Include code examples to help discover optimization strategies
        if self._code_example_sections:
            selected_names = self._select_code_examples(prob, candidate.code, isa_text)
            if selected_names:
                bodies = self._get_code_example_bodies(selected_names)
                if bodies:
                    parts = [f"### {name}\n{body}" for name, body in bodies.items()]
                    prompt_text += (
                        "Reference patterns showing how key APIs are used:\n\n"
                        + "\n\n".join(parts) + "\n\n"
                    )

        prompt_text += self._architecture + "\n"
        prompt_text += isa_text + "\n"
        prompt_text += "Here is the kernel to optimize:\n"
        prompt_text += candidate.code + "\n"
        prompt_text += "The following optimization strategies are already in the menu:\n"
        for i, opt in enumerate(self.get_opt_menu_options(prob)):
            prompt_text += f"{i + 1}. {opt}\n"

        if self.menu_strategy == "one-shot":
            prompt_text += (
                "You are an expert performance engineer generating high-performance code "
                "for this hardware target. Analyze the kernel code and the high-level operation it performs, and identify about 10 new "
                "optimization strategies that could improve performance and are NOT already "
                "listed above. When presenting your final strategies, list them between <strategies> and </strategies> tags, "
                "one per line prefixed with \"- \". Example:\n\n"
                "<strategies>\n"
                "- strategy 1\n"
                "- strategy 2\n"
                "- ...\n"
                "</strategies>\n"
            )

        return prompt_text

    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None,
                                    code_icl_examples: bool = True) -> str:
        if prob is None:
            raise ValueError("BuiltLLMAgent requires prob parameter to be provided")

        prompt_text = self._architecture + "\n"
        prompt_text += self._get_isa_for_problem(prob, candidate.parent.code) + "\n"
        prompt_text += "The original code is as follows:\n"
        prompt_text += candidate.parent.code
        prompt_text += "\nYou are an expert performance engineer generating high-performance code for this hardware target. "
        prompt_text += "Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan
        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nOptimized code:"

        return prompt_text

    def _get_combine_candidates_prompt(self, candidates: list[CodeCandidate],
                                        prob: Prob = None) -> str:
        prompt_text = self._architecture + "\n"
        prompt_text += "You are an expert performance engineer generating high-performance code for this hardware target. "
        prompt_text += "Let's combine the following optimized code samples to extract the high-performance characteristics of each:\n"
        for i, c in enumerate(candidates):
            prompt_text += f"Sample {i + 1}:\n{c.code}\n"

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nOptimized combined code:"
        return prompt_text

    def _get_reimplement_failed_code_prompt(self, candidate: CodeCandidate,
                                             prob: Prob = None) -> str:
        if prob is None:
            raise ValueError("BuiltLLMAgent requires prob parameter to be provided")

        prompt_text = self._architecture + "\n"
        prompt_text += self._get_isa_for_problem(prob, candidate.code) + "\n"
        prompt_text += "\nYou are an expert performance engineer generating high-performance code for this hardware target. "
        prompt_text += "\nThe code was:\n"
        prompt_text += candidate.code

        prompt_text += "\n\nHowever, the code failed with the following output:\n"
        if candidate.stderr:
            prompt_text += "=== STDERR ===\n"
            stderr_lines = candidate.stderr.split("\n")
            stderr_lines = [line[:400] for line in stderr_lines]
            prompt_text += "\n".join(stderr_lines) + "\n"
        if candidate.stdout:
            prompt_text += "=== STDOUT ===\n"
            stdout_lines = candidate.stdout.split("\n")
            stdout_lines = [line[:400] for line in stdout_lines]
            prompt_text += "\n".join(stdout_lines) + "\n"

        prompt_text += "\nPlease fix the code to address the errors while still applying the optimization plan. "
        prompt_text += "Make sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nFixed and optimized code:"

        return prompt_text
