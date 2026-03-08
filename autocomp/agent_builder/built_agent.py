"""
Runtime agent that loads its prompt components from config files
produced by the AgentBuilder.
"""

import random
import re
import yaml
from pathlib import Path

from autocomp.common import logger, LLMClient
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

    def __init__(self, model: str, config_dir: str | Path,
                 hw_config: HardwareConfig, eval_backend: EvalBackend):
        super().__init__(model)
        self.hw_config = hw_config
        self.eval_backend = eval_backend
        self.config_dir = Path(config_dir)
        self.menu_strategy = "static" # choose from: [static, one-shot, progressive]

        # Load all config files
        self._architecture = self._load_text("architecture.md")
        self._isa_docs_raw = self._load_text("isa_docs.md")
        self._isa_sections = self._parse_isa_sections(self._isa_docs_raw)
        self._optimization_menu = self._load_optimization_menu()
        self._rules = self._load_rules()

        # Cache for ISA section selection per problem
        self._isa_selection_cache: dict[str, list[str]] = {}

        # Cache for new menu options per candidate
        self._new_menu_cache: dict[str, list[str]] = {}

        logger.info(
            "BuiltLLMAgent loaded from %s: %d ISA sections, %d optimizations",
            self.config_dir, len(self._isa_sections), len(self._optimization_menu),
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
            return ["Other methods not listed here."]
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        items = data.get("optimizations", [])
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

    @staticmethod
    def _parse_isa_sections(isa_text: str) -> dict[str, str]:
        """Parse ISA docs markdown into sections keyed by header."""
        sections: dict[str, str] = {}
        current_header = "General"
        current_lines: list[str] = []

        for line in isa_text.split("\n"):
            if line.startswith("## "):
                # Save previous section
                if current_lines:
                    sections[current_header] = "\n".join(current_lines).strip()
                current_header = line[3:].strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections[current_header] = "\n".join(current_lines).strip()

        return sections

    def _get_isa_for_problem(self, prob: Prob, code: str) -> str:
        """
        Select relevant ISA sections for a problem.

        On first call for a given problem, makes a lightweight LLM call to
        select relevant sections. Result is cached for subsequent calls.
        """
        cache_key = f"{prob.prob_type}:{prob.prob_id}"

        if cache_key in self._isa_selection_cache:
            selected = self._isa_selection_cache[cache_key]
            return self._assemble_isa_sections(selected)

        # If ISA is small enough, include everything
        if len(self._isa_docs_raw) < 30_000:
            self._isa_selection_cache[cache_key] = list(self._isa_sections.keys())
            return self._isa_docs_raw

        # Ask the LLM to select relevant sections
        section_list = "\n".join(f"- {name}" for name in self._isa_sections.keys())
        prob_context = getattr(prob, "context", "")

        prompt = f"""Given the following code and problem context, select which ISA documentation sections are relevant. Return ONLY the section names, one per line.

Problem type: {prob.prob_type}
{f'Problem context: {prob_context}' if prob_context else ''}

Code:
```
{code}
```

Available ISA sections:
{section_list}

Relevant sections (one per line):"""

        try:
            responses = self.llm_client.chat(prompt=prompt, num_candidates=1, temperature=0)
            raw = responses[0] if responses else ""
            selected = []
            for line in raw.strip().split("\n"):
                line = line.strip().lstrip("- ").strip()
                if line in self._isa_sections:
                    selected.append(line)
                else:
                    # Fuzzy match
                    for name in self._isa_sections:
                        if line.lower() in name.lower() or name.lower() in line.lower():
                            selected.append(name)
                            break
            if not selected:
                selected = list(self._isa_sections.keys())
        except Exception:
            selected = list(self._isa_sections.keys())

        self._isa_selection_cache[cache_key] = selected
        logger.info("ISA selection for %s: %d/%d sections", cache_key, len(selected), len(self._isa_sections))
        return self._assemble_isa_sections(selected)

    def _assemble_isa_sections(self, section_names: list[str]) -> str:
        parts = []
        for name in section_names:
            if name in self._isa_sections:
                parts.append(self._isa_sections[name])
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def get_opt_menu_options(self, prob: Prob, code: str = None) -> list[str]:
        base = list(self._optimization_menu)
        if code is not None:
            base = base + self._new_menu_cache.get(code, [])
        return base

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

    def _get_propose_optimizations_prompt(
        self, candidate: CodeCandidate, prob: Prob,
        force_opt_menu, prompt_end, analysis, shuffle_opts,
        give_score_feedback, give_util_feedback, give_hw_feedback,
        include_ancestors, plan_icl_examples, cur_iter, num_iters,
        dropout_menu_options, translate,
    ) -> str:
        # Select menu options
        opt_lst = self.get_opt_menu_options(prob, code=candidate.code)
        if dropout_menu_options < 1 and not force_opt_menu:
            opt_lst = [opt for opt in opt_lst if random.random() < dropout_menu_options]
        if shuffle_opts:
            random.shuffle(opt_lst)
        include_score_feedback = random.random() < give_score_feedback
        include_hw_feedback_flag = random.random() < give_hw_feedback

        # Build parent history
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

        # Build prompt: architecture + ISA + code history + menu + rules
        prompt_text = self._architecture + "\n"
        prompt_text += self._get_isa_for_problem(prob, candidate.code) + "\n"
        prompt_text += parents_prompt

        # Optimization menu
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
            if choose_or_invent < 0.1 and not translate:
                prompt_text += "Invent a new optimization inspired by the <optimizations> to apply to the above code to reduce execution time, and explain how it will improve performance."
            elif choose_or_invent < 0.2 and not translate:
                prompt_text += "Think of a new optimization different from the <optimizations> to apply to the above code to reduce execution time, and explain how it will improve performance."
            else:
                prompt_text += "Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its execution time."

        prompt_text += " The plan should be specific to this code and explain how to change it."
        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=True, coding=False, prob=prob)

        if prompt_end:
            prompt_text += "\n" + prompt_end
        return prompt_text

    def _get_propose_new_menu_prompt(self, candidate: CodeCandidate, prob: Prob):
        prompt_text = self._architecture + "\n"
        prompt_text += self._get_isa_for_problem(prob, candidate.code) + "\n"
        prompt_text += "Here is the kernel to optimize:\n"
        prompt_text += candidate.code + "\n"
        prompt_text += "The following optimization strategies are already in the menu:\n"
        for i, opt in enumerate(self.get_opt_menu_options(prob)):
            prompt_text += f"{i + 1}. {opt}\n"

        if self.menu_strategy == "one-shot":
            prompt_text += "You are an expert performance engineer generating high-performance code for this hardware target. "
            prompt_text += "Identify optimization opportunities specific to this kernel and hardware that are NOT already listed above. "
            prompt_text += "Return a list of new optimization strategies, one per line, that could improve this kernel's performance. "
        elif self.menu_strategy == "progressive":
            #TODO
            pass

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
