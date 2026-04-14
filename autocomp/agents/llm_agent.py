import json
import pathlib
import random
import re

from autocomp.common import logger, LLMClient
from autocomp.common.llm_utils import llm_phase
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate, copy_candidate

EDITS_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "code_edits",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "plan": {"type": "string", "description": "Brief reasoning about which optimization to apply and why."},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_str": {"type": "string", "description": "Exact substring to find in the current code. All occurrences are replaced."},
                            "new_str": {"type": "string", "description": "Replacement string."},
                        },
                        "required": ["old_str", "new_str"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["plan", "edits"],
            "additionalProperties": False,
        },
    },
}


def apply_edits(code: str, edits: list[dict]) -> str:
    """Apply a sequence of str_replace edits to code.

    Each edit is {"old_str": ..., "new_str": ...}.
    Replaces all occurrences of old_str. Raises ValueError if old_str is not found.
    """
    for i, edit in enumerate(edits):
        old = edit["old_str"]
        new = edit["new_str"]
        if old == new:
            continue
        if old not in code:
            raise ValueError(f"Edit {i}: old_str not found in code:\n{old[:200]}")
        code = code.replace(old, new)
    return code


def parse_edits_response(response_text: str) -> list[dict] | None:
    """Parse a structured-output JSON response into a list of edit dicts.

    Returns None if parsing fails.
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if isinstance(data, dict) and "edits" in data:
        edits = data["edits"]
        if isinstance(edits, list) and all(
            isinstance(e, dict) and "old_str" in e and "new_str" in e for e in edits
        ):
            return edits
    return None

_FENCED_CODE_RE = re.compile(r"```\w*\n(.*?)```", re.DOTALL)

def extract(code_str: str) -> str:
    """Extract code from an LLM response.

    Tries, in order:
    1. Fenced code blocks (```<optional-lang> ... ```) — returns the *last*
       block, which is typically the final complete implementation.
    2. Gemmini-style ``void solution(...) { ... }`` brace matching.
    3. Fallback: the entire string.
    """
    if not code_str:
        return code_str

    blocks = _FENCED_CODE_RE.findall(code_str)
    if blocks:
        return blocks[-1]

    if "void solution" in code_str:
        from_void_solution_str = code_str[code_str.find("void solution"):]
        open_braces = 0
        in_comment = False
        in_single_line_comment = False
        for i, char in enumerate(from_void_solution_str):
            if from_void_solution_str[i:i+2] == '/*':
                in_comment = True
            elif from_void_solution_str[i:i+2] == '*/':
                in_comment = False
            if from_void_solution_str[i:i+2] == '//':
                in_single_line_comment = True
            elif char == '\n':
                in_single_line_comment = False
            if in_comment or in_single_line_comment:
                continue
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
                if open_braces == 0:
                    end = i
                    break
        try:
            body = from_void_solution_str[:end+1]
        except:
            return from_void_solution_str
        return body

    return code_str

def extract_plan(plan_str: str) -> str:
    """
    Takes LLM-generated plan and extracts the plan
    """
    plan_str = plan_str.split("</think>")[-1].split("</budget:thinking>")[-1].split("</planning>")[-1]
    return plan_str

class LLMAgent:
    """
    A mock-up of the LLM used to propose, evaluate, and implement optimizations.
    """
    menu_strategy = None

    def __init__(self, model_with_provider: str):
        if "::" in model_with_provider:
            provider, model = model_with_provider.split("::", 1)
        else:
            provider = None
            model = model_with_provider
        self.llm_client = LLMClient(model, provider)

    def get_opt_menu_options(self, prob: Prob, candidate: CodeCandidate = None) -> list[str]:
        raise NotImplementedError

    def analyze_code(self, candidate: CodeCandidate, num_to_gen: int, save_dir: pathlib.Path, save_str: str, prob: Prob = None) -> list[str]:
        raise NotImplementedError

    def _get_propose_optimizations_prompt(self, candidate: CodeCandidate, prob: Prob,
                                          force_opt_menu=None, prompt_end="", analysis="",
                                          shuffle_opts=False, give_score_feedback=1.0,
                                          give_util_feedback=0.0, give_hw_feedback=1.0,
                                          include_ancestors=True, plan_icl_examples=False,
                                          cur_iter=None, num_iters=None,
                                          dropout_menu_options=1, translate=False) -> str:
        raise NotImplementedError

    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None,
                                   code_icl_examples: bool = True) -> str:
        raise NotImplementedError

    def _get_direct_implement_prompt(self, candidate: CodeCandidate, prob: Prob,
                                     give_score_feedback: float = 1.0,
                                     give_hw_feedback: float = 1.0,
                                     include_ancestors: bool = False,
                                     dropout_menu_options: float = 1.0,
                                     cur_iter: int = None,
                                     num_iters: int = None,
                                     translate: bool = False) -> str:
        """Build a prompt that generates optimized code directly, without a separate planning phase.

        Subclasses should override to provide hardware-specific prompts.
        When *translate* is True the prompt should ask for conversion to the
        target representation rather than optimization.
        """
        raise NotImplementedError

    def _get_implement_edits_messages(self, candidate: CodeCandidate, prob: Prob = None) -> list[dict]:
        """Return a messages list for the edit-based code generation path.

        Subclasses should override this to provide system/user messages that
        instruct the model to return structured JSON edits.
        """
        raise NotImplementedError

    def _get_direct_implement_edits_messages(self, candidate: CodeCandidate, prob: Prob,
                                             give_score_feedback: float = 1.0,
                                             give_hw_feedback: float = 1.0,
                                             include_ancestors: bool = False,
                                             dropout_menu_options: float = 1.0,
                                             cur_iter: int = None,
                                             num_iters: int = None,
                                             translate: bool = False) -> list[dict]:
        """Return messages for direct (plan-free) edit-based code generation.

        Subclasses should override this to provide system/user messages that
        combine optimization selection with structured JSON edit output.
        """
        raise NotImplementedError

    def _get_combine_candidates_prompt(self, candidates: list[CodeCandidate],
                                        prob: Prob = None) -> str:
        raise NotImplementedError

    def score_translation_completeness(self, original_code: str, candidates: list[CodeCandidate], prob: Prob) -> list[float]:
        """Score how completely each candidate has been translated to target hardware kernels.
        
        Returns a list of scores (0-10) for each candidate.
        """
        raise NotImplementedError

    def _get_propose_new_menu_prompt(self, candidate: CodeCandidate, prob: Prob) -> str:
        raise NotImplementedError

    def evaluate_code_quality(self, candidates: list[CodeCandidate], save_dir: pathlib.Path = None) -> list[float]:
        """
        Evaluate the quality of code candidates using LLM before running benchmarks.
        Returns a list of quality scores (0.0 to 1.0) where higher is better.
        """
        raise NotImplementedError

    def propose_new_menu_parallel(self, prob: Prob, candidates: list[CodeCandidate]) -> dict[str, list[str]]:
        """Generate workload-specific menu additions for each candidate.

        Returns a dict keyed by candidate.code -> list of new menu option strings.
        """
        prompts_lst = [self._get_propose_new_menu_prompt(candidate, prob) for candidate in candidates]

        llm_phase.set("menu_generation")
        responses = self.llm_client.chat_async(
            prompts_lst=prompts_lst,
            num_samples=1,
            temperature=1,
        )

        result = {}
        for candidate, response in zip(candidates, responses):
            raw = response[0] if response else ""
            result[candidate.code] = self._parse_menu_response(raw)
        return result

    @staticmethod
    def _parse_menu_response(raw: str) -> list[str]:
        """Parse a menu response, extracting from <strategies> tags if present."""
        import re, ast
        raw = raw.strip()

        m = re.search(r"<strategies>\s*(.*?)\s*</strategies>", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()

        if raw.startswith("["):
            try:
                items = ast.literal_eval(raw)
                if isinstance(items, list):
                    return [str(item).strip() for item in items if str(item).strip()]
            except (ValueError, SyntaxError):
                pass
        items: list[str] = []
        for line in raw.splitlines():
            line = line.strip().lstrip("- ").lstrip("0123456789.").strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                try:
                    sub = ast.literal_eval(line)
                    if isinstance(sub, list):
                        items.extend(str(s).strip() for s in sub if str(s).strip())
                        continue
                except (ValueError, SyntaxError):
                    pass
            items.append(line)
        return items

    def update_new_menu_cache(self, new_menu: dict[str, list[str]]):
        pass

    @staticmethod
    def _plan_path(save_dir, save_str, force_opt_menu_lst, c_i, s_i):
        path = f"plan{'' if not save_str else '_' + save_str}"
        if force_opt_menu_lst is not None:
            path += "_" + str(force_opt_menu_lst[c_i])
        path += "_" + str(s_i) + ".txt"
        return save_dir / path

    def plans_cached(self, candidate_lst, num_plans, save_dir, save_strs, force_opt_menu_lst=None):
        """Check if all plan files already exist on disk."""
        for c_i in range(len(candidate_lst)):
            for s_i in range(num_plans):
                if not self._plan_path(save_dir, save_strs[c_i], force_opt_menu_lst, c_i, s_i).exists():
                    return False
        return True

    def propose_optimizations_parallel(self, candidate_lst: list[CodeCandidate], num_plans: int, save_dir: pathlib.Path, save_strs: list[str], 
                              prob: Prob,
                              force_opt_menu_lst: int = None, 
                              prompt_end: str = "", 
                              analysis_lst: list[str] = None, 
                              shuffle_opts: bool = False, 
                              give_score_feedback: float = 1.0,
                              give_util_feedback: float = 0.0,
                              give_hw_feedback: float = 1.0,
                              include_ancestors: bool = True,
                              plan_icl_examples: bool = False,
                              cur_iter: int = None,
                              num_iters: int = None,
                              dropout_menu_options: float = 1,
                              translate: bool = False,
                             ) -> list[CodeCandidate]:
        """
        dropout_menu_options: probability of keeping each menu option
        """
        loaded_plans = []
        not_found = False
        for c_i in range(len(candidate_lst)):
            this_cand_loaded_plans = []
            save_str = save_strs[c_i]
            for s_i in range(num_plans):
                path = self._plan_path(save_dir, save_str, force_opt_menu_lst, c_i, s_i)
                if path.exists():
                    with open(path, "r") as f:
                        plan = f.read()
                        logger.debug("Loaded optimization plan from %s", path)
                        this_cand_loaded_plans.append(plan)
                else:
                    not_found = True
                    break
            if not_found:
                break
            loaded_plans.append(this_cand_loaded_plans)
        else:
            loaded_cands = []
            for c_i, this_cand_loaded_plans in enumerate(loaded_plans):
                for plan in this_cand_loaded_plans:
                    loaded_cands.append(CodeCandidate(candidate_lst[c_i], plan, None, plan_gen_model=self.llm_client.model))
            logger.info("Loaded %d optimization plans rather than generating new ones", len(loaded_cands))
            return loaded_cands

        if dropout_menu_options < 1 or (0 < give_score_feedback < 1) or (0 < give_util_feedback < 1) or (0 < give_hw_feedback < 1):
            num_unique_prompts_per_cand = num_plans
        else:
            num_unique_prompts_per_cand = 1
        prompts_lst = []
        for c_i, candidate in enumerate(candidate_lst):
            save_str = save_strs[c_i]
            for p in range(num_unique_prompts_per_cand):
                # Add the previously iterated plans and code to the prompt
                analysis = "" if analysis_lst is None else analysis_lst[c_i]
                force_opt_menu = None if force_opt_menu_lst is None else force_opt_menu_lst[c_i]
                prompt_text = self._get_propose_optimizations_prompt(candidate, prob, force_opt_menu, prompt_end, analysis, shuffle_opts,
                                                                    give_score_feedback, give_util_feedback, give_hw_feedback, include_ancestors, plan_icl_examples, cur_iter, num_iters,
                                                                    dropout_menu_options, translate)

                # Save full prompt
                prompt_path = f"prompt{'' if not save_str else '_' + save_str}"
                if force_opt_menu_lst is not None:
                    prompt_path += "_" + str(force_opt_menu_lst[c_i])
                else:
                    prompt_path += "_" + str(p)
                prompt_path += ".txt"
                prompt_path = save_dir / prompt_path

                with open(prompt_path, "w") as f:
                    f.write(prompt_text)

                prompts_lst.append(prompt_text)

        temperature = 1
        samples_per_prompt = num_plans // num_unique_prompts_per_cand

        llm_phase.set("plan_generation")
        extended_responses = self.llm_client.chat_async(
            prompts_lst=prompts_lst,
            num_samples=samples_per_prompt,
            temperature=temperature
        )
        logger.info("%s: finished generating %d plans for %d candidates.", self.llm_client.model, len(prompts_lst) * samples_per_prompt, len(candidate_lst))
        # Need to sort the responses back into a flattened list for each parent candidate
        full_responses = [[] for _ in range(len(candidate_lst))]
        for r_i, per_prompt_responses in enumerate(extended_responses):
            c_i = r_i // num_unique_prompts_per_cand
            full_responses[c_i].extend(per_prompt_responses)

        # responses contains the extracted plans
        responses = [[] for _ in range(len(candidate_lst))]
        for c_i in range(len(full_responses)):
            for s_i in range(len(full_responses[c_i])):
                responses[c_i].append(extract_plan(full_responses[c_i][s_i]))

        # Save the extracted plans and the full plans
        for c_i in range(len(responses)):
            save_str = save_strs[c_i]
            for s_i in range(num_plans):
                path = f"plan{'' if not save_str else '_' + save_str}"
                if force_opt_menu_lst is not None:
                    path += "_" + str(force_opt_menu_lst[c_i])
                path += "_" + str(s_i)
                full_plan_path = save_dir / (path + "_full.txt")
                plan_path = save_dir / (path + ".txt")
                full_plan = full_responses[c_i][s_i]
                extracted_plan = responses[c_i][s_i]
                if extracted_plan != full_plan:
                    with open(full_plan_path, "w") as f:
                        f.write(full_plan)
                with open(plan_path, "w") as f:
                    f.write(extracted_plan)
                logger.debug("Saved optimization plan to %s", plan_path)

        # Create the new plans
        new_cands = []
        for c_i, cand_resps in enumerate(responses):
            for plan in cand_resps:
                new_cands.append(CodeCandidate(candidate_lst[c_i], plan, None, plan_gen_model=self.llm_client.model))
        return new_cands

    def implement_code_parallel(self, candidate_lst: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_strs: list[str] = None, code_icl_examples: bool = True, prob: Prob = None) -> list[CodeCandidate]:
        if save_strs is not None:
            assert len(candidate_lst) == len(save_strs)
        loaded_code = []
        code_not_found = False
        for c_i in range(len(candidate_lst)):
            this_cand_loaded_code = []
            save_str = save_strs[c_i]
            for s_i in range(num_samples):
                path = save_dir / f"impl{'' if not save_str else '_' + save_str}_{s_i}_full.txt"
                if path.exists():
                    with open(path, "r") as f:
                        code = extracted_code = extract(f.read())
                        logger.debug("Loaded optimization plan from %s", path)
                        this_cand_loaded_code.append(code)
                else:
                    code_not_found = True
                    break
            if code_not_found:
                break
            loaded_code.append(this_cand_loaded_code)
        else:
            loaded_candidates = []
            for c_i in range(len(candidate_lst)):
                cand = candidate_lst[c_i]
                for s_i in range(num_samples):
                    new_cand = copy_candidate(cand)
                    new_cand.code = loaded_code[c_i][s_i]
                    new_cand.code_gen_model = self.llm_client.model
                    loaded_candidates.append(new_cand)
            logger.info("Loaded %d code implementations rather than generating new ones", len(loaded_candidates))
            return loaded_candidates

        prompts_lst = []
        for c_i in range(len(candidate_lst)):
            prompt_text = self._get_implement_code_prompt(candidate_lst[c_i], prob, code_icl_examples)
            # Save full prompt
            prompt_path = save_dir / f"prompt{'' if not save_strs[c_i] else '_' + save_strs[c_i]}.txt"
            with open(prompt_path, "w") as f:
                f.write(prompt_text)
            prompts_lst.append(prompt_text)

        temperature = 1
        llm_phase.set("code_generation")
        responses = self.llm_client.chat_async(
            prompts_lst=prompts_lst,
            num_samples=num_samples,
            temperature=temperature,
            reasoning_effort="medium"
        )
        logger.info("%s: finished generating %d code implementations for %d candidates.", self.llm_client.model, len(prompts_lst) * num_samples, len(candidate_lst))

        candidates: list[CodeCandidate] = []
        for c_i, cand_responses in enumerate(responses):
            this_plan_cands = []
            for s_i, sample_response in enumerate(cand_responses):
                # Save full response
                full_path = save_dir / f"impl{'' if not save_strs[c_i] else '_' + save_strs[c_i]}_{s_i}_full.txt"
                with open(full_path, "w") as f:
                    f.write(sample_response)
                # Extract just the code
                extracted_code = extract(sample_response)
                if not extracted_code:
                    logger.warning("Failed to extract code from plan %d response %d, full response was %s", 
                                   c_i, s_i, sample_response)
                path = save_dir / f"impl{'' if not save_strs[c_i] else '_' + save_strs[c_i]}_{s_i}.txt"
                with open(path, "w") as f:
                    f.writelines(extracted_code)
                logger.debug("Saved plan %d code impl %d to %s", c_i, s_i, path)
                new_cand = copy_candidate(candidate_lst[c_i])
                new_cand.code = extracted_code
                new_cand.code_gen_model = self.llm_client.model
                this_plan_cands.append(new_cand)
            candidates.extend(this_plan_cands)
        return candidates

    def direct_implement_code_parallel(
        self, candidate_lst: list[CodeCandidate], num_samples: int,
        save_dir: pathlib.Path, save_strs: list[str], prob: Prob,
        give_score_feedback: float = 1.0, give_hw_feedback: float = 1.0,
        include_ancestors: bool = False, dropout_menu_options: float = 1.0,
        cur_iter: int = None, num_iters: int = None,
        translate: bool = False,
    ) -> list[CodeCandidate]:
        """Generate optimized code directly from parent candidates, bypassing the planning phase.

        Each candidate in candidate_lst is a parent; we generate num_samples code
        implementations per parent using a single prompt that combines planning and
        implementation context.
        """
        if save_strs is not None:
            assert len(candidate_lst) == len(save_strs)

        loaded_code = []
        code_not_found = False
        for c_i in range(len(candidate_lst)):
            this_cand_loaded_code = []
            save_str = save_strs[c_i]
            for s_i in range(num_samples):
                path = save_dir / f"direct_impl{'' if not save_str else '_' + save_str}_{s_i}_full.txt"
                if path.exists():
                    with open(path, "r") as f:
                        code = extract(f.read())
                        this_cand_loaded_code.append(code)
                else:
                    code_not_found = True
                    break
            if code_not_found:
                break
            loaded_code.append(this_cand_loaded_code)
        else:
            loaded_candidates = []
            for c_i in range(len(candidate_lst)):
                cand = candidate_lst[c_i]
                for s_i in range(num_samples):
                    new_cand = CodeCandidate(cand, "direct implementation (no plan)", loaded_code[c_i][s_i],
                                             plan_gen_model=self.llm_client.model, code_gen_model=self.llm_client.model)
                    loaded_candidates.append(new_cand)
            logger.info("Loaded %d direct implementations from cache", len(loaded_candidates))
            return loaded_candidates

        if dropout_menu_options < 1 or (0 < give_score_feedback < 1) or (0 < give_hw_feedback < 1):
            num_unique_prompts_per_cand = num_samples
        else:
            num_unique_prompts_per_cand = 1

        prompts_lst = []
        for c_i, candidate in enumerate(candidate_lst):
            save_str = save_strs[c_i]
            for p in range(num_unique_prompts_per_cand):
                prompt_text = self._get_direct_implement_prompt(
                    candidate, prob,
                    give_score_feedback=give_score_feedback,
                    give_hw_feedback=give_hw_feedback,
                    include_ancestors=include_ancestors,
                    dropout_menu_options=dropout_menu_options,
                    cur_iter=cur_iter, num_iters=num_iters,
                    translate=translate,
                )
                prompt_path = save_dir / f"direct_prompt{'' if not save_str else '_' + save_str}_{p}.txt"
                with open(prompt_path, "w") as f:
                    f.write(prompt_text)
                prompts_lst.append(prompt_text)

        samples_per_prompt = num_samples // num_unique_prompts_per_cand

        llm_phase.set("code_generation")
        extended_responses = self.llm_client.chat_async(
            prompts_lst=prompts_lst,
            num_samples=samples_per_prompt,
            temperature=1,
        )
        logger.info("%s: finished generating %d direct implementations for %d candidates.",
                     self.llm_client.model, len(prompts_lst) * samples_per_prompt, len(candidate_lst))

        full_responses = [[] for _ in range(len(candidate_lst))]
        for r_i, per_prompt_responses in enumerate(extended_responses):
            c_i = r_i // num_unique_prompts_per_cand
            full_responses[c_i].extend(per_prompt_responses)

        candidates: list[CodeCandidate] = []
        for c_i in range(len(candidate_lst)):
            save_str = save_strs[c_i]
            for s_i, sample_response in enumerate(full_responses[c_i]):
                full_path = save_dir / f"direct_impl{'' if not save_str else '_' + save_str}_{s_i}_full.txt"
                with open(full_path, "w") as f:
                    f.write(sample_response)
                extracted_code = extract(sample_response)
                if not extracted_code:
                    logger.warning("Failed to extract code from direct impl %d response %d", c_i, s_i)
                path = save_dir / f"direct_impl{'' if not save_str else '_' + save_str}_{s_i}.txt"
                with open(path, "w") as f:
                    f.write(extracted_code)
                new_cand = CodeCandidate(candidate_lst[c_i], "direct implementation (no plan)", extracted_code,
                                         plan_gen_model=self.llm_client.model, code_gen_model=self.llm_client.model)
                candidates.append(new_cand)
        return candidates

    def direct_implement_code_edits_parallel(
        self, candidate_lst: list[CodeCandidate], num_samples: int,
        save_dir: pathlib.Path, save_strs: list[str], prob: Prob,
        give_score_feedback: float = 1.0, give_hw_feedback: float = 1.0,
        include_ancestors: bool = False, dropout_menu_options: float = 1.0,
        cur_iter: int = None, num_iters: int = None,
        translate: bool = False,
    ) -> list[CodeCandidate]:
        """Direct (plan-free) edit-based code generation.

        Like direct_implement_code_parallel but outputs structured JSON edits
        instead of full code rewrites.
        """
        if save_strs is not None:
            assert len(candidate_lst) == len(save_strs)

        messages_lst = []
        for c_i, candidate in enumerate(candidate_lst):
            messages = self._get_direct_implement_edits_messages(
                candidate, prob,
                give_score_feedback=give_score_feedback,
                give_hw_feedback=give_hw_feedback,
                include_ancestors=include_ancestors,
                dropout_menu_options=dropout_menu_options,
                cur_iter=cur_iter, num_iters=num_iters,
                translate=translate,
            )
            messages_lst.append(messages)

        return self._run_edits_pipeline(
            messages_lst, candidate_lst, num_samples, save_dir, save_strs,
            file_prefix="direct_edit", plan_label="direct implementation (no plan)",
            base_code_lst=[c.code for c in candidate_lst],
        )

    def _run_edits_pipeline(
        self, messages_lst: list[list[dict]], candidate_lst: list[CodeCandidate],
        num_samples: int, save_dir: pathlib.Path, save_strs: list[str],
        file_prefix: str, plan_label: str,
        base_code_lst: list[str] = None,
    ) -> list[CodeCandidate]:
        """Shared pipeline: call LLM with structured JSON edits, parse, and apply.

        Returns one CodeCandidate per (candidate, sample) pair.

        If *base_code_lst* is provided, edits are applied against that code
        (one entry per candidate).  Otherwise falls back to
        ``candidate.parent.code`` (plan-then-edit path) or
        ``candidate.code`` (direct-edit path).
        """
        # Check for cached results
        loaded_code = []
        code_not_found = False
        for c_i in range(len(candidate_lst)):
            this_cand_loaded = []
            save_str = save_strs[c_i]
            for s_i in range(num_samples):
                path = save_dir / f"{file_prefix}{'' if not save_str else '_' + save_str}_{s_i}.txt"
                if path.exists():
                    with open(path, "r") as f:
                        this_cand_loaded.append(f.read())
                else:
                    code_not_found = True
                    break
            if code_not_found:
                break
            loaded_code.append(this_cand_loaded)
        else:
            loaded_candidates = []
            for c_i in range(len(candidate_lst)):
                cand = candidate_lst[c_i]
                for s_i in range(num_samples):
                    new_cand = CodeCandidate(
                        cand, plan_label or cand.plan, loaded_code[c_i][s_i],
                        plan_gen_model=self.llm_client.model,
                        code_gen_model=self.llm_client.model,
                    )
                    loaded_candidates.append(new_cand)
            logger.info("Loaded %d %s implementations from cache", len(loaded_candidates), file_prefix)
            return loaded_candidates

        # Save prompts
        for c_i in range(len(candidate_lst)):
            prompt_path = save_dir / f"{file_prefix}_prompt{'' if not save_strs[c_i] else '_' + save_strs[c_i]}.txt"
            with open(prompt_path, "w") as f:
                f.write(str(messages_lst[c_i]))

        llm_phase.set("code_generation")
        grouped_results = self.llm_client.chat_messages_async(
            messages_lst,
            num_samples=num_samples,
            response_format=EDITS_JSON_SCHEMA,
            temperature=1,
        )
        logger.info("%s: finished generating %d %s responses for %d candidates.",
                    self.llm_client.model, len(messages_lst) * num_samples, file_prefix, len(candidate_lst))

        candidates: list[CodeCandidate] = []
        for c_i in range(len(candidate_lst)):
            if base_code_lst is not None:
                base_code = base_code_lst[c_i]
            elif candidate_lst[c_i].parent:
                base_code = candidate_lst[c_i].parent.code
            else:
                base_code = candidate_lst[c_i].code
            for s_i in range(num_samples):
                response = grouped_results[c_i][s_i]
                response_text = response.get("content", "") or ""

                full_path = save_dir / f"{file_prefix}{'' if not save_strs[c_i] else '_' + save_strs[c_i]}_{s_i}_full.txt"
                with open(full_path, "w") as f:
                    try:
                        f.write(json.dumps(json.loads(response_text), indent=2))
                    except (json.JSONDecodeError, TypeError):
                        f.write(response_text)

                edits = parse_edits_response(response_text)
                edited_code = None
                if edits is not None:
                    try:
                        edited_code = apply_edits(base_code, edits)
                    except ValueError as e:
                        logger.warning("%s: Edit application failed for %s %d implementation %d: %s",
                                       self.llm_client.model, file_prefix, c_i, s_i, e)

                if edited_code is None:
                    edited_code = extract(response_text)
                    if edited_code and edited_code != response_text:
                        logger.info("%s: Edit parse failed, fell back to full-code extraction for %s %d implementation %d",
                                    self.llm_client.model, file_prefix, c_i, s_i)
                    else:
                        logger.warning("%s: Failed to get edits or code for %s %d implementation %d",
                                       self.llm_client.model, file_prefix, c_i, s_i)
                        edited_code = base_code

                path = save_dir / f"{file_prefix}{'' if not save_strs[c_i] else '_' + save_strs[c_i]}_{s_i}.txt"
                with open(path, "w") as f:
                    f.write(edited_code)

                new_cand = CodeCandidate(
                    candidate_lst[c_i], plan_label or candidate_lst[c_i].plan, edited_code,
                    plan_gen_model=self.llm_client.model,
                    code_gen_model=self.llm_client.model,
                )
                candidates.append(new_cand)
        return candidates

    def implement_code_edits_parallel(self, candidate_lst: list[CodeCandidate], num_samples: int,
                                      save_dir: pathlib.Path, save_strs: list[str] = None,
                                      code_icl_examples: bool = True, prob: Prob = None) -> list[CodeCandidate]:
        """Edit-based code generation: ask the LLM for structured JSON edits instead of full rewrites."""
        if save_strs is not None:
            assert len(candidate_lst) == len(save_strs)

        messages_lst = []
        for c_i in range(len(candidate_lst)):
            messages = self._get_implement_edits_messages(candidate_lst[c_i], prob)
            messages_lst.append(messages)

        return self._run_edits_pipeline(
            messages_lst, candidate_lst, num_samples, save_dir, save_strs,
            file_prefix="edit_impl", plan_label=None,
        )

    def combine_candidates(self, candidates: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_str: str="", prob: Prob = None) -> list[CodeCandidate]:
        loaded_code = []
        for c_i in range(num_samples):
            path = save_dir / f"combined{'' if not save_str else '_' + save_str}_{c_i}.txt"
            if path.exists():
                with open(path, "r") as f:
                    code = f.read()
                    logger.debug("Loaded optimization plan from %s", path)
                    loaded_code.append(code)
            else:
                break
        else:
            logger.info("Loaded %d code implementations rather than generating new ones", num_samples)
            loaded_candidates = []
            for c_i in range(num_samples):
                loaded_candidates.append(CodeCandidate(candidates, "Combined code", loaded_code[c_i], code_gen_model=self.llm_client.model))
            return loaded_candidates

        prompt_text = self._get_combine_candidates_prompt(candidates, prob)

        # Save full prompt
        prompt_path = save_dir / f"prompt{'' if not save_str else '_' + save_str}.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt_text)
        
        temperature = 1
        responses = self.llm_client.chat(
            prompt=prompt_text,
            num_samples=num_samples,
            temperature=temperature
        )
        combined_candidates = []
        for c_i, c in enumerate(responses):
            # Save full response
            full_path = save_dir / f"combined{'' if not save_str else '_' + save_str}_{c_i}_full.txt"
            with open(full_path, "w") as f:
                f.write(c)
            # Extract just the code
            extracted_code = extract(c)
            if not extracted_code:
                logger.warning("Failed to extract code from response %d, full response was %s", c_i, c)
            path = save_dir / f"combined{'' if not save_str else '_' + save_str}_{c_i}.txt"
            with open(path, "w") as f:
                f.writelines(extracted_code)
            logger.debug("Saved combined code %d to %s", c_i, path)

            combined_candidates.append(CodeCandidate(candidates, "Combine parents", extracted_code, code_gen_model=self.llm_client.model))
        return combined_candidates

    def reimplement_failed_code_parallel(self, candidate_lst: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_strs: list[str] = None, prob: Prob = None) -> list[CodeCandidate]:
        """
        Reimplement failed implementations using stdout/stderr from the last attempt.
        """
        if save_strs is not None:
            assert len(candidate_lst) == len(save_strs)
        
        loaded_code = []
        code_not_found = False
        for c_i in range(len(candidate_lst)):
            this_cand_loaded_code = []
            save_str = save_strs[c_i]
            for s_i in range(num_samples):
                path = save_dir / f"reimplement{'' if not save_str else '_' + save_str}_{s_i}_full.txt"
                if path.exists():
                    with open(path, "r") as f:
                        code = extracted_code = extract(f.read())
                        logger.debug("Loaded reimplemented code from %s", path)
                        this_cand_loaded_code.append(code)
                else:
                    code_not_found = True
                    break
            if code_not_found:
                break
            loaded_code.append(this_cand_loaded_code)
        else:
            loaded_candidates = []
            for c_i in range(len(candidate_lst)):
                cand = candidate_lst[c_i]
                for s_i in range(num_samples):
                    new_cand = copy_candidate(cand)
                    new_cand.code = loaded_code[c_i][s_i]
                    new_cand.code_gen_model = self.llm_client.model
                    loaded_candidates.append(new_cand)
            logger.info("Loaded %d reimplemented code implementations rather than generating new ones", len(loaded_candidates))
            return loaded_candidates

        prompts_lst = []
        for c_i in range(len(candidate_lst)):
            prompt_text = self._get_reimplement_failed_code_prompt(candidate_lst[c_i], prob)
            # Save full prompt
            prompt_path = save_dir / f"reimplement_prompt{'' if not save_strs[c_i] else '_' + save_strs[c_i]}.txt"
            with open(prompt_path, "w") as f:
                f.write(prompt_text)
            prompts_lst.append(prompt_text)

        llm_phase.set("code_generation")
        responses = self.llm_client.chat_async(
            prompts_lst=prompts_lst,
            num_samples=num_samples,
            temperature=1,
        )
        logger.info("%s: finished generated %d reimplemented code implementations for %d failed candidates.", self.llm_client.model, len(prompts_lst) * num_samples, len(candidate_lst))

        candidates: list[CodeCandidate] = []
        for c_i, cand_responses in enumerate(responses):
            this_cands = []
            for s_i, sample_response in enumerate(cand_responses):
                # Save full response
                full_path = save_dir / f"reimplement{'' if not save_strs[c_i] else '_' + save_strs[c_i]}_{s_i}_full.txt"
                with open(full_path, "w") as f:
                    f.write(sample_response)
                # Extract just the code
                extracted_code = extract(sample_response)
                if not extracted_code:
                    logger.warning("Failed to extract code from reimplement %d response %d, full response was %s", 
                                   c_i, s_i, sample_response)
                path = save_dir / f"reimplement{'' if not save_strs[c_i] else '_' + save_strs[c_i]}_{s_i}.txt"
                with open(path, "w") as f:
                    f.writelines(extracted_code)
                logger.debug("Saved reimplemented %d code impl %d to %s", c_i, s_i, path)
                new_cand = copy_candidate(candidate_lst[c_i])
                new_cand.code = extracted_code
                new_cand.code_gen_model = self.llm_client.model
                this_cands.append(new_cand)
            candidates.extend(this_cands)
        return candidates

    def _get_reimplement_failed_code_prompt(self, candidate: CodeCandidate, prob: Prob = None) -> str:
        """
        Base method to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_reimplement_failed_code_prompt")
