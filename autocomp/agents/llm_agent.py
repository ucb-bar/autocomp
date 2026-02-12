import pathlib
import random

from autocomp.common import logger, LLMClient, llm_utils
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate, copy_candidate


def extract(code_str: str) -> str:
    """
    Takes LLM-generated code and extracts the "test" wrapper function

    for example:
    '''
    void test(Kinf, r, K_r) {
        config_ex(WEIGHT_STATIONARY,  NO_ACTIVATION, true, false);
        config_st(1 * sizeof(float)); // output K_r has 1 column in DRAM
        ...
        mvout(K_r + 0x8, K_r_acc_addr + 8, 1, 4); // mvout the third 4x1 block of K_r
        fence();
    }
    '''
    """
    # Remove the function wrapper and return only the body of the function
    if not code_str:
        return code_str
    
    if "```python" in code_str:
        code_str = code_str.split("```python")[1].split("```")[0]
        return code_str
    if "```cuda" in code_str:
        code_str = code_str.split("```cuda")[1].split("```")[0]
        return code_str

    if "void test" in code_str:
        from_void_test_str = code_str[code_str.find("void test"):]
        # end = from_void_test_str.rfind('}\n')
        # Iterate through all characters until matching curly braces have been found
        # Ignore curly braces in comments
        open_braces = 0
        in_comment = False
        in_single_line_comment = False
        for i, char in enumerate(from_void_test_str):
            if from_void_test_str[i:i+2] == '/*':
                in_comment = True
            elif from_void_test_str[i:i+2] == '*/':
                in_comment = False
            if from_void_test_str[i:i+2] == '//':
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
            body = from_void_test_str[:end+1]
        except:
            return from_void_test_str
        return body

    # Fallback: just return the whole thing
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
    def __init__(self, model_with_provider: str):
        if "::" in model_with_provider:
            provider, model = model_with_provider.split("::", 1)
        else:
            provider = None
            model = model_with_provider
        self.llm_client = LLMClient(model, provider)

    def analyze_code(self, candidate: CodeCandidate, num_to_gen: int, save_dir: pathlib.Path, save_str: str) -> list[str]:
        return ["Analysis 1", "Analysis 2", "Analysis 3"]

    def evaluate_code_quality(self, candidates: list[CodeCandidate], save_dir: pathlib.Path = None) -> list[float]:
        """
        Evaluate the quality of code candidates using LLM before running benchmarks.
        Returns a list of quality scores (0.0 to 1.0) where higher is better.
        """
        # Default implementation returns neutral scores for all candidates
        return [0.5] * len(candidates)

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
                path = f"plan{'' if not save_str else '_' + save_str}"
                if force_opt_menu_lst is not None:
                    path += "_" + str(force_opt_menu_lst[c_i])
                path += "_" + str(s_i) + ".txt"
                path = save_dir / path
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
        candidates_to_gen = num_plans // num_unique_prompts_per_cand

        extended_responses = self.llm_client.chat_async(
            prompts_lst=prompts_lst,
            num_candidates=candidates_to_gen,  # number of candidates,
            temperature=temperature
        )
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

        # Create the new candidates
        new_cands = []
        for c_i, cand_resps in enumerate(responses):
            for plan in cand_resps:
                new_cands.append(CodeCandidate(candidate_lst[c_i], plan, None, plan_gen_model=self.llm_client.model))
        return new_cands

    def propose_optimizations(self, candidate: CodeCandidate, num_plans: int, save_dir: pathlib.Path, save_str: str, 
                              prob: Prob,
                              force_opt_menu: int = None, 
                              prompt_end: str = "", 
                              analysis: str = "", 
                              shuffle_opts: bool = False, 
                              give_score_feedback: bool = True,
                              give_util_feedback: bool = False,
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
        for c_i in range(num_plans):
            path = f"plan{'' if not save_str else '_' + save_str}"
            if force_opt_menu is not None:
                path += "_" + str(force_opt_menu)
            path += "_" + str(c_i) + ".txt"
            path = save_dir / path
            if path.exists():
                with open(path, "r") as f:
                    plan = f.read()
                    logger.debug("Loaded optimization plan from %s", path)
                    loaded_plans.append(plan)
            else:
                break
        else:
            loaded_cands = [CodeCandidate(candidate, plan, None, plan_gen_model=self.llm_client.model) for plan in loaded_plans]
            logger.info("Loaded %d optimization plans rather than generating new ones", num_plans)
            return loaded_cands

        responses = []
        if dropout_menu_options < 1:
            num_unique_prompts_per_cand = num_plans
        else:
            num_unique_prompts_per_cand = 1
        for p in range(num_unique_prompts_per_cand):
            # Add the previously iterated plans and code to the prompt
            prompt_text = self._get_propose_optimizations_prompt(candidate, prob, force_opt_menu, prompt_end, analysis, shuffle_opts,
                                                                 give_score_feedback, give_util_feedback, include_ancestors, plan_icl_examples, cur_iter, num_iters,
                                                                 dropout_menu_options, translate)

            # Save full prompt
            prompt_path = f"prompt{'' if not save_str else '_' + save_str}"
            if force_opt_menu is not None:
                prompt_path += "_" + str(force_opt_menu)
            else:
                prompt_path += "_" + str(p)
            prompt_path += ".txt"
            prompt_path = save_dir / prompt_path

            with open(prompt_path, "w") as f:
                f.write(prompt_text)

            temperature = 1
            candidates_to_gen = num_plans // num_unique_prompts_per_cand

            resp = self.llm_client.chat_async(
                prompts_lst=[prompt_text],
                num_candidates=candidates_to_gen,  # number of candidates,
                temperature=temperature
            )[0]
            # resp = self.llm_client.chat(
            #     messages=messages,
            #     num_candidates=candidates_to_gen,  # number of candidates,
            #     temperature=temperature
            # )
            responses.extend(resp)

        for c_i, c in enumerate(responses):
            path = f"plan{'' if not save_str else '_' + save_str}"
            if force_opt_menu is not None:
                path += "_" + str(force_opt_menu)
            path += "_" + str(c_i) + ".txt"
            path = save_dir / path
            with open(path, "w") as f:
                f.write(c)
            logger.debug("Saved optimization plan to %s", path)

        new_cands = [CodeCandidate(candidate, plan, None, plan_gen_model=self.llm_client.model) for plan in responses]
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
        responses = self.llm_client.chat_async(
            prompts_lst=prompts_lst,
            num_candidates=num_samples,  # number of candidates,
            temperature=temperature,
            reasoning_effort="medium"
        )

        candidates: list[CodeCandidate] = [] # Flat list of new implemented candidates
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
                # Make a copy of the candidate with the new code impl
                new_cand = copy_candidate(candidate_lst[c_i])
                new_cand.code = extracted_code
                new_cand.code_gen_model = self.llm_client.model
                this_plan_cands.append(new_cand)
            candidates.extend(this_plan_cands)
        return candidates

    def implement_code(self, candidate: CodeCandidate, num_samples: int, save_dir: pathlib.Path, save_str: str="", prob: Prob = None) -> list[CodeCandidate]:
        assert num_samples > 0, "Number of samples must be greater than 0"

        loaded_code = []
        for c_i in range(num_samples):
            path = save_dir / f"impl{'' if not save_str else '_' + save_str}_{c_i}.txt"
            if path.exists():
                with open(path, "r") as f:
                    code = f.read()
                    logger.debug("Loaded optimization plan from %s", path)
                    loaded_code.append(code)
            else:
                break
        else:
            logger.info("Loaded %d code implementations rather than generating new ones", num_samples)
            candidate.code = loaded_code[0]
            loaded_candidates = [candidate]
            for c_i in range(1, num_samples):
                new_cand = copy_candidate(candidate)
                new_cand.code = loaded_code[c_i]
                new_cand.code_gen_model = self.llm_client.model
                loaded_candidates.append(new_cand)
            return loaded_candidates

        prompt_text = self._get_implement_code_prompt(candidate, prob)
        # Save full prompt
        prompt_path = save_dir / f"prompt{'' if not save_str else '_' + save_str}.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt_text)

        temperature = 1
        responses = self.llm_client.chat(
            prompt=prompt_text,
            num_candidates=num_samples,  # number of candidates,
            temperature=temperature
        )

        candidates = []
        for c_i, c in enumerate(responses):
            # Save full response
            full_path = save_dir / f"impl{'' if not save_str else '_' + save_str}_{c_i}_full.txt"
            with open(full_path, "w") as f:
                f.write(c)
            # Extract just the code
            extracted_code = extract(c)
            if not extracted_code:
                logger.warning("Failed to extract code from response %d, full response was %s", c_i, c)
            path = save_dir / f"impl{'' if not save_str else '_' + save_str}_{c_i}.txt"
            with open(path, "w") as f:
                f.writelines(extracted_code)
            logger.debug("Saved code impl %d to %s", c_i, path)

            # Make a copy of the candidate with the new code impl
            new_cand = copy_candidate(candidate)
            new_cand.code = extracted_code
            new_cand.code_gen_model = self.llm_client.model
            candidates.append(new_cand)
        return candidates

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
                loaded_candidates.append(CodeCandidate(candidates, "Combined code", loaded_code[c_i]), code_gen_model=self.llm_client.model)
            return loaded_candidates

        prompt_text = self._get_combine_candidates_prompt(candidates, prob)

        # Save full prompt
        prompt_path = save_dir / f"prompt{'' if not save_str else '_' + save_str}.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt_text)
        
        temperature = 1
        responses = self.llm_client.chat(
            prompt=prompt_text,
            num_candidates=num_samples,  # number of candidates,
            temperature=temperature
        )
        candidates = []
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

            candidates.append(CodeCandidate(candidates, "Combine parents", extracted_code, code_gen_model=self.llm_client.model))
        return candidates

    def reimplement_failed_code_parallel(self, candidate_lst: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_strs: list[str] = None, prob: Prob = None) -> list[CodeCandidate]:
        """
        Reimplement failed code candidates using stdout/stderr from the last attempt.
        This method is parallelized across multiple candidates.
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

        responses = self.llm_client.chat_async(
            prompts_lst=prompts_lst,
            num_candidates=num_samples,
            temperature=1,
        )

        candidates: list[CodeCandidate] = []
        for c_i, cand_responses in enumerate(responses):
            this_plan_cands = []
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
                # Make a copy of the candidate with the new code impl
                new_cand = copy_candidate(candidate_lst[c_i])
                new_cand.code = extracted_code
                new_cand.code_gen_model = self.llm_client.model
                this_plan_cands.append(new_cand)
            candidates.extend(this_plan_cands)
        return candidates

    def _get_reimplement_failed_code_prompt(self, candidate: CodeCandidate, prob: Prob = None) -> str:
        """
        Base method to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_reimplement_failed_code_prompt")
