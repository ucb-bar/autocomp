import pathlib
import random

from autocomp.common import logger, LLMClient, llm_utils
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate, copy_candidate
# from autocomp.search.annotate_perf import GemminiCode
from prompts.gemmini import isa_prompt_conv, isa_prompt_admm,plan_prompt, gemmini_rules, tiling_example, if_example, if_example_matmul
from prompts.trn import fusion_example
from prompts.trn.nki_isa_generator import NkiIsaGenerator
from prompts.cuda import tensor_examples

prob_macs_map = {
"exo": [
    512 * 512 * 512,
    12544 * 64 * 256,
    12544 * 256 * 64,
    3136 * 128 * 512,
    3136 * 512 * 128,
    784 * 256 * 1024,
],
"exo-conv": [
    462422016,
    462422016,
    462422016,
],
"admm-multifunction": [
    (12*4*1+12*4*1+12*12*1)*5,
    (4*12*1+4*4*1+4*12*1+12*12*1)*10,
]
}

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

    # end = from_void_test_str.find('fence();')
    # find_last_curly = 0
    # last_idx = len(from_void_test_str)
    # while end + find_last_curly < len(from_void_test_str):
    #     if from_void_test_str[end + find_last_curly] == '}':
    #         last_idx = end + find_last_curly
    #     find_last_curly += 1
    # body = from_void_test_str[:last_idx+1]
    # return body
    # body = code_str[start:end].split("\n")
    # new_body = []
    # for line in body:
    #     new_body.append(line.strip())
    # return "\n".join(new_body)

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
    def __init__(self, model):
        self.llm_client = LLMClient(model)

    def get_opt_menu_options(self):
        # Mock-up method to get the optimization menu options
        return ["Opt 1", "Opt 2", "Opt 3"]

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
                              give_spad_acc_feedback: float = 1.0,
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

        if dropout_menu_options < 1 or (0 < give_score_feedback < 1) or (0 < give_util_feedback < 1) or (0 < give_spad_acc_feedback < 1):
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
                                                                    give_score_feedback, give_util_feedback, give_spad_acc_feedback, include_ancestors, plan_icl_examples, cur_iter, num_iters,
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
            temperature=temperature
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

    def combine_candidates(self, candidates: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_str: str="") -> list[CodeCandidate]:
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

        prompt_text = self._get_combine_candidates_prompt(candidates)

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


class GemminiLLMAgent(LLMAgent):
    def __init__(self, model, pe_dim):
        super().__init__(model)
        self.pe_dim = pe_dim

    def __repr__(self):
        return f"GemminiLLMAgent({self.llm_client.model}, {self.pe_dim})"

    def get_opt_menu_options(self, prob: Prob):
        if "admm" in prob.prob_type:
            return [
                "remove unnecessary code",
                "simplify arithmetic and propagate constants to simplify expressions",
                "merge instructions",
                "merge high-level operations",
                "reorder operations or blocks of operations",
                "move CPU-based computation to the accelerator",
                "add or subtract a matrix using the bias",
                "hoist redundant operations out of loops",
                "substitute operations with equivalent operations that are faster",
                "pipeline operations to better overlap computation and data movement",
                "eliminate data dependencies and fence operations",
                "minimize data movement",
                "minimize loop overhead",
                "other methods not listed here.",
            ]
        else: # gemm and conv
            return [
                "modify loop tiling",
                "loop reordering",
                "split loops",
                "fuse loops",
                "simplify arithmetic and propagate constants to simplify expressions",
                "reorder computations or blocks of computations",
                "loop unrolling",
                "double buffering",
                "move more data to the scratchpad in a more outer loop to increase data reuse",
                "spread data throughout the scratchpad rather than loading to the same location repeatedly",
                "load data to the scratchpad across outer loop iterations and use if statements to prevent redundant loads on loops inner to those",
                "hoist redundant operations out of loops",
                "substitute operations with equivalent operations that are faster",
                "pipeline operations to better overlap computation and data movement",
                "minimize data movement",
                "minimize loop overhead",
                "other methods not listed here.",
            ]
        # "remove redundant code",
        # "compute common sub-expressions ahead of time",
        # "merge instructions",
        # "merge high-level operations",
        # "move CPU-based computation to the accelerator",

    def analyze_code(self, candidate: CodeCandidate, num_to_gen: int, save_dir: pathlib.Path, save_str: str) -> list[str]:
        """
        Generate an analysis of current code based on performance feedback
        """
        if self.pe_dim == 4:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_admm.PROMPT(self.pe_dim) + "\n"
        elif self.pe_dim == 16:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.PROMPT(self.pe_dim) + "\n"
        prompt_text += "Elements in the scratchpad are 1 byte, and elements in the accumulator are 4 bytes.\n"
        prompt_text += "The scratchpad size is 256KB and the accumulator size is 64KB. The systolic array is 16 by 16.\n"
        prompt_text += "The original code is as follows:\n" + candidate.code + "\n"
        prompt_text += "You are an optimizing compiler that produces high-performance Gemmini code. Based on this information, analyze the code and identify the single most impactful bottleneck increasing cycle count."
        # Save prompt
        prompt_path = f"prompt{'' if not save_str else '_' + save_str}"
        prompt_path += ".txt"
        prompt_path = save_dir / prompt_path
        with open(prompt_path, "w") as f:
            f.write(prompt_text)
        # Call LLM to generate responses
        messages = [{"role": "user", "content": prompt_text}]
        temperature = 1
        responses = self.llm_client.chat(
            messages=messages,
            num_candidates=num_to_gen,  # number of candidates,
            temperature=temperature
        )

        # Save responses
        for c_i, c in enumerate(responses):
            path = f"analyze{'' if not save_str else '_' + save_str}"
            path += "_" + str(c_i) + ".txt"
            path = save_dir / path
            with open(path, "w") as f:
                f.write(c)
            logger.debug("Saved select_optimization response to %s", path)

        return responses

    # def select_optimization(self, candidate: CodeCandidate, num_to_gen: int, save_dir: pathlib.Path, save_str: str, iteration: int, num_interations: int, analysis_str: str = "") -> str:
    #     """
    #     Select an optimization to apply from the menu
    #     """
    #     # Construct the prompt
    #     prompt_text: str = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.TEMPLATE_TEXT + "\n"
    #     # Add the previously iterated plans and code to the prompt
    #     parents_prompt = ""
    #     cur_cand: CodeCandidate = candidate
    #     while cur_cand is not None:
    #         # Go up to each parent and append to front of prompt
    #         # annotated_code = GemminiCode(cur_cand.code, self.pe_dim).annotate_perf()
    #         if len(cur_cand.feedback) > 0:
    #             parents_prompt = "\n".join(cur_cand.feedback) + "\n" + parents_prompt
    #         if cur_cand.score is not None:
    #             parents_prompt = f"The latency of this code was: {cur_cand.score} cycles.\n" + parents_prompt
    #         if cur_cand.plan is not None:
    #             parents_prompt = "\nNext, we applied this <optimization> to the code:\n" + cur_cand.opt_selection + "\nThe generated code was:\n" + cur_cand.code + "\n" + parents_prompt
    #         else:
    #             parents_prompt = "The original unoptimized code was:\n" + cur_cand.code + "\n" + parents_prompt
    #         cur_cand = cur_cand.parent
    #     prompt_text += parents_prompt
    #     if analysis_str:
    #         prompt_text += analysis_str + "\n"
    #     prompt_text += "It is currently phase " + str(iteration) + " out of " + str(num_interations) + " optimization phases.\n"
    #     prompt_text += "You are an optimizing compiler that produces high-performance Gemmini code. Based on this information, select an <optimization> to apply.\n"
    #     menu_options_text = "<optimizations>\n"
    #     for i, opt in enumerate(self.get_opt_menu_options()):
    #         menu_options_text += f"{i+1}. {opt}\n"
    #     prompt_text += menu_options_text

    #     # Save prompt
    #     prompt_path = f"prompt{'' if not save_str else '_' + save_str}"
    #     prompt_path += ".txt"
    #     prompt_path = save_dir / prompt_path
    #     with open(prompt_path, "w") as f:
    #         f.write(prompt_text)

    #     # Call LLM to generate responses
    #     messages = {"role": "user", "content": prompt_text}
    #     temperature = 1
    #     responses = self.llm_client.chat(
    #         messages=messages,
    #         num_candidates=num_to_gen,  # number of candidates,
    #         temperature=temperature
    #     )

    #     # Save responses
    #     for c_i, c in enumerate(responses):
    #         path = f"select{'' if not save_str else '_' + save_str}"
    #         path += "_" + str(c_i) + ".txt"
    #         path = save_dir / path
    #         with open(path, "w") as f:
    #             f.write(c)
    #         logger.debug("Saved select_optimization response to %s", path)

    #     return responses

    # def gen_plan_from_opt(self, candidate: CodeCandidate, num_to_gen: int, opt_selection: str, save_dir: pathlib.Path, save_str: str) -> str:
    #     # Construct the prompt
    #     prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.TEMPLATE_TEXT + "\n"
    #     prompt_text += "The original code was:" + candidate.code + "\n"
    #     prompt_text += "You have selected the following <optimization> to apply:\n" + opt_selection + "\n"
    #     prompt_text += "You are an optimizing compiler that produces high-performance Gemmini code. Based on this information, come up with a plan of how you would implement the <optimization> described.\n"

    #     # Save prompt
    #     prompt_path = f"prompt{'' if not save_str else '_' + save_str}"
    #     prompt_path += ".txt"
    #     prompt_path = save_dir / prompt_path
    #     with open(prompt_path, "w") as f:
    #         f.write(prompt_text)

    #     # Call LLM to generate responses
    #     messages = {"role": "user", "content": prompt_text}
    #     temperature = 1
    #     responses = self.llm_client.chat(
    #         messages=messages,
    #         num_candidates=num_to_gen,  # number of candidates,
    #         temperature=temperature
    #     )

    #     # Save responses
    #     for c_i, c in enumerate(responses):
    #         path = f"plan{'' if not save_str else '_' + save_str}"
    #         path += "_" + str(c_i) + ".txt"
    #         path = save_dir / path
    #         with open(path, "w") as f:
    #             f.write(c)
    #         logger.debug("Saved optimization plan to %s", path)

    #     return responses

    def _get_propose_optimizations_prompt(self, candidate: CodeCandidate,
                                          prob: Prob,
                                          force_opt_menu: int, 
                                          prompt_end: str, 
                                          analysis: str, 
                                          shuffle_opts: bool, 
                                          give_score_feedback: float,
                                          give_util_feedback: float,
                                          give_spad_acc_feedback: float,
                                          include_ancestors: bool,
                                          plan_icl_examples: bool,
                                          cur_iter: int,
                                          num_iters: int,
                                          dropout_menu_options: float,
                                          translate: bool,
                                         ) -> list[str]:
        # Select which menu options will appear
        plan_prompt_texts = plan_prompt.PROMPT(self.pe_dim)
        menu_options_text = ""
        opt_lst = self.get_opt_menu_options(prob)
        if dropout_menu_options < 1 and not force_opt_menu:
            opt_lst = [opt for opt in opt_lst if random.random() < dropout_menu_options]
        if shuffle_opts:
            random.shuffle(opt_lst)
        include_util_feedback = random.random() < give_util_feedback
        include_score_feedback = random.random() < give_score_feedback
        include_spad_acc_feedback = random.random() < give_spad_acc_feedback

        parents_prompt = ""
        cur_cand = candidate
        while cur_cand is not None:
            # Go up to each parent and append to front of prompt
            # annotated_code = GemminiCode(cur_cand.code, self.pe_dim).annotate_perf()
            if include_spad_acc_feedback:
                parents_prompt = "\n".join(cur_cand.spad_acc_stats) + "\n" + parents_prompt
            if include_util_feedback and (cur_cand.score is not None):
                macs = prob_macs_map[prob.prob_type][prob.prob_id]
                theoretical_min_cycles = macs / (self.pe_dim ** 2)
                util = theoretical_min_cycles / cur_cand.score * 100
                parents_prompt = f"The utilization of this code was {round(util)}%.\n" + parents_prompt
            if include_score_feedback and (cur_cand.score is not None):
                parents_prompt = f"The latency of this code was {cur_cand.score} cycles.\n" + parents_prompt
            if not include_ancestors:
                parents_prompt = "\nThe original unoptimized code was:\n" + cur_cand.code + "\n" + parents_prompt
                break # No need to go up past the immediate parent
            elif cur_cand.plan is not None:
                parents_prompt = "\nNext, we applied this plan to the code:\n" + cur_cand.plan + "\nThe generated code was:\n" + cur_cand.code + "\n" + parents_prompt
            else:
                parents_prompt = "The original unoptimized code was:\n" + cur_cand.code + "\n" + parents_prompt
            cur_cand = cur_cand.parent

        if analysis:
            parents_prompt += "\n" + analysis

        # Initialize the prompt with the parents info
        # prompt_text = ""
        if self.pe_dim == 4:
            prompt_text = "\nThe Gemmini accelerator's ISA is as follows:" + isa_prompt_admm.PROMPT(self.pe_dim)
        elif self.pe_dim == 16:
            prompt_text = "\nThe Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.PROMPT(self.pe_dim)
        if plan_icl_examples:
            if "modify loop tiling" in opt_lst:
                prompt_text += tiling_example.PROMPT()
            if "move more data to the scratchpad in a more outer loop to increase data reuse" in opt_lst:
                prompt_text += if_example.PROMPT()
            elif "spread data throughout the scratchpad rather than loading to the same location repeatedly" in opt_lst:
                prompt_text += if_example.PROMPT()
            elif "load data to the scratchpad across outer loop iterations and use if statements to prevent redundant loads on loops inner to those" in opt_lst:
                prompt_text += if_example.PROMPT()
        prompt_text += parents_prompt

        # Now add the actual planning prompt
        for i, opt in enumerate(opt_lst):
            menu_options_text += f"{i+1}. {opt}\n"
        prompt_text += plan_prompt_texts["PRE_OPT_TEXT"] + "\n" + menu_options_text + "\n" + plan_prompt_texts["POST_OPT_TEXT"]
        if force_opt_menu:
            prompt_text += "Explain how to apply <optimization> " + str(force_opt_menu) + ": '" + opt_lst[force_opt_menu-1] + "' to the above code to reduce cycle count, and explain how it will improve performance."
        else:
            prompt_text += plan_prompt_texts["FINAL_TEXT"]

        if prompt_end:
            logger.debug("Appended the following as prompt_end: '%s'", prompt_end)
            prompt_text += "\n" + prompt_end
        return prompt_text

    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None, code_icl_examples: bool = True) -> list[CodeCandidate]:
        if self.pe_dim == 4:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_admm.PROMPT(self.pe_dim)
        elif self.pe_dim == 16:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.PROMPT(self.pe_dim)

        prompt_text += "\nThe original code is as follows:\n"
        prompt_text += candidate.parent.code
        prompt_text += "\nYou are an optimizing compiler generating high-performance Gemmini code. Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan

        # # TODO: For certain optimizations, add more context helping it implement prompt correctly.
        # # e.g. for tiling, add examples of how to tile the code.
        if code_icl_examples:
            if "tiling" in candidate.plan:
                prompt_text += "\n" + tiling_example.PROMPT()
            if " gate" in candidate.plan or "Gate" in candidate.plan:
                # prompt_text += "\n" + if_example_matmul.PROMPT()
                prompt_text += "\n" + if_example.PROMPT()

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += gemmini_rules.PROMPT()

        # prompt_text += "\nRespond with only the optimized code:"
        prompt_text += "Optimized code:"

        return prompt_text

    def _get_combine_candidates_prompt(self, candidates: list[CodeCandidate]) -> str:
        if self.pe_dim == 4:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_admm.PROMPT(self.pe_dim)
        elif self.pe_dim == 16:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.PROMPT(self.pe_dim)
        prompt_text += "\nYou are an optimizing compiler generating high-performance Gemmini code. Let's combine the following optimized code samples to extract the high-performance characteristics of each:\n"
        for i, c in enumerate(candidates):
            prompt_text += f"Sample {i+1}:\n{c.code}\n"

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += gemmini_rules.PROMPT()
        prompt_text += "Optimized code:"
        return prompt_text


class CudaLLMAgent(LLMAgent):
    def __init__(self, model):
        super().__init__(model)

    def _get_convert_to_cuda_menu_options(self) -> list[str]:
        return [
            # "Convert PyTorch code to functional PyTorch code",
            "Convert a PyTorch operation to inline CUDA C++ code",
            # "Use CUDA Graph Capture to eliminate launch overhead",
            # "Convert a PyTorch operation to Triton code",
        ]

    def get_opt_menu_options(self) -> list[str]:
        return [
            # "Convert PyTorch code to functional PyTorch code",
            "Convert an operation to optimized CUDA C++ code",
            "Convert an operation to CUDA C++ code",
            "Convert an operation to optimized Triton code",
            "Reduce PyTorch launch overhead",
            "Use compilation flags like -O3 and --use_fast_math when compiling CUDA code",
            # General Kernel and Memory Optimizations
            "Minimize global memory accesses",
            "Use shared memory to reduce global memory bandwidth usage",
            "Cache redundantly computed data in shared memory",
            "Use pointers to global memory rather than copying to shared memory",
            "Coalesce global memory accesses",
            "Avoid bank conflicts in shared memory",
            "Use registers efficiently; avoid register spilling",
            "Minimize divergent branches within warps",
            "Use CUDA warp-level primitives for synchronization",
            "Fuse kernels when possible to reduce kernel launch overhead",
            "Minimize number of synchronization points",
            "Store more data and reduce at the end rather than using atomic operations",
            "Use grid-stride loops",
            "Tile operations for optimal cache utilization",
            "Use L2 persisting cache window to keep frequently reused tensors resident in L2",
            "Use multiple CUDA streams to overlap computation and data movement",
            # CUDA graph-related Optimizations
            "overlap host-to-device transfers with the CUDA-Graph replay",
            # Thread and Block-Level Optimizations
            "Maximize occupancy without excessive register usage",
            "Choose optimal block sizes (typically multiples of 32 threads)",
            "Use __restrict__ to help compiler with pointer aliasing",
            "Loop unrolling (#pragma unroll)",
            # # Tensor and GEMM Specific Optimizations
            "Use cuBLASLt for Tensor Core GEMM operations",
            "Use cuBLASLt, cuBLAS, or cuDNN for GEMM and convolution operations instead of custom kernels",
            "Use Tensor Cores (e.g. wmma APIs) for mixed precision acceleration (FP16, TF32, INT8)",
            "Use PyTorch's tensor core APIs (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.amp) to enable Tensor Cores",
            "Use lower precision (e.g. bfloat16, float16, float8_e4m3fn) for computations",
            "Quantize weights or activations where accuracy permits (e.g. bfloat16)",
            "Leverage fused operations in cuDNN (e.g. convolution + bias + ReLU)",
            # Memory Transfer Optimizations
            "Overlap computation and data transfer using CUDA streams and asynchronous copies",
            "Use pinned (page-locked) host memory for faster host-device transfers",
            "Minimize host-device transfer frequency",
            # # Algorithmic Optimizations
            "Choose optimal convolution algorithms (FFT, Winograd, implicit GEMM) based on kernel size",
            "Prune unneeded weights for sparse computation",
            "Batch inputs to maximize GPU utilization",
            "Reuse intermediate results where possible (e.g. shared activations)",
            "Vectorize operations by using wider data types",
            "Use Tensor core GEMMs for GEMM-like operations",
            "Convert convolution operations to Tensor core GEMMs",
            # "Convert to a lower precision",
            # From CUDA-L1
            "Skip computation when data-dependent execution encounters zero values or a branch that will never be taken",
            "Ensure data is stored in contiguous memory blocks",
            "Arrange data access patterns to maximize memory bandwidth and minimize latency through techniques like shared memory usage, coalesced global memory access, and memory padding",
            "Memory Coalescing: optimize CUDA kernel performance by ensuring threads in the same warp access contiguous memory locations",
            "Pre-allocate input and output tensors during graph initialization and reuse them",
            "Merge low-level operations",
            "Merge high-level operations",
            "Reorder operations or blocks of operations",
            "Hoist redundant operations out of loops",
            "Substitute operations with equivalent operations that are faster",
            "Double buffering",
            "Pipeline operations to better overlap computation and data movement",
            "Minimize data movement",
            # Other random stuff
            "Use built-in CUDA primitive functions",
            "Call torch:: functions from C++ rather than from Python",
            "Use ATen at:: functions rather than PyTorch functions",
            "Use CUDA graph capture",
            "Use dedicated CUDA streams",
            "Profile the code and capture CUDA graphs in the __init__ function",
            "Simplify operations where possible",
            "Classical compiler optimizations",
            "Any other optimizations that you think are relevant",
        ]

    def analyze_code(self, candidate: CodeCandidate, num_to_gen: int, save_dir: pathlib.Path, save_str: str) -> list[str]:
        return []
    
    def _get_prompt_rules(self, planning: bool, coding: bool) -> str:
        rules = [
            "You will be running the code on an NVIDIA L40S GPU with PyTorch 2.5.0 and CUDA 12.4",
            "The rewritten program should be semantically equivalent to the original program, within a small numerical tolerance.",
            "All generated code should be contained in a single Python file (inline CUDA code is allowed).",
            "Only class ModelNew will be imported during evaluation. Feel free to define other variables, functions, or classes, but make sure they are used by ModelNew.",
            "When using torch.utils.cpp_extension load() or load_inline(), make sure to place C++ code in cpp_sources and CUDA code in cuda_sources.",
            "Do not use the `function` argument of load_inline(), make a PYBIND11 binding instead.",
            "Do not add fallback paths that revert to the original code.",
        ]
        if planning:
            rules.append("Limit the scope of the plan to the selected optimization.")
        if coding:
            rules.append("Wrap the generated code with ```python at the beginning and ``` at the end.")
        rules_text = ""
        for i, rule in enumerate(rules):
            rules_text += f"{i+1}. {rule}\n"
        return rules_text

    def _get_propose_optimizations_prompt(self, candidate: CodeCandidate,
                                          prob: Prob,
                                          force_opt_menu: int, 
                                          prompt_end: str, 
                                          analysis: str, 
                                          shuffle_opts: bool, 
                                          give_score_feedback: float,
                                          give_util_feedback: float,
                                          give_spad_acc_feedback: float,
                                          include_ancestors: bool,
                                          plan_icl_examples: bool,
                                          cur_iter: int,
                                          num_iters: int,
                                          dropout_menu_options: float,
                                          translate: bool,
                                         ) -> list[str]:
        # Select which menu options will appear
        if translate:
            opt_lst = self._get_convert_to_cuda_menu_options()
        else:
            opt_lst = self.get_opt_menu_options()
            if dropout_menu_options < 1 and not force_opt_menu:
                opt_lst = [opt for opt in opt_lst if random.random() < dropout_menu_options]
            if shuffle_opts:
                random.shuffle(opt_lst)
        include_score_feedback = random.random() < give_score_feedback

        parents_prompt = ""
        cur_cand = candidate
        while cur_cand is not None:
            # Go up to each parent and append to front of prompt
            if include_score_feedback and (cur_cand.score is not None):
                parents_prompt = f"The latency of this code was {cur_cand.score} seconds.\n" + parents_prompt
            if not include_ancestors:
                parents_prompt = "\nThe original unoptimized code was:\n" + cur_cand.code + "\n" + parents_prompt
                break # No need to go up past the immediate parent
            elif cur_cand.plan is not None:
                parents_prompt = "\nNext, we applied this plan to the code:\n" + cur_cand.plan + "\nThe generated code was:\n" + cur_cand.code + "\n" + parents_prompt
            else:
                parents_prompt = "The original unoptimized code was:\n" + cur_cand.code + "\n" + parents_prompt
            cur_cand = cur_cand.parent

        if analysis:
            parents_prompt += "\n" + analysis

        # Initialize the prompt with the parents info
        # prompt_text = tensor_examples.PROMPT()
        prompt_text = parents_prompt

        # Now add the actual planning prompt
        menu_options_text = ""
        for i, opt in enumerate(opt_lst):
            menu_options_text += f"{i+1}. {opt}\n"
        prompt_text += """Please carefully review the program to identify any inefficiencies. 
Speedup can be increased by using the following optimizations:
<optimizations>: \n""" + menu_options_text + "\n"
        
        prompt_text += "You are an expert GPU performance engineer generating high-performance PyTorch and CUDA code. "
        if force_opt_menu:
            prompt_text += "Explain how to apply <optimization> " + str(force_opt_menu) + ": '" + opt_lst[force_opt_menu-1] + "' to the above code to reduce execution time, and explain how it will improve performance."
        else:
            if random.random() < 0.1:
                prompt_text += "Invent an optimization different from the <optimizations> above to address the inefficiencies of the above code and reduce its execution time, and explain how it will improve performance."
            else:
                prompt_text += "Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its execution time. The plan should be specific to this code and explain how to change it."

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=True, coding=False)

        if prompt_end:
            logger.debug("Appended the following as prompt_end: '%s'", prompt_end)
            prompt_text += "\n" + prompt_end
        return prompt_text


    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None, code_icl_examples: bool = True) -> list[CodeCandidate]:
        prompt_text = ""
        if "tensor core" in candidate.plan.lower():
            if random.random() < 0.5:
                prompt_text += tensor_examples.PROMPT() + "\n"
        prompt_text += "\nThe original code is as follows:\n```python\n"
        prompt_text += candidate.parent.code
        prompt_text += "\n```\nYou are an expert GPU performance engineer generating high-performance PyTorch and CUDA code. Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan

        # # # TODO: For certain optimizations, add more context helping it implement prompt correctly.
        # # # e.g. for tiling, add examples of how to tile the code.
        # if code_icl_examples:
        #     if "tiling" in candidate.plan:
        #         prompt_text += "\n" + tiling_example.PROMPT()
        #     if " gate" in candidate.plan or "Gate" in candidate.plan:
        #         # prompt_text += "\n" + if_example_matmul.PROMPT()
        #         prompt_text += "\n" + if_example.PROMPT()

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=False, coding=True)

        prompt_text += "Optimized code:"

        return prompt_text

    def _get_combine_candidates_prompt(self, candidates: list[CodeCandidate]) -> str:
        prompt_text = "You are an expert GPU performance engineer generating high-performance PyTorch and CUDA code. Let's combine the following optimized code samples to extract the high-performance characteristics of each:\n"
        for i, c in enumerate(candidates):
            prompt_text += f"Sample {i+1}:\n{c.code}\n"

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=False, coding=True)
        prompt_text += "Optimized code:"
        return prompt_text

class TrnLLMAgent(LLMAgent):
    def __init__(self, model):
        super().__init__(model)
        self.nki_isa_generator = NkiIsaGenerator()

    def __repr__(self):
        return f"TrnLLMAgent({self.llm_client.model})"

    def _get_convert_to_nki_menu_options(self) -> list[str]:
        return [
            "convert non-NKI code into NKI code",
            "move a non-NKI transpose into the NKI kernel",
            "fuse multiple NKI kernels into a single kernel",
        ]

    def get_opt_menu_options(self, prob: Prob):
        """Get optimization menu options for NKI/Trainium kernels"""
        return [
            "eliminate loads and stores as much as possible, keeping data in SBUF/PSUM instead",
            "minimize data movement",
            "overlap data movement and compute",
            "improve data layout and access patterns",
            "loop reordering and restructuring",
            "inline a function so it can be more easily optimized and fused",
            "skip computation when it is not needed (e.g. it is completely masked out)",
            "fuse loops (reordering if necessary)",
            "increase reuse by keeping data in SBUF across outer loop iterations",
            "hoist redundant operations out of loops",
            "delay softmax division until after all reductions are complete",
            "Perform nc_matmul on large contiguous blocks within its own affine_range loop to maximize compute throughput.",
            "Group nc_matmul calls into larger blocks, organizing inputs ahead of time, to maximize Tensor Engine utilization.",
            "do operations in lower precision such as nl.bfloat16",
            "double buffering",
            "fuse multiple instructions into one, for example by doing reduction inside nisa.activation()",
            "software pipelining",
            "keep data in SBUF/PSUM instead of storing to and loading from HBM",
            "stronger tiling for contraction / moving-free split",
            "reorder operations to improve locality",
            "fuse dependent operations",
            "fuse operations into a single loop so intermediate data does not need to be stored to and loaded from HBM",
            "fuse loops that iterate over the same dimension to improve intermediate data reuse",
            "allocate a larger tile in SBUF so we can keep data in it rather than storing to and loading from HBM",
            "allocate buffers in lower precision such as nl.bfloat16",
            "downcast to lower precision during operations that take dtype as an argument",
            "keep data in the same layout to avoid transpose operations",
            "eliminate intermediate tensor materialization by using in-place operations (storing the output in the same buffer as the input)",
            "use the streaming softmax with running max and scaling trick",
            "optimize accumulation patterns in PSUM",
            "optimize reduction by fusing tile-wise reductions with transformation passes",
            "Load larger blocks of data to increase SBUF data reuse and reduce memory traffic",
            "Add additional loop levels so larger blocks of data can be loaded (multi-level tiling)",
            "Combine adjacent tiles into contiguous blocks before nl.store() to maximize memory throughput.",
            "Scan carry-over to parallelize the scan operation",
            "Hoist nl.load() operations for reused data (e.g., LHS tiles) outside inner loops to reduce redundant HBM→SBUF transfers.",
            "Kernel Fusion via SBUF residency",
            "Modify one particular parameter to maximize performance",
            "Target the specific data shapes and shapes of the input and output tensors to maximize performance",
            "Use constant 128-iteration loops for Vector Engine instructions to coalesce them",
            # "Replace general-purpose code with faster specialized instructions",
            # "transpose inside the NKI kernel",
            # "move non-NKI code into the NKI kernel",
            "Overlap execution across compute engines through pipelining",
            "Swap stationary and moving tensors in nc_matmul",
            "Make the vector the moving tensor in nc_matmul",
            "Use conditional execution instead of masking, or vice versa",
            "Simplify or eliminate any unnecessary code"
            "Other methods not listed here.",
        ]

    def _get_prompt_rules(self, planning: bool, coding: bool) -> str:
        rules = ["The rewritten program should be semantically equivalent to the original program, within a small numerical tolerance.",
                #  "Use proper NKI syntax and decorators (@nki.jit).",
                #  "Ensure proper memory buffer usage (sbuf, psum, hbm).",
                 "Maintain correct tensor shapes and indexing patterns. Remember not to index with affine_range loop variables. Avoid loop carried dependencies.",
                 "The following imports have already been run: import neuronxcc.nki as nki; import neuronxcc.nki.isa as nisa; import neuronxcc.nki.language as nl; import neuronxcc.nki.typing as nt; import numpy as np;",
                 "nisa and nl may have similar functions (for example, nisa.nc_matmul() and nl.matmul()), but they may have different arguments or functionality. Make sure to follow the documentation above."
                #  "Try to use the nki.language and nki.isa functions defined above.",
                ]
        if planning:
            rules.append("Limit the scope of the plan to the selected optimization.")
            if random.random() < 0.4:
                rules.append("Limit the scope of the plan so that the rewritten program is still correct.")
            elif random.random() < 0.3:
                rules.append("Plans can be highly targeted to one particular part of the code.")
            rules.append("Do not count out any of the <optimizations> unless they are clearly irrelevant to the code.")
        if coding:
            rules.append("Optimize the test() function and do not change its name.")
            rules.append("Wrap the generated code with ```python at the beginning and ``` at the end.")
        rules.append("Ensure that loop dependencies are not violated inside affine_range loops.")
        # rules.append("You are optimizing for constant shapes: x.shape = (1, 1, 2048), post_attention_layernorm_weight.shape = (2048,), up_proj_weight.shape = (8192, 2048), gate_proj_weight.shape = (8192, 2048), down_proj_weight.shape = (2048, 8192), output.shape = (1, 2048). Make sure to take advantage of these shapes, especially the fact that x is a vector.")
        # rules.append("You are optimizing for constant shapes: x.shape = (1, 1, 2048), up_proj_weight.shape = (2048, 4096), gate_proj_weight.shape = (2048, 4096), down_proj_weight.shape = (4096, 2048), output.shape = (1, 2048). Make sure to take advantage of these shapes, especially the fact that x is a vector.")
        # rules.append("You are optimizing for constant shapes: x.shape = (1, 1, 2048), up_w.shape = (2048, 4096), gate_w.shape = (2048, 4096), down_w.shape = (4096, 2048), output.shape = (1, 2048). Make sure to take advantage of these shapes.")
        # rules.append("You are optimizing for constant shapes: R = 1, H = 2048, U = 8192, D = 2048. Make sure to take advantage of these shapes, especially the fact that x is a vector.")
        # rules.append("You are optimizing for constant shapes. Make sure to take advantage of these shapes.")
        # rules.append("You are optimizing for constant shapes: Q.shape = (1, 16, 1, 64), K.shape = (1, 4, 1, 64), V.shape = (1, 4, 1, 64), past_key_value[0].shape = (1, 4, 512, 64), past_key_value[1].shape = (1, 4, 512, 64), attention_mask.shape = (1, 1, 1, 512). Make sure to take advantage of these shapes.")
        # rules.append("You are optimizing for constant shapes: Q.shape = (1, 16, 1, 64), K.shape = (1, 4, 1, 64), V.shape = (1, 4, 1, 64), past_k.shape = (1, 4, 512, 64), past_v.shape = (1, 4, 512, 64), attention_mask.shape = (1, 1, 1, 512). Make sure to take advantage of these shapes.")
        # rules.append("You are optimizing for constant shapes: hidden_states.shape = (1, 1, 2048), lm_head_weight.shape = (2048, 64128). Make sure to take advantage of these shapes.")
        # rules.append("You are optimizing for constant shapes: lhsT.shape = (K, M) = (2048, 64128), rhs.shape = (K, N) = (2048, 1). Make sure to take advantage of these shapes.")
        # rules.append("You are optimizing for constant shapes: Q.shape = (32, 16, 1, 64), K.shape = (32, 4, 1, 64), V.shape = (32, 4, 1, 64), past_key_value[0].shape = (32, 4, 512, 64), past_key_value[1].shape = (32, 4, 512, 64), attention_mask.shape = (32, 16, 1, 512). Make sure to take advantage of these shapes.")
        rules.append("You are optimizing for constant shapes: x.shape = (32, 1, 2048), up_proj_weight.shape = (2048, 4096), gate_proj_weight.shape = (2048, 4096), down_proj_weight.shape = (4096, 2048). Make sure to take advantage of these shapes.")
        # rules.append("IMPORTANT: Minimize the amount of non-NKI code.")
        prompt_text = ""
        for i, rule in enumerate(rules):
            prompt_text += f"{i+1}. {rule}\n"
        return prompt_text

    def _get_propose_optimizations_prompt(self, candidate: CodeCandidate,
                                          prob: Prob,
                                          force_opt_menu: int, 
                                          prompt_end: str, 
                                          analysis: str, 
                                          shuffle_opts: bool, 
                                          give_score_feedback: float,
                                          give_util_feedback: float,
                                          give_spad_acc_feedback: float,
                                          include_ancestors: bool,
                                          plan_icl_examples: bool,
                                          cur_iter: int,
                                          num_iters: int,
                                          dropout_menu_options: float,
                                          translate: bool,
                                         ) -> str:
        # Select which menu options will appear
        menu_options_text = ""
        if translate:
            opt_lst = self._get_convert_to_nki_menu_options()
        else:
            opt_lst = self.get_opt_menu_options(prob)
            if dropout_menu_options < 1 and not force_opt_menu:
                opt_lst = [opt for opt in opt_lst if random.random() < dropout_menu_options]
            if shuffle_opts:
                random.shuffle(opt_lst)
        include_score_feedback = random.random() < give_score_feedback

        parents_prompt = ""
        cur_cand = candidate
        while cur_cand is not None:
            # Go up to each parent and append to front of prompt
            if include_score_feedback and (cur_cand.score is not None):
                parents_prompt = f"The latency of this code was {cur_cand.score} ms.\n" + parents_prompt
            if not include_ancestors:
                parents_prompt = "\nThe original unoptimized code was:\n```\n" + cur_cand.code + "\n```\n" + parents_prompt
                break # No need to go up past the immediate parent
            elif cur_cand.plan is not None:
                parents_prompt = "\nNext, we applied this plan to the code:\n" + cur_cand.plan + "\nThe generated code was:\n" + cur_cand.code + "\n" + parents_prompt
            else:
                parents_prompt = "\nThe original unoptimized code was:\n```\n" + cur_cand.code + "\n```\n" + parents_prompt
            cur_cand = cur_cand.parent

        if analysis:
            parents_prompt += "\n" + analysis

        # Initialize the prompt with NKI context
        prompt_text = "The NKI (Neuron Kernel Interface) is used for writing high-performance kernels on AWS Trainium and Inferentia chips.\n"
        prompt_text += self.nki_isa_generator.generate_isa(prob)
        
        prompt_text += parents_prompt

        # Now add the actual planning prompt
        for i, opt in enumerate(opt_lst):
            menu_options_text += f"{i+1}. {opt}\n"
        
        prompt_text += "Please carefully review the NKI code to identify any inefficiencies. "
        prompt_text += "Performance can be improved by using the following optimizations:\n"
        prompt_text += "<optimizations>:\n" + menu_options_text + "\n"
        
        if force_opt_menu:
            prompt_text += "Explain how to apply <optimization> " + str(force_opt_menu) + ": '" + opt_lst[force_opt_menu-1] + "' to the above code to reduce execution time, and explain how it will improve performance."
        else:
            prompt_text += "You are an expert NKI performance engineer generating high-performance Trainium/Inferentia kernels. "

            # prompt_text += "Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its execution time. The plan should be specific to this code and explain how to change it."
            # TODO make it a parameter
            choose_or_invent = random.random()
            if choose_or_invent < 0.1 and not translate:
                # Prompt to invent a new optimization inspired by the <optimizations>
                prompt_text += "Invent a new optimization inspired by the <optimizations> to apply to the above code to reduce execution time, and explain how it will improve performance."
            elif choose_or_invent < 0.2 and not translate:
                # Prompt to invent a new optimization different from the <optimizations>
                prompt_text += "Think of a new optimization different from the <optimizations> to apply to the above code to reduce execution time, and explain how it will improve performance."
            else:
                prompt_text += "Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its execution time."

        prompt_text += " The plan should be specific to this code and explain how to change it."
        # # TODO make it a parameter
        # if random.random() < 0.5:
        #     prompt_text += " The plan should be specific to this code and explain how to change it."
        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=True, coding=False)

        if prompt_end:
            logger.debug("Appended the following as prompt_end: '%s'", prompt_end)
            prompt_text += "\n" + prompt_end
        return prompt_text

    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None, code_icl_examples: bool = True) -> str:
        prompt_text = "The NKI (Neuron Kernel Interface) is used for writing high-performance kernels on AWS Trainium and Inferentia chips.\n"
        if prob is None:
            raise ValueError("TrnLLMAgent requires prob parameter to be provided")
        prompt_text += self.nki_isa_generator.generate_isa(prob)

        if "fusion" in candidate.plan.lower() or "fuse" in candidate.plan.lower():
            rand_val = random.random()
            if rand_val < 0.15:
                prompt_text += "\n" + fusion_example.PROMPT() + "\n"
            elif rand_val < 0.3:
                prompt_text += "\n" + fusion_example.PROMPT_2() + "\n"

        prompt_text += "The original code is as follows:\n"
        prompt_text += candidate.parent.code
        prompt_text += "\nYou are an expert NKI performance engineer generating high-performance Trainium/Inferentia kernels. "
        prompt_text += "Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan

        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True)
        prompt_text += "\nOptimized NKI code:"

        return prompt_text

    def _get_combine_candidates_prompt(self, candidates: list[CodeCandidate]) -> str:
        prompt_text = "The NKI (Neuron Kernel Interface) is used for writing high-performance kernels on AWS Trainium and Inferentia chips.\n"
        prompt_text += "You are an expert NKI performance engineer generating high-performance Trainium/Inferentia kernels. "
        prompt_text += "Let's combine the following optimized NKI code samples to extract the high-performance characteristics of each:\n"
        for i, c in enumerate(candidates):
            prompt_text += f"Sample {i+1}:\n{c.code}\n"

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=False, coding=True)
        prompt_text += "\nOptimized combined NKI code:"
        return prompt_text

    def _get_reimplement_failed_code_prompt(self, candidate: CodeCandidate, prob: Prob = None) -> str:
        """
        Generate a prompt to reimplement failed code based on stdout/stderr feedback.
        """
        if prob is None:
            raise ValueError("TrnLLMAgent requires prob parameter to be provided")

        prompt_text = "The NKI (Neuron Kernel Interface) is used for writing high-performance kernels on AWS Trainium and Inferentia chips.\n"
        prompt_text += self.nki_isa_generator.generate_isa(prob)

        # prompt_text += "The original code is as follows:\n"
        # prompt_text += candidate.parent.code
        prompt_text += "\n\nYou are an expert NKI performance engineer generating high-performance Trainium/Inferentia kernels. "
        # prompt_text += "We attempted to optimize the original code based on the following plan:\n"
        # prompt_text += candidate.plan
        # prompt_text += "\n\nThe generated code was:\n"
        prompt_text += "\nThe code was:\n"
        prompt_text += candidate.code
        
        # Add error information
        prompt_text += "\n\nHowever, the code failed with the following output:\n"
        if candidate.stderr:
            prompt_text += "=== STDERR ===\n"
            # Limit the length of each line to 400 characters
            stderr_lines = candidate.stderr.split("\n")
            stderr_lines = [line[:400] for line in stderr_lines]
            stderr_lines = "\n".join(stderr_lines)
            prompt_text += stderr_lines
            prompt_text += "\n"
        if candidate.stdout:
            prompt_text += "=== STDOUT ===\n"
            stdout_lines = candidate.stdout.split("\n")
            stdout_lines = [line[:400] for line in stdout_lines]
            stdout_lines = "\n".join(stdout_lines)
            prompt_text += stdout_lines
            prompt_text += "\n"
        
        prompt_text += "\nPlease fix the code to address the errors while still applying the optimization plan. "
        prompt_text += "Make sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True)
        prompt_text += "\nFixed and optimized NKI code:"

        return prompt_text
