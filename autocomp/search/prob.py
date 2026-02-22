import pathlib

from autocomp.common import TESTS_DIR

class Prob():
    def __init__(self, prob_type: str, prob_id: int, *, test_file: pathlib.Path | str | None = None, sol_file: pathlib.Path | str | None = None):
        self.prob_type = prob_type
        self.prob_id = prob_id
        self.test_file = pathlib.Path(test_file) if test_file else None
        self.sol_file = pathlib.Path(sol_file) if sol_file else None
        self.tests: list[Test] = []

        if not self.test_file:
            test_dir = TESTS_DIR
            test_files = list((test_dir / prob_type).glob(f"test{prob_id}.c"))
            for test_file_path in test_files:
                self.tests.append(Test(test_file_path))

        # perf_test_files = list((test_dir / prob_type).glob(f"test{prob_id}_perf.c"))
        # for perf_test_file in perf_test_files:
        #     self.perf_tests.append(Test(perf_test_file))

    def __repr__(self):
        return f"Prob({self.prob_type}, {self.prob_id})"

class Test():
    def __init__(self, test_file: pathlib.Path):
        self.test_file = test_file

    def get_test_code(self, sol_code_strs: list[str], check_correct: bool=True, error_on_incorrect: bool=True, repeat_iters=None) -> str:
        """
        Returns the code for this test.
        args:
        - sol_code_strs: list of strings, each string is a solution code snippet
        - check_correct: bool, whether to check correctness
        - error_on_incorrect: bool, whether to error on incorrectness
        - repeat_iters: int, number of iterations to repeat the test; if None, default number of repeats is used
        """
        combined_sol_code_str = " "*4 + "int generated_implementation_start_cycle;\nint generated_implementation_end_cycle;\n"
        for code_str in sol_code_strs:
            code_lines = [
                "gemmini_flush(0);",
                "fence();",
                "generated_implementation_start_cycle = read_cycles();",
                "{",
                code_str,
                "}",
                "fence();",
                "generated_implementation_end_cycle = read_cycles();",
            ]
            if check_correct:
                code_lines.append("if (!full_is_equal(OUTPUT_MATRIX_NAME, gold)) {")
                if error_on_incorrect:
                    code_lines.append('printf("Incorrect result\\n");')
                    code_lines.append('exit(1);')
                else:
                    code_lines.append('printf("Generated implementation latency: 99999999999 cycles\\n");')
                code_lines.extend([
                    "} else {",
                    'printf("Generated implementation latency: %d cycles\\n", generated_implementation_end_cycle - generated_implementation_start_cycle);',
                    "}",
                ])
            else:
                code_lines.append('printf("Generated implementation latency: %d cycles\\n", generated_implementation_end_cycle - generated_implementation_start_cycle);')
            for code_line in code_lines:
                combined_sol_code_str += " "*4 + code_line + "\n"
        
        modified_test_code = self.modify_test_code(combined_sol_code_str)
        lines = modified_test_code.splitlines()

        for line_i, line in enumerate(lines):
            if repeat_iters is not None:
                if "#define REPEAT_TEST_ITERS" in line:
                    lines[line_i] = f"#define REPEAT_TEST_ITERS {repeat_iters}"
            if not check_correct:
                if "#define RUN_BASELINE_CODE" in line:
                    lines[line_i] = "#define RUN_BASELINE_CODE 0"

        return "\n".join(lines)

    def modify_test_code(self, code_str: str) -> str:
        """
        Inserts a given string of code into the test file between markers.

        This function reads the content of a file and looks for specific markers
        indicating where to substitute the provided code string. It inserts the
        code string right after the start marker and continues to copy the new
        content until it reaches the end marker. The modified content is then written
        back to the same file.

        Parameters:
        - code_str (str): The string of code to be inserted into the file.

        Returns:
        None
        """
        with open(self.test_file, "r") as file:
            content = file.readlines()

        new_content = []
        substitute = False
        for line in content:
            if "// SUBSTITUTE HERE" in line:
                substitute = True
                new_content.append(line)
                new_content.append(code_str + "\n")
            elif "// SUBSTITUTE END" in line:
                substitute = False
            if not substitute or "// SUBSTITUTE END" in line:
                new_content.append(line)

        return "".join(new_content)

