from prompts.gemmini import gemmini_rules

def PROMPT(DIM):
    texts = {
        "PRE_OPT_TEXT": f"""Please carefully review the program to identify any inefficiencies. 
Cycles can be reduced by using the following optimizations:
<optimizations>: """
    }
    
    # POST_OPT_TEXT = f"""The rewritten program should be semantically equivalent to the original program. Systolic array size is {DIM}x{DIM} (DIM={DIM})."""
    POST_OPT_TEXT = f"""Systolic array size is {DIM}x{DIM} (DIM={DIM})."""

    if DIM == 4:
        POST_OPT_TEXT += " Each element is 4 bytes."
    elif DIM == 16:
        POST_OPT_TEXT += " Elements in the scratchpad are 1 byte, and elements in the accumulator are 4 bytes."

    POST_OPT_TEXT += f" The scratchpad size is 256KB and the accumulator size is 64KB."
    POST_OPT_TEXT += gemmini_rules.PROMPT()
    texts["POST_OPT_TEXT"] = POST_OPT_TEXT

    texts["FINAL_TEXT"] = "You are an optimizing compiler that generates high-performance Gemmini code. Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its cycle count. The plan should be specific to this code and explain how to change it."

    return texts