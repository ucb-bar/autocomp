def PROMPT():
    rules = [
        "The rewritten program should be semantically equivalent to the original program",
        "Limit the scope of the plan to the selected optimization",
        "All code must be inside the test() function",
        "Do not use C preprocessing directives (#ifdef, #define, etc.)",
        "If modifying loops, modify other related loop bounds and adjust address and index calculations to ensure the code is still correct",
        "If increasing loaded tile size, ensure that data is spread throughout the scratchpad across all relevant dimensions",
        "If loading across new dimensions, add the loop indices of those dimensions to scratchpad address calculations",
        "If increasing loaded tile size, update preload and compute instructions to match the new data layout",
        "If increasing loaded tile size, update base scratchpad addresses to fit new tile size",
    ]
    rules_str = """
Rules:
"""
    for i, rule in enumerate(rules):
        rules_str += f"{i+1}. {rule}\n"
    return rules_str