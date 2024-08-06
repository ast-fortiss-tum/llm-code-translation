from codetrans.postprocessing import comment_syntax


direct_translation_template = (
    """Translate the following source code from {source_pl} to {target_pl}: <begin_code> {source_code} <end_code>"""
)


via_description_translation_template_1 = """You are a skilled software developer proficient in multiple programming languags. The goal is to re-write source code in a different programming language. Describe the following source code written in {source_pl}: 

Source code:
```
{source_code}
```

### Response:"""

via_description_translation_template_2 = """You are a skilled software developer proficient in multiple programming languages. Your task is to write a program following the given description. Below is the description of the source code that you should re-write into {target_pl} programming language. You must respond with the {target_pl} output code only. 

Description:
{description} 

### Response:
```"""

########################################## Use an example for giving a description ##########################################

# source code in Python
example_source_code = """
num1 = 1.5
num2 = 6.3
sum = num1 + num2
print(f'The sum of {0} and {1} is {2}'.format(num1, num2, sum))
"""

example_description = """This program adds two numbers 1.5 and 6.3.
1. The numbers are defined as variables.
2. The numbers are added.
3. The sum is printed in the format: The sum of <number 1> and <number 2> is <sum>. 
"""

via_description_translation_template_1_w_example = """You are a skilled software developer proficient in multiple programming languages. The goal is to re-write source code in a different programming language. Describe the source code written in {source_pl}.

Here is an example how to describe code. 

Example source code:
```
{example_source_code}
```

Description for the example source code:
{example_description}

Describe the following source code written in {source_pl}:

Source code:
```
{source_code}
```

### Response:"""

translation_allowing_questions_template = """Translate the following source code from {source_pl} to {target_pl}: <begin_code> {source_code} <end_code> If you have questions about the code, ask them."""


##########################################  prompt templates from related work ##########################################
lit_paper_naive_template = """{source_pl} Code:

{source_code}

Translate the above {source_pl} code to {target_pl}.

{target_pl} Code:

"""

# prompt templates from the "LLM translation output paper"
vanilla_template = """You are a skilled software developer proficient in multiple programming languages. Your task is to re-write the input source code. Below is the input source code written in {source_pl} that you should re-write into {target_pl} programming language. You must respond with the {target_pl} output code only.

Source code:
{source_code}

### Response:"""

controlled_template = """You are a skilled software developer proficient in multiple programming languages. Your task is to re-write the input source code. Below is the input source code written in {source_pl} that you should re-write into {target_pl} programming language. You must respond with the {target_pl} output code only.

Source code:
```
{source_code}
```

### Response:
```"""

##########################################

controlled_md_template = """You are a skilled software developer proficient in multiple programming languages. Your task is to re-write the input source code. Below is the input source code written in {source_pl} that you should re-write into {target_pl} programming language. You must respond with the {target_pl} output code only placed in a Markdown code block.
A Markdown code block for {target_pl} looks like this:
```{target_pl_md}
{target_pl_comment} Example code
```

Source code:
```
{source_code}
```

### Response:
```{target_pl_md}"""

controlled_md_no_system_template = """Your task is to re-write the input source code. Below is the input source code written in {source_pl} that you should re-write into {target_pl} programming language. You must respond with the {target_pl} output code only placed in a Markdown code block.
A Markdown code block for {target_pl} looks like this:
```{target_pl_md}
{target_pl_comment} Example code
```

Source code:
```
{source_code}
```

### Response:
```{target_pl_md}"""


direct_md_template = """Translate the given source code from {source_pl} to {target_pl}. Below is the input source code written in {source_pl} that you should re-write into {target_pl} programming language. You must respond with the {target_pl} output code only placed in a Markdown code block.
A Markdown code block for {target_pl} looks like this:
```{target_pl_md}
{target_pl_comment} Example code
```

Source code:
```
{source_code}
```

### Response:"""

short_md_template = """Translate the given source code from {source_pl} to {target_pl}. You must respond with the {target_pl} output code only placed in a Markdown code block.
A Markdown code block for {target_pl} looks like this:
```{target_pl_md}
{target_pl_comment} Example code
```

Source code:
```
{source_code}
```

### Response:"""


########################################## Iterative Approach ##########################################

compile_runtime_error_evalplus_template = """
You are a skilled software developer proficient in multiple programming languages. Your task is to re-write the input source code. Below is the input source code written in {source_pl} that you should re-write into {target_pl} programming language. You must respond with the {target_pl} output code only placed in a Markdown code block.

Source code:
```
{source_code}
```

### Response:
```{target_pl_md}
{translated_code}
```

Executing your generated code gives the following error because it is syntactically incorrect.

Error message:

{stderr}

Please re-generate your response and translate the above {source_pl} code to {target_pl}. You must respond with the {target_pl} output code only placed in a Markdown code block. Make sure your generated code is syntactically correct.

### Response:
```{target_pl_md}"""


compile_runtime_error_template = """You are a skilled software developer proficient in multiple programming languages. Your task is to re-write the input source code. Below is the input source code written in {source_pl} that you should re-write into {target_pl} programming language. You must respond with the {target_pl} output code only placed in a Markdown code block.

Source code:
```
{source_code}
```

### Response:
```{target_pl_md}
{translated_code}
```

Executing your generated code gives the following error because it is syntactically incorrect.

Error message:

{stderr}

Please re-generate your response and translate the above {source_pl} code to {target_pl}. You must respond with the {target_pl} output code only placed in a Markdown code block. Make sure your generated code is syntactically correct.

### Response:
```{target_pl_md}"""


test_failure_evalplus_template = """You are a skilled software developer proficient in multiple programming languages. Your task is to re-write the input source code. Below is the input source code written in {source_pl} that you should re-write into {target_pl} programming language. You must respond with the {target_pl} output code only placed in a Markdown code block.

Source code:
```
{source_code}
```

### Response:
```{target_pl_md}
{translated_code}
```

Executing your generated code gives the following test failure:

{stderr}

Please re-generate your response and translate the above {source_pl} code to {target_pl}. You must respond with the {target_pl} output code only placed in a Markdown code block and keep the method signature from the incorrect translation. Make sure your generated code is syntactically correct.

### Response:
```{target_pl_md}"""


test_failure_io_based_template = """You are a skilled software developer proficient in multiple programming languages. Your task is to re-write the input source code. Below is the input source code written in {source_pl} that you should re-write into {target_pl} programming language. You must respond with the {target_pl} output code only placed in a Markdown code block.

Source code:
```
{source_code}
```

### Response:
```{target_pl_md}
{translated_code}
```

Executing your generated code gives the following output:
{generated_output}

instead of the following expected output:
{test_outputs}

Please re-generate your response and translate the above {source_pl} code to {target_pl}. You must respond with the {target_pl} output code only placed in a Markdown code block and keep the method signature from the incorrect translation. Make sure your generated code is syntactically correct. Your generated {target_pl} code should take the following input and generate the expected output:

Input:
{test_inputs}

Expected Output:
{test_outputs}

### Response:
```{target_pl_md}"""
