REVISE_SYSTEM_FORMAT = f"""
You are a helpful assistant. You will be given a STATEMENT and a RESPONSE. The STATEMENT may contain vague references.

Vague references include but are not limited to:
- Pronouns (e.g., "his", "they", "her")
- Unknown entities (e.g., "this event", "the research", "the invention")
- Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)

Your task is to revise the STATEMENT by replacing the vague references with the proper entities from the RESPONSE that they are referring to. \
Here are the instructions:

Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities \
from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT. \
For example, given the response "Titanic is a movie starring Leonardo \
DiCaprio," the statement "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your \
reasoning. As part of your reasoning, be sure to identify the subjects in the \
STATEMENT and determine whether they are vague references. If they are vague \
references, identify the proper entity that they are referring to and be sure \
to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in \
a markdown code block.
7. Your task is to do this for the STATEMENT and RESPONSE under "Your Task".
""".strip()

FIRST_SEARCH_SYSTEM = f"""\
Instructions:
1. You will be given a STATEMENT. Your goal is to try to find evidence \
that either supports or does not support the factual accuracy of the given STATEMENT.
2. To do this, you are allowed to issue ONE Google Search query that you think \
will allow you to find useful evidence.
3. Your query should aim to obtain information that is useful for determining \
the factual accuracy of the given STATEMENT.
4. Format your final query by putting it in a markdown code block.
""".strip()

SUPPORTED_LABEL = 'Supported'
NOT_SUPPORTED_LABEL = 'Not Supported'
FINAL_ANSWER_SYSTEM = f"""\
Instructions:
1. You will be given a STATEMENT and some KNOWLEDGE points.
2. Determine whether the given STATEMENT is supported by the given KNOWLEDGE. \
The STATEMENT does not need to be explicitly supported by the KNOWLEDGE, but \
should be strongly implied by the KNOWLEDGE.
3. Before showing your answer, think step-by-step and show your specific \
reasoning. As part of your reasoning, summarize the main points of the \
KNOWLEDGE.
4. If the STATEMENT is supported by the KNOWLEDGE, be sure to show the \
supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your \
final answer based on your reasoning and the STATEMENT.
6. Your final answer should be either "{SUPPORTED_LABEL}" or \
"{NOT_SUPPORTED_LABEL}". Wrap your final answer in square brackets.
""".strip()