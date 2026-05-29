import json
from pathlib import Path
import re
from typing import Any, Union
import logging
import regex
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# nltk.download('punkt_tab')
# nltk.download('punkt')

GOOGLE_TO_FACTSCORE_LABEL = {
    "Supported": "True",
    "Not Supported": "False",
}

################################################################################
#                             JSON FILE OPERATION                              #
################################################################################

def jsonlines_load(fname: str | Path):
    with open(fname, 'r') as f:
        return [json.loads(line) for line in f]


def jsonlines_dump(fname: str | Path, data: Union[dict, list]):
    try:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        with open(fname, 'a+') as f:
            if isinstance(data, dict):
                f.write(json.dumps(data)+'\n')
            elif isinstance(data, list):
                for d in data:
                    f.write(json.dumps(d)+'\n')

    except (FileNotFoundError, FileExistsError) as e:
        print(f'Error: {e}')
        print(f'Could not write to {fname}')


def output_stem(input_path: str | Path) -> str:
    input_path = Path(input_path)
    if input_path.name.endswith(".jsonl"):
        return input_path.name[: -len(".jsonl")]
    return input_path.stem


def default_factscore_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    if input_path.name.endswith(".jsonl"):
        return input_path.with_name(input_path.name.replace(".jsonl", "_factscore_output.json"))
    return input_path.with_name(f"{input_path.stem}_factscore_output.json")


def prepare_output_path(path: str | Path, overwrite: bool = False, append: bool = False):
    path = Path(path)
    if overwrite and append:
        raise ValueError("Use only one of overwrite or append.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return
    if path.stat().st_size == 0:
        path.unlink()
    elif overwrite:
        path.unlink()
    elif not append:
        raise FileExistsError(
            f"{path} already exists. Use overwrite=True to replace it or append=True."
        )


def prepare_output_paths(paths: list[str | Path], overwrite: bool = False, append: bool = False):
    if overwrite and append:
        raise ValueError("Use only one of overwrite or append.")
    for path in paths:
        prepare_output_path(path, overwrite=overwrite, append=append)


def clear_files(paths: list[str | Path]):
    for path in paths:
        path = Path(path)
        if path.exists():
            path.unlink()


def slice_rows(rows: list[dict[str, Any]], start: int, end: int):
    end_index = len(rows) if end == -1 else end
    return rows[start:end_index]


def count_items(tasks: list[dict[str, Any]], key: str) -> int:
    return sum(len(task.get(key, [])) for task in tasks)


def normalize_bool_label(value: Any) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if lowered == "true":
        return "True"
    if lowered == "false":
        return "False"
    raise ValueError(f"Unsupported boolean label: {value!r}")


def is_false_label(value: Any) -> bool:
    return normalize_bool_label(value) == "False"


def normalize_google_label(value: Any) -> str:
    text = str(value).strip()
    if text in GOOGLE_TO_FACTSCORE_LABEL:
        return GOOGLE_TO_FACTSCORE_LABEL[text]
    lowered = text.lower()
    if lowered == "supported":
        return "True"
    if lowered == "not supported":
        return "False"
    raise ValueError(f"Unsupported Google answer label: {value!r}")


def clean_answer_label(answer: str) -> str:
    return re.sub(r'[^\w\s]', '', answer).strip()


def deduplicate_adjacent_decisions(decisions: list[dict[str, Any]]):
    previous_atom = None
    deduplicated = []
    repeated_count = 0
    for decision in decisions:
        if decision.get("atom") != previous_atom:
            deduplicated.append(decision.copy())
            previous_atom = decision.get("atom")
        else:
            repeated_count += 1
    return deduplicated, repeated_count

################################################################################
#                             ABSTENTION DETECTION                             #
################################################################################
def generic_abstain_detect(generation):
    # === For vicuna 7B/13B, gpt-3.5-turbo, gpt-4o: I'm sorry ===
    # === For llama-2-chat 7B: I apologize ===
    # === For llama3.1-instruct: couldn't find, provide more, I'm unable to, I'm not ===
    return generation.startswith("I'm sorry") or generation.startswith("I apologize") \
        or generation.startswith("Sorry") or "provide more" in generation or "couldn't find" in generation\
        or generation.startswith("I'm not")

################################################################################
#                             STRING MANIPULATION                              #
################################################################################
def join_segments(*args: str | list[str], separator: str = '\n\n\n') -> str:
  """Joins an unspecified number of strings using the separator."""
  all_segments = []

  for arg in args:
    if isinstance(arg, list):
      all_segments.extend(arg)
    else:
      all_segments.append(strip_string(str(arg)))

  return strip_string(separator.join(all_segments))


def strip_string(s: str) -> str:
  """Strips a string of newlines and spaces."""
  return s.strip(' \n')


def extract_first_square_brackets(input_string: str) -> str:
  """Extracts the contents of the FIRST string between square brackets."""
  raw_result = re.findall(r'\[.*?\]', input_string, flags=re.DOTALL)

  if raw_result:
    return raw_result[0][1:-1]
  else:
    return ''

def extract_last_square_brackets(input_string: str) -> str:
  """Extracts the contents of the LAST string between square brackets."""
  raw_result = re.findall(r'\[.*?\]', input_string, flags=re.DOTALL)

  if raw_result:
    return raw_result[-1][1:-1]
  else:
    return ''

def extract_first_code_block(
    input_string: str, ignore_language: bool = False
) -> str:
  """Extracts the contents of a string between the first code block (```)."""
  if ignore_language:
    pattern = re.compile(r'```(?:\w+\n)?(.*?)```', re.DOTALL)
  else:
    pattern = re.compile(r'```(.*?)```', re.DOTALL)

  match = pattern.search(input_string)
  return strip_string(match.group(1)) if match else ''

def extract_last_code_block(
    input_string: str, ignore_language: bool = False
) -> str:
    """Extracts the contents of a string between the last code block (```)."""
    if ignore_language:
        pattern = re.compile(r'```(?:\w+\n)?(.*?)```', re.DOTALL)
    else:
        pattern = re.compile(r'```(.*?)```', re.DOTALL)
    
    match = pattern.findall(input_string)
    return strip_string(match[-1]) if match else ''

def extract_hash_block(
    input_string: str, number_of_blocks: int
) -> str:
    """Extracts the contents of a string under the hash block (###)."""
    pattern = re.compile(r'###\s*(.*?)\s*###\n(.*?)(?=\n###|$)', re.DOTALL)
  
  # search all the matched patterns
    match = pattern.findall(input_string)
   
    if number_of_blocks == 1:
        assert len(match) == 1, f'Error: {match}'
        return strip_string(match[0][1])
    elif number_of_blocks == 2:
        assert len(match) == 2, f'Error: {match}'
        return strip_string(match[0][1]), strip_string(match[1][1])
    
    elif number_of_blocks == 3:
        assert len(match) == 3, f'Error: {match}'
        return strip_string(match[0][1]), strip_string(match[1][1]), strip_string(match[2][1])
    
    else:
        raise ValueError(f'Not implemented. Error: {number_of_blocks}')

# def extract_one_hash_block(
#     input_string: str
# ) -> str:
#   """Extracts the first content of a string under the hash block (###)."""
#   pattern = re.compile(r'###\s*(.*?)\s*###\n(.*?)(?=\n###|$)', re.DOTALL)
  
#   # search all the matched patterns
#   match = pattern.findall(input_string)
#   assert len(match) == 1, f'Error: {match}'
#   return strip_string(match[0][1])

# def extract_three_hash_block(
#     input_string: str
# ) -> str:
#     """Extracts the contents of a string between the first code block (```)."""
#     pattern = re.compile(r'###\s*(.*?)\s*###\n(.*?)(?=\n###|$)', re.DOTALL)
    
#     # search all the matched patterns
#     match = pattern.findall(input_string)
#     assert len(match) == 3, f'Error: {match}'
#     return strip_string(match[0][1]), strip_string(match[1][1]), strip_string(match[2][1])

# def count_words(response_text: str) -> int:
#     return len(nltk.word_tokenize(response_text))

def count_words(response_text: str, mode='nltk') -> int:
    if mode == 'nltk':
        if nltk is None:
            raise ModuleNotFoundError("nltk is required for count_words(mode='nltk').")
        return len(nltk.word_tokenize(response_text))
    elif mode == 'split':
        return len(response_text.split())

################################################################################
#                             LLAMA2 PROMPT OPERATION                          #
################################################################################
def get_prompt_message(system_message: str, user_message: str, assistant_message: str):
    selection_prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>
"""
    split_user_messages = user_message.split('\n\n\n\n\n')
    split_assistant_messages = assistant_message.split('\n\n\n')

    selection_prompt += f"""{split_user_messages[0]}[/INST]

{split_assistant_messages[0]}</s>

"""

    for i in range(1, len(split_user_messages)):
        question = split_user_messages[i]
        answer = split_assistant_messages[i]
        selection_prompt += f"""<s>[INST]{question}[/INST]

{answer}</s>

"""

    return selection_prompt

# === Code from FActScore repository ===
def split_sentences(text):
    if sent_tokenize is None:
        raise ModuleNotFoundError("nltk is required for split_sentences().")
    sentences = []
    initials = detect_initials(text)
    
    curr_sentences = sent_tokenize(text)
    curr_sentences_2 = sent_tokenize(text)

    curr_sentences = fix_sentence_splitter(curr_sentences, initials)
    curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)

    # checking this, just to ensure the crediability of the sentence splitter fixing algorithm
    assert curr_sentences == curr_sentences_2, (text, curr_sentences, curr_sentences_2)

    sentences += curr_sentences
    
    return sentences

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

# === To fix sentence tokenization ===
def fix_sentence_splitter(curr_sentences, initials):
    if np is None:
        raise ModuleNotFoundError("numpy is required for fix_sentence_splitter().")
    for initial in initials:
        # if not found in any sentence, then use the following logic to merge the sentences
        if not np.any([initial in sent for sent in curr_sentences]):
            print(f'Hello: {initial}')
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            print(f'alpha 1: {alpha1}, alpha 2: {alpha2}')
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                print(f'sent1: {sent1}, sent2: {sent2}')
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    print(f'curr_sentences: {curr_sentences}')
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences


def main():
    # === Test the sentence splitter ===
    text = "Hello. How are you? I am fine"
    sentences = split_sentences(text)
    print(sentences)


if __name__ == '__main__':
    main()
