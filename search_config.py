import os

################################################################################
#                              API KEYS
################################################################################
serper_api_key = ""
openai_api_key = ""

################################################################################
#                              SEARCH SETTINGS
# search_type: str = Google Search API used. Choose from ['serper'].
# num_searches: int = Number of results to show per search.
################################################################################
search_type = 'serper'
num_searches = 5
# num_searches = 3

################################################################################
#                               SAFE SETTINGS
# max_steps: int = maximum number of break-down steps for factuality check.
# max_retries: int = maximum number of retries when fact checking fails.
# debug_safe: bool = show debugging printouts when running SAFE.
################################################################################
max_steps = 5 # no use here, we just search once
max_retries = 10
debug_safe = False

################################################################################
#                         VERIFY UNSUPPORTED PIPELINE
################################################################################
verify_default_input_path = \
    ''

verify_default_start_stage = 'find'
verify_default_stop_stage = 'rate'
verify_default_start = 0
verify_default_end = -1
verify_google_output_dirname = 'google_verification'

verify_default_model = 'gpt-4o-mini'
verify_default_temperature = 0
verify_default_seed = 0
verify_default_openai_organization = os.environ.get('OPENAI_ORGANIZATION')

verify_retries = {
    'revise': 10,
    'query': 10,
    'rate': 5,
}

verify_workers = {
    'revise': 10,
    'query': 10,
    'search': 10,
    'rate': 10,
}

verify_output_paths = {
    'unsupported_output_dir': None,
    'revise_output_dir': None,
    'google_output_dir': None,
    'query_search_output_dir': None,
    'unsupported_path': None,
    'revised_path': None,
    'search_query_path': None,
    'search_results_path': None,
    'raw_search_results_path': None,
    'final_answers_path': None,
    'parameters_path': None,
}

verify_search_postamble = ''
