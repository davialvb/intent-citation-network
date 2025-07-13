import re

## scicite title section map
ALLOC_MAP = {
    'introduction':
        [
            'introduction',
            'introduction and preliminaries',
            'introduction and related works',
            'background',
            'motivation',
        ],
    'methods':
        [
            'method',
            'methods',
            'methodology',
            'model',
            'the model',
            'system model',
            'experiments',
            'experimental setup',
            'numerical experiments',
            'evaluation',
            'evaluation setup',
            'experimental evaluation',
            'experimental design',
            'implementation details',
            'analysis',
            'simulations',
            'implementation',
            'experiment setup',
            'performance evaluation',
            'experimental setting',
            'materials and methods',
            'evaluation results',
            'experimental settings',
            'experiment',
        ],
    'results':
        [
            'result',
            'results',
            'main result',
            'main results',
            'the main result',
            'numerical results',
            'experimental results',
            'simulation results',
            'auxiliary results',
            'observations',
            'results and discussion',
            'results & discussion',
        ],
    'discussion':
        [
            'discussion',
            'discussions',
            'discussion and conclusion',
            'discussion and conclusions',
            'conclusions and discussion',
            'conclusions',
            'conclusion',
            'concluding remarks',
            'conclusions and outlook',
            'conclusion and future work',
            'conclusions and future work',
            'summary and discussion',
            'summary and conclusion',
            'summary and conclusions',
            'summary and outlook',
            'conclusion and discussion',
        ],
    'related work':
        [
            'related work',
            'related works',
        ]
}


def map_title_section(sec_clean):
    if sec_clean in ALLOC_MAP['introduction']:
        return 'introduction'
    elif sec_clean in ALLOC_MAP['methods']:
        return 'methods'
    elif sec_clean in ALLOC_MAP['results']:
        return 'results'
    elif sec_clean in ALLOC_MAP['discussion']:
        return 'discussion'
    elif sec_clean in ALLOC_MAP['related work']:
        return 'related work'
    else:
        return sec_clean


def normalize_title_section(text):
    if text:
        text = text.lower()
        text = re.sub(r'\d[\.\s]', '', text)
        text = re.sub(r'\w{1}[\.]', '', text)
        text = text.strip()
        text = map_title_section(text)
    else:
        text = 'no_title_section'
    return text


def count_words(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words)
