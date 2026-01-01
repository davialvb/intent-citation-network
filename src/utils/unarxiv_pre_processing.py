import numpy as np
import pandas as pd
from glob import glob
import json
import re
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import igraph as ig

tqdm.pandas()

def get_paper_data(path_jsonl, full=True, just_metadata=False):
    df = None
    try:
        df = pd.read_json(path_jsonl, lines=True)
        df.rename(columns = {'paper_id': 'id'}, inplace=True)
        df.id = df.id.astype(str)
        
        metadata_df = pd.concat([pd.json_normalize(metadatas) for metadatas in df.metadata.values])
        
        body_text_normalized = []
        for index, row in df.iterrows():
            body_text_temp = pd.json_normalize(row['body_text'])
            body_text_temp['id'] = row['id']
            body_text_normalized.append(body_text_temp)
            
        body_text_df = pd.concat(body_text_normalized)
        body_text_df.id = body_text_df.id.astype(str)
    
        columns_to_keep = ['id', '_source_hash', 'discipline', 'bib_entries']
        if not full:
            columns_to_keep.remove('_source_hash')
            #main_columns = columns_to_keep + ['authors', 'title', 'section', 'text', 'content_type']
            main_columns = columns_to_keep + ['section', 'authors','title', 'text']
        
        df = df[columns_to_keep]
        df_aux = df.merge(metadata_df, on=['id'])
        df_result = df_aux.merge(body_text_df, on=['id'])
        if just_metadata:
            return df_result[["id", "title", "update_date", "submitter", "authors", "journal-ref", "categories"]].drop_duplicates()
        
        return df_result[main_columns] if not full else df_result 
    except Exception as e:
        print(path_jsonl)
        print(e) 


def replace_citations(text, citations):
    def replacement(match):
        citation_id = match.group(1)
        citations_text = citations[citation_id]['bib_entry_raw'] if citation_id in citations else match.group(0)
        return citations_text 
    return re.sub(r'\{\{cite:([a-z0-9]+)\}\}', replacement, text)


def replace_citations_random(text):
    def replacement(match):
        citation_id = match.group(1)
        random_citation_number = np.random.randint(30)
        return f"[{random_citation_number}]"
    return re.sub(r'\{\{cite:([a-z0-9\-]+)\}\}', replacement, text)


def replace_formula(text):
    return re.sub(r'\{\{formula:([a-z0-9\-]+)\}\}', "[mathematical_formula]", text)
def replace_figure(text):
    return re.sub(r'\{\{figure:([a-z0-9\-]+)\}\}', "[figure]", text)


def normalize_text(text):
    text = text.replace('\xa0', ' ')
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

def get_cited_paper(bib_entries):
    cited_ids = []
    for key, value in bib_entries.items():
        if value.get('ids'):
            cited_ids.append(value.get('ids').get('arxiv_id'))
    return list(filter(None, cited_ids))

def get_cited_paper(bib_entries):
    cited_ids = []
    for key, value in bib_entries.items():
        if value.get('ids'):
            cited_ids.append(value.get('ids').get('arxiv_id'))
    return list(filter(None, cited_ids))

def get_cited_ids(cited_hash, bib_entries):
    cited_ids = []
    bib_raw = []
    discipline = []
    for key, value in bib_entries.items():
        if value.get('ids'):
            if key in cited_hash:
                arxiv_id = value.get('ids').get('arxiv_id')
                if arxiv_id:
                    cited_ids.append(arxiv_id)
                    bib_raw.append(value.get('bib_entry_raw'))
                    discipline.append(value.get('discipline'))
                    
    return cited_ids, bib_raw, discipline

def get_cited_raw(cited_hash, bib_entries):
    bib_raw= []
    for key, value in bib_entries.items():
        if value.get('bib_entry_raw'):
            if key in cited_hash:
                arxiv_id = value.get('bib_entry_raw')
                bib_raw.append(arxiv_id)
    return list(filter(None, cited_ids))