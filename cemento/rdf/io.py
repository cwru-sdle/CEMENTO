from collections.abc import Iterable
from functools import reduce
from pathlib import Path
from typing import Container

import pandas as pd
from rdflib import URIRef, Graph


def aggregate_graphs(folder_path: str | Path):
    files = Path(folder_path).rglob("*.ttl")
    graph = Graph()
    return reduce(lambda acc, item: acc.parse(item), files, graph)


def save_substitution_log(
    input_terms: dict[str, str],
    term_search_keys: dict[str, list[str]],
    term_substitution: dict[str, URIRef],
    substituted_terms: Container[str],
    log_substitution_path: str | Path,
) -> None:
    substitute_entries = {
        key: {
            "input": input_terms[key],
            "search_keys": search_keys,
            "substitute": term_substitution[key],
        }
        for key, search_keys in term_search_keys.items()
        if key in substituted_terms
    }
    df = pd.DataFrame.from_dict(substitute_entries, orient="index")
    df.to_csv(log_substitution_path, index=False)
