from collections.abc import Iterable
from functools import reduce
from pathlib import Path

import pandas as pd
from rdflib import URIRef, Graph


def aggregate_graphs(folder_path: str | Path):
    files = Path("cemento/data").rglob("*.ttl")
    graph = Graph()
    return reduce(lambda acc, item: acc.parse(item), files, graph)

def save_substitute_log(
    substitution_results: dict[str, tuple[URIRef, Iterable[str], Iterable[str]]],
    log_substitution_path: str | Path,
) -> None:
    log_entries = [
        (original_term, search_key, term, score, matched_term)
        for original_term, (
            matched_term,
            search_keys,
            matches,
        ) in substitution_results.items()
        for (search_key, (term, score)) in zip(search_keys, matches, strict=False)
    ]
    df = pd.DataFrame(
        log_entries,
        columns=[
            "original_term",
            "search_key",
            "search_result",
            "score",
            "matched_term",
        ],
    )
    df.to_csv(log_substitution_path)
