import re

from rdflib import Literal
from rdflib.namespace import split_uri


def clean_literal_string(literal_term: str) -> str:
    new_literal_term = literal_term.strip().replace('"', "")
    new_literal_term = re.sub(r"@\w+", "", new_literal_term)
    new_literal_term = re.sub(r"\^\^\w+:\w+", "", new_literal_term)
    return new_literal_term


def extract_aliases(term: str):
    return [
        alias.strip()
        for alias in re.match(r".*\((.*)\)", term).group(1).split(",")
        if alias.strip()
    ]


def format_literal(literal: Literal, prefix: str) -> str:
    if prefix is None:
        raise ValueError(
            "The literal datatype prefix was not specified. Literal datatype namespaces cannot be None."
        )
    literal_value = literal.value if literal.value else str(literal)
    literal_str = f'"{literal_value}"'
    lang_str = (
        f"@{literal.language}"
        if hasattr(literal, "language") and literal.language
        else ""
    )

    datatype_str = ""
    if hasattr(literal, "datatype") and literal.datatype:
        datatype = literal.datatype
        _, abbrev = split_uri(datatype)
        datatype_str = f"^^{prefix}:{abbrev}"

    return f"{literal_str}{lang_str}{datatype_str}"
