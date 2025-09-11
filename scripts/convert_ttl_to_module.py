from pathlib import Path

import rdflib
from rdflib import OWL, Namespace, URIRef
from rdflib.namespace import RDF, OWL, split_uri
import textwrap
INPUT_PATH = "/Users/gabriel/dev/sdle/CEMENTO/cemento/data/reserved/ms.ttl"

if __name__ == "__main__":
    g = rdflib.Graph()
    g.parse(INPUT_PATH, format='turtle')
    onto_term = next(g.subjects(RDF.type, OWL.Ontology))
    print(onto_term)
    onto_term_uri, _ = split_uri(onto_term)

    input_path = Path(INPUT_PATH)
    output_file_path = input_path.parent / f"{input_path.stem}.py"
    abbrev = input_path.stem.upper()
    tab = "    "
    with open(output_file_path, 'w') as f:
        header = f"""
        from rdflib.namespace import DefinedNamespace, Namespace
        from rdflib.term import URIRef
        
        class {abbrev}(DefinedNamespace):
        
            _fail = True

        """
        f.write(textwrap.dedent(header))

        subj_terms = map(lambda term: split_uri(term)[1], filter(lambda term: isinstance(term, URIRef), g.subjects()))
        for subj_term in set(subj_terms):
            f.write(f"{tab}{subj_term}: URIRef\n")
        f.write('\n')
        f.write(f'{tab}_NS = Namespace("{onto_term_uri}")')
