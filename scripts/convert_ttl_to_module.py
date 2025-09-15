from pathlib import Path

import rdflib
from rdflib import OWL, Namespace, URIRef
from rdflib.namespace import RDF, OWL, split_uri
import textwrap
from pathlib import Path
import os
INPUT_FOLDER = "/Users/gabriel/dev/sdle/CEMENTO/cemento/data/reserved"
OUTPUT_FOLDER = "/Users/gabriel/dev/sdle/CEMENTO/cemento/axioms/modules"

if __name__ == "__main__":
    for path in os.scandir(INPUT_FOLDER):
        input_path = Path(path)
        g = rdflib.Graph()
        g.parse(input_path, format='turtle')
        onto_term = next(g.subjects(RDF.type, OWL.Ontology))
        onto_term_uri, _ = split_uri(onto_term)

        output_file_path = Path(OUTPUT_FOLDER) / f"{input_path.stem}.py"
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
