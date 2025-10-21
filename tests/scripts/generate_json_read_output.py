import json
import sys

from cemento.draw_io.read_diagram import read_drawio

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("you must input an input and output path")
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    elements, all_terms, triples, output_containers = read_drawio(input_path)
    term_dict = {term_id: elements[term_id].get("value", None) for term_id in all_terms}
    triples = [list(map(lambda term: term_dict[term], triple)) for triple in triples]
    output_containers = {
        term_dict[key]: list(map(lambda term: term_dict[term], values))
        for key, values in output_containers
    }
    with open(output_path, "w") as f:
        output_dict = {
            "term_dict": term_dict,
            "triples": triples,
            "output_containers": output_containers,
        }
        json.dump(output_dict, f)
