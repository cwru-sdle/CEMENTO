import json

from cemento.draw_io.read_diagram import read_drawio

INPUT_PATH = "/Users/gponon/dev/CEMENTO/tests/test_files/read-diagram-03.drawio"
OUTPUT_PATH = "/Users/gponon/dev/CEMENTO/tests/test_refs/read-diagram-03.json"

if __name__ == "__main__":
    elements, all_terms, triples, output_containers = read_drawio(INPUT_PATH)
    term_dict = {term_id: elements[term_id].get("value", None) for term_id in all_terms}
    triples = [list(map(lambda term: term_dict[term], triple)) for triple in triples]
    output_containers = {
        term_dict[key]: list(map(lambda term: term_dict[term], values))
        for key, values in output_containers
    }
    with open(OUTPUT_PATH, "w") as f:
        output_dict = {
            "term_dict": term_dict,
            "triples": triples,
            "output_containers": output_containers,
        }
        json.dump(output_dict, f)
