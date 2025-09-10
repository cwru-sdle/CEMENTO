import argparse
import json

from cemento.draw_io.read_diagram import read_drawio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_json_graph_output",
        description="generates a JSON file corresponding to the graph of the given input drawio",
    )
    parser.add_argument(
        "input", help="the path to the input .drawio file", metavar="input_path"
    )
    parser.add_argument(
        "output", help="the path to the input .drawio file", metavar="input_path"
    )
    args = parser.parse_args()
    graph, _ = read_drawio(args.input, check_errors=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "nodes": dict(graph.nodes(data=True)),
                "edges": list(graph.edges(data=True)),
            },
            f,
            ensure_ascii=False,
        )
