from cemento.rdf.drawio_to_rdf import convert_drawio_to_rdf
from cemento.utils.io import (
    get_default_defaults_folder,
    get_default_prefixes_file,
    get_default_references_folder,
)


def register(subparsers):
    parser = subparsers.add_parser(
        "drawio_ttl",
        help="subcommand for converting drawio files into rdf triples in the turtle format.",
    )

    parser.add_argument(
        "input",
        help="the path to the input drawio diagram file.",
        metavar="input_file_path",
    )
    parser.add_argument(
        "output",
        help="the path to the desired output ttl file.",
        metavar="output_file_path",
    )
    parser.add_argument(
        "-r",
        "--onto-ref-folder-path",
        help="the path to the folder containing the reference ontologies.",
        metavar="ref_ontologies_folder_path",
        default=get_default_references_folder(),
    )
    parser.add_argument(
        "-d",
        "--defaults-folder-path",
        help="the path to the folder containing the ttl files of the default namespaces.",
        default=get_default_defaults_folder(),
        metavar="default_ontologies_folder_path",
    )
    parser.add_argument(
        "-p",
        "--prefix-file-path",
        help="the path to the json file containing prefixes.",
        default=get_default_prefixes_file(),
        metavar="prefix_file_path",
    )
    parser.add_argument(
        "-lsp",
        "--log-substitution-path",
        help="the path to a csv file containing substitution results from term matching.",
        default=None,
        metavar="log_file_path",
    )
    parser.add_argument(
        "-dce",
        "--dont-check-errors",
        help="Set whether to check for diagram errors and to generate a diagram with errors indicated.",
        action="store_false",
    )
    parser.add_argument(
        "-cdr",
        "--collect-domains-ranges",
        help="Set whether to aggregate instances that are in the domain and range of a custom object property (Class inference coming soon).",
        action="store_true",
    )
    parser.set_defaults(_handler=run)


def run(args):
    print(f"converting {args.input} into a turtle file at {args.output}...")
    convert_drawio_to_rdf(
        args.input,
        args.output,
        file_format="turtle",
        onto_ref_folder=args.onto_ref_folder_path,
        defaults_folder=args.defaults_folder_path,
        prefixes_path=args.prefix_file_path,
        check_errors=args.dont_check_errors,
        log_substitution_path=args.log_substitution_path,
    )
