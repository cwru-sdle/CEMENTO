from cemento.utils.io import make_data_dirs, download_default_reference_ontos


def register(subparsers):
    parser = subparsers.add_parser(
        "download",
        help="subcommand for downloading default reference ontologies.",
    )
    parser.add_argument(
        "output",
        help="the path to the desired output folder for downloaded default reference ontologies.",
        metavar="download_folder_path",
    )
    parser.set_defaults(_handler=run)


def run(args):
    print(f"downloading default reference ontologies to folder {args.output}...")
    download_path = make_data_dirs(args.output)
    download_default_reference_ontos(download_path)
