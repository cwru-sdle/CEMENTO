import importlib.resources as pkg_resources
import os
from collections.abc import Iterable
from pathlib import Path
from string import Template

from defusedxml import ElementTree as ET


def get_template_files() -> dict[str, str | Path]:
    current_file_folder = Path(__file__)
    # retrieve the template folder from the grandparent directory
    template_path = current_file_folder.parent.parent.parent / "templates"

    if not template_path.exists():
        template_path = pkg_resources.files("cemento").joinpath("templates")

    template_files = [
        Path(file) for file in os.scandir(template_path) if file.name.endswith(".xml")
    ]
    template_dict = dict()
    for file in template_files:
        with open(file, "r") as f:
            template = Template(f.read())
            template_dict[file.stem] = template
    return template_dict


def write_error_diagram(
    file_path: str | Path, errors: Iterable[tuple[str, BaseException]]
) -> Path:
    file_path = Path(file_path)
    tree = ET.parse(file_path)
    root = tree.getroot()
    for elem_id, _ in errors:
        for element in root.findall(f".//*[@id='{elem_id}']"):

            styles = list(
                map(
                    lambda x: x.strip().split("="),
                    element.get("style").strip().split(";"),
                )
            )
            styles = filter(lambda style: style[0], styles)
            style_dict = {
                style[0]: "" if len(style) < 2 else style[1] for style in styles
            }
            style_dict["strokeColor"] = "#ff0000"

        new_style = [
            f"{key}={value}" if value else f"{key}" for key, value in style_dict.items()
        ]
        new_style = ";".join(new_style)
        element.set("style", new_style)

    new_file_path = file_path
    if "error_check" not in file_path.stem:
        new_file_path = file_path.parent / f"{file_path.stem}-error_check{file_path.suffix}"
    tree.write(new_file_path)

    return new_file_path
