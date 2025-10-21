# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "cemento"))

project = "CEMENTO"
copyright = "2025, CWRU SDLE-MDS3 Center of Excellence"
author = "Gabriel Obsequio Ponon"
release = "0.13.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_design",
    "sphinx_iframes",
    "sphinxcontrib.shtest",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Options for pydata theme

html_theme_options = {
    "logo": {
        # In a left-to-right context, screen readers will read the alt text
        # first, then the text, so this example will be read as "P-G-G-P-Y
        # (short pause) Home A pretty good geometry package"
        "alt_text": "CEMENTO - Home",
        "text": "CEMENTO",
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
    },
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/cwru-sdle/CEMENTO/",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-github fa-lg",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            # Label for this link
            "name": "PyPI",
            # URL where the link will redirect
            "url": "https://pypi.org/project/cemento/",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-python fa-lg",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
    ],
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["sphinx-version", "icon-attrib"],
    "footer_center": ["copyright"],
    "footer_end": ["theme-version"],
}

# sitemap options

html_baseurl = "https://cwru-sdle.github.io/CEMENTO/"

# myst_parser options

source_suffix = {
    ".rst": None,
    ".md": None,
}
