# CEMENTO

<img src="https://github.com/cwru-sdle/CEMENTO/blob/master/figures/logo.svg" width=200 alt="logo">

[![Homepage](https://img.shields.io/badge/cemento-docs-blue)](https://cwru-sdle.github.io/CEMENTO/)
[![Version](https://img.shields.io/pypi/v/cemento)](https://pypi.org/project/cemento/)
[![GitHub Stars](https://img.shields.io/github/stars/cwru-sdle/CEMENTO)](https://github.com/cwru-sdle/CEMENTO/stargazers/)

`CEMENTO` is a component python package of the larger SDLE FAIR application suite of tools for creating scientific ontologies more efficiently. This package provides functional interfaces for converting draw.io diagrams of ontologies into RDF triple file formats and vice versa. This package is able to provide term matching between reference ontology files and terms used in draw.io diagrams allowing for faster ontology deployment while maintaining robust cross-references.

`CEMENTO` stands for the Centralized Entity Mapping & Extraction Nexus for Triples and Ontologies – a mouthful for an acronym, but an important metaphor for the package building the road to ontologies for materials data.

## Documentation Page

This `README.md` is supplemented by a more comprehensive documentation page. It can be found [in our homepage](https://cwru-sdle.github.io/CEMENTO/).

Check out MDS-Onto, the modular ontology for materials data, and other FAIR-related projects [through this link](https://cwrusdle.bitbucket.io/).

## Features

To summarize, the package offers the following features:

1. Converting RDF triples into draw.io diagrams of the ontology terms and relationships and vice versa
2. Substituting and matching terms based on ontologies that YOU provide
3. Support for nested collections, axioms and restrictions for directly inputting your rulesets into the diagrams!
4. Creating coherent tree-based layouts for terms for visualizing ontology class and instance relationships
5. Tree-splitting diagram layouts to support multiple inheritance between classes (though multiple inheritance is not recommended by BFO)
6. Support for URI prefixes (via binding) and literal annotations (language annotations like `@en` and datatype annotations like `^^xsd:string`)
7. Providing a log for substitutions made and suppressing substitutions by adding a key (\*).
8. Support for Property definitions. Properties that do not have definitions will default as an Object Property type. 
9. Support for multiple pages in a draw.io file, for when you want to organize terms your way.

## Installation

To install this particular package, use pip to install the latest version of the package:

```{bash}
# use a python environment
python -m venv .cemento
source .cemento/bin/activate

# install the actual package
pip install cemento
```

## Usage

To convert from turtle to drawio and vice versa:

```{bash}
# converting from .ttl to drawio
$ cemento ttl_drawio your_triples.ttl your_output_diagram.drawio

# converting from .drawio to .ttl
$ cemento drawio_ttl your_output_diagram.drawio your_triples.ttl
```

To convert in another RDF file format, do:

```{bash}
converting from .xml to drawio
 $ cemento rdf_drawio your_triples.xml your_output_diagram.drawio

alternatively, specify the format
 $ cemento rdf_drawio -f xml your_triples.xml your_output_diagram.drawio
```

You can also use the inverse:

```{bash}
# converting from .drawio to .ttl
$ cemento drawio_ttl your_output_diagram.drawio your_triples.ttl

# converting from .drawio to .xml
$ cemento drawio_rdf your_output_diagram.drawio your_triples.xml

# alternatively, specify the format
$ cemento drawio_rdf -f xml your_output_diagram.drawio your_triples.xml
```

### Adding Reference Ontologies

When using `cemento ttl_drawio` and `cemento rdf_drawio`, point the `--onto-ref-folder-path` argument with the folder containing the files you want to reference. The package comes pre-bundled with the Common Core Ontology (CCO). CCO will be used by default if the reference folder is not specified.

**CAUTION:** Repeated references are overwritten in the order the files are read by python (usually alphabetical order). If your reference files conflict with one another, please be advised and resolve those conflicts first by deleting the terms or modifying them.

### Adding Custom Prefixes

Add your custom prefixes and namespaces to a `prefixes.json` file. An example can be found in `examples/prefixes.json`. Add your prefix-namespace pair at the bottom. To use your new `prefixes.json` file, use `--prefix-file-path` when calling `cemento drawio_ttl` or `cemento drawio_rdf`.

## Scripting

To convert from draw.io diagram into an RDF file:

```{python}
from cemento.rdf.drawio_to_rdf import convert_drawio_to_rdf

INPUT_PATH = "happy-example.drawio"
OUTPUT_PATH = "sample.ttl"
LOG_PATH = "substitution-log.csv"

if __name__ == "__main__":
    convert_drawio_to_rdf(
        INPUT_PATH,
        OUTPUT_PATH,
        file_format="turtle", # set the desired format for the rdf file output. The format is inferred if this is set to None
        check_errors=True, # set whether to check for diagram errors prior to processing
        log_substitution_path=LOG_PATH, # set where to save the substitution log for term fuzzy search
    )
```

To do the opposite:

```{python}
from cemento.rdf.rdf_to_drawio import convert_rdf_to_drawio

INPUT_PATH = "your_onto.ttl"
OUTPUT_PATH = "your_diagram.drawio"

if __name__ == "__main__":
    convert_ttl_to_drawio(
        INPUT_PATH,
        OUTPUT_PATH,
        file_format="turtle", # set the desired format for the rdf input. The format is inferred if this is set to None
        horizontal_tree=False, #sets whether to display tree horizontally or vertically
        set_unique_literals=False, # sets whether to make literals with the same content, language and type unique
        classes_only=False, # sets whether to display classes only, useful for large turtles like CCO
        demarcate_boxes=True, # sets whether to move all instances to A-box and classes to T-box
    )
```

## Drawing Basics

The following diagram goes through an example supplied with the repository called `happy-example.drawio` with its corresponding `.ttl` file called `happy-example.ttl`. We used [CCO terms](https://github.com/CommonCoreOntology/CommonCoreOntologies) to model the ontology.

![happy-exampl-explainer-diagram](figures/happy-example-explainer.drawio.svg)

**NOTE:** Click on the figure and click the `Raw` button on the subsequent page to enlarge. If you prefer, your can also refer to the `do-not-input-this-happy-example-explainer.drawio` file found in the `figures` folder.

## Future Features

This package was designed with end-to-end conversion in mind. The package is still in active development, and future features may include, but are not limited to the following:

- **Visualizing Axioms and Restrictions.** Users will be able to visualize the axioms and restrictions already in the input RDF file.
- **An interactive mode.** Users will be able to visualize syntax errors, improper term connections (leveraging domains and ranges), and substitutions and make edits in iterations before finalizing a draw.io or `.ttl` output.
- **Integrated reasoner.** Packages like `owlready2` have reasoners like `HermiT` and `Pellet` that will be integrated to diagram-to-triple conversion. This is for when some implicit connections that you would want to make are a little bit tedious to draw but are equally as important.

## License

This project was released under the BSD-3-Clause License. For more information about the license, please check the attached `LICENSE.md` file.

## Third-party Licenses

For information about third-party licenses for packages used in this project, please refer to the `THIRD_PARTY_LICENSES.txt` file or the [Licenses Page](https://cwru-sdle.github.io/CEMENTO/license-info/licenses.html) on the documentation.

## Contact Information

If you have any questions or need further assistance, please open a GitHub issue and we can assist you there.
