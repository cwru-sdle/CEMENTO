# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Guide]

- **Added** for new features.
- **Changed** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.
- **Security** in case of vulnerabilities.

_[Unreleased]_ section for tracking changes prior to binning to versions.

_[X.X.X] - YYYY-MM-YY_ for version-date header

## [0.13.0] - 2025-10-21

### Added

- support for defining axioms and restrictions via a hybrid notation of Manchester syntax
- a light reference ontology for custom hybrid Manchester syntax terms and their counterparts
- naming a term explicitly as a class or named individual except for property terms
- using xsd terms to the search pool for term matching
- choosing between term matching algorithms when using `substitute_term`
- tests and new test cases for axiom reading

### Changed

- refactored the entirety of the read workflow
- changed all read associated functions with more functional counterpart
- interface functions to avoid intermediate graph generation scripts
- read_drawio function signature. The function now outputs all required elements for converting to RDF formats rather than a unified graph object
- test scripts to adapt to new read_drawio function signature

### Removed

- all functions for `convert_drawio_to_rdf` prior to this version
- all functions associated with `substitute_term` and every one of its variants
- feature for collecting domains and ranges
- intermediate (`networkx`) graph file generation for converting between drawio to RDF

## [0.12.3] - 2025-10-01

### Hotfix

- added `more_itertools` to `pyproject.toml` as a dependency

## [0.12.2] - 2025-10-01

### Fixed

- missing dependency `more_itertools`

## [0.12.1] - 2025-10-01

### Fixed

- `Self` object from `typing` causing incompatibility with python 3.10.

## [0.12.0] - 2025-08-16

### Added

- custom scorer for `substitute_term`
- support for collection reading and writing as objects to RDF triples
- error checks for diagrams with containers
- identifying errors for floating containers, containers used as subjects, and nested unlabelled containers
- container and container content input into diagram error checks
- rdf_graph output testing

### Changed

- variable in `read_diagram` for elements to distinguish container and non-container elements
- condition for strict camel case. It now adds lowercase for capitalized abbreviated predicate terms
- diagram term iteration scheme to ensure unique terms
- `inv_constructed_terms` into `preferred_alias_keyed_inv_constructed_terms`

### Fixed

- `substitute_term` replacing the matched result with a lowercased string
- domain range and collection option being activated by false function signature
- wrong capitalizations on predicate terms
- error in iterating over predicate terms
- diagram iteration overwriting predicate status of a term due to ducktyped class redefinitions
- unstable label assignment due to duplicates overwriting alias keys

## [0.11.2] - 2025-08-13

### Added

- processing file prefixes within the input file when converting to drawio

### Fixed

- `get_root_node` function not returning the only node in the graph if it has a self-referential edge
- cycles in graphs getting processed alongside regular edges

## [0.11.1] - 2025-08-12

### Added

- more contextual error message for converting graphs directly into trees

### Fixed

- example on the user-guide page in the documentation

## [0.11.0] - 2025-08-12

### Added

- support for other rdf-compliant datatypes supported by `rdflib`
- prefix inference from format input
- RDF file format enum
- specifying format for reading and writing output
- added third party-licenses to repository and documentation

### Changed

- arguments for rdf and cli functions
- function signature for graph conversion functions
- file names referencing ttl into ones referencing rdf
- updated and streamlined `README.md` for new features

### Removed

- file conversion functions: `convert_drawio_to_ttl`, `convert_ttl_to_drawio`
- graph conversion functions: `convert_graph_to_ttl`, `convert_ttl_to_graph`

## [0.10.0] - 2025-08-11

### Added

- property family of exempted properties to exempt when drawing diagrams
- tests for diagram error checks
- error and error check for bidirectional and inverted arrows
- step for assigning nested edge label values to label in `extract_elements` as its own function
- tests for diagram element parsing and reading into graphs
- dynamic prefix generation for literal datatypes versus previous hard-coded `xsd`
- support for imported non-`xsd` datatypes. Hand-coded datatypes not supported yet (issue with `rdflib`)

### Changed

- quote substitution in shape content to general html escape substitution
- `parse_elements` implementation to be more functional
- moved `parse_elements` post-processing steps and `parse_elements` into updated function
- hard-coded `xsd` requirements for literal datatypes
- changed from searching just `xsd` types to entire search term pool

### Deprecated

- file conversion functions: `convert_drawio_to_ttl`, `convert_ttl_to_drawio`
- graph conversion functions: `convert_graph_to_ttl`, `convert_ttl_to_graph`

### Fixed

- properties not showing up as parent classes when outputting diagrams
- none-type check in diagram error parsing to consider capitalized `None`

## [0.9.6] - 2025-08-05

### Fixed

- wrong dividing line alignment for trees without instances

## [0.9.5] - 2025-08-05

### Fixed

- print statements when running `convert_drawio_to_ttl`

## [0.9.4] - 2025-08-05

### Added

- dynamic version number detection in CLI message

## [0.9.3] - 2025-08-05

### Added

- center coordinates to `Connector` dataclass
- coordinate centering when rendering arrow positions (hasn't fixed stray arrows)

### Fixed

- horizontal offsets not switching in `draw_tree`
- non-rank stratified predicates being rendered as rank terms

## [0.9.2] - 2025-08-04

### Fixed

- `read_diagram` and `convert_graph_to_ttl` not running when using scripts
- null safety for folder variables. Get the default if not specified instead
- `convert_ttl_to_graph` with improper null behavior for folders
- example scripts in the example folder

## [0.9.1] - 2025-08-01

### Added

- contributor

### Fixed

- `assign_literal_ids` zips with uneven lengths

## [0.9.0] - 2025-08-01

### Added

- instance conformation for T-box and A-box separation for each tree
- draw divider line to demarcate instance conformation for T-box and A-box separation
- tree conformation to align all demarcation lines
- T-box and A-box labels
- box demarcation option to CLI
- divider line template xml file
- option for activating domain-range instance collection. Default behavior now is not to collect.

### Changed

- separated `draw_diagram` from `draw_tree` for future layout schemes

### Fixed

- `get_aliases` "eating up" the label if more alt labels are provided
- `get_severed_connectors` with a dangling function argument
- severed links not being added to the graph
- severed links being in reverse when displayed on the graph
- `convert_drawio_to_ttl` null safety for reference folder arguments

## [0.8.9] - 2025-07-25

### Added

- FAQs page for common issues and solutions
- section for citations and license in about page

### Changed

- adjusted `write_diagram` graph flipping for repaired orientations
- updated default arrow position calculation on `Connector` dataclass to reflect repaired orientations
- repository location. The CEMENTO repository is now "owned" by the CWRU-SDLE organization
- base URL for documentation. It is now in [https://cwru-sdle.github.io/CEMENTO/](https://cwru-sdle.github.io/CEMENTO/)

### Fixed

- term location not printing on diagram error causing error
- reversed arrow configuration on `connector.xml` template
- missing parent content for `missingChildError`

## [0.8.8] - 2025-07-25

### Added

- Shape type implementation for generating shapes
- New enum `ShapeType` for determining shape type

## [0.8.7] - 2025-07-24

### Added

- added new template files for class, instance, and literal
- more detailed key error message for `generate_graph`
- null safety checks after diagram error detection in `diagram_terms_iter`
- connected term location and ID when outputting diagram errors
- checks to ignore horizontal lines for diagram error checking

### Changed

- error check to make changes are made in-place if the user is already working n a file with "error_check" on the file name
- default terms in `drawio_to_ttl` to use all default terms in rdflib and in default file folders
- print out triples that passed diagram checks but caught in null check in `convert_graph_to_ttl`
- replaced all ghost connectors with straight orthogonal connectors

### Fixed

- fixed class and instance designation in graph_to_tll
- fixed error message input parsing

### Removed

removed root IDs from extracted terms and relationships in extract_elements

## [0.8.6] - 2025-07-24

### Added

- error check option on CLI
- term content in diagram error message
- defaults folder file contents to search terms
- feature to remove redundant statements about default namespace terms
- ability to define object properties
- support for multipage inputs
- feature to replace default object-property assignment to custom properties to swap with definitions if available

### Changed

- domain-range `.ttl` output to single element if only one
- to check errors by default
- updated examples for the new version
- updated figure with the new features

## [0.8.5] - 2025-07-23

### Added

- package documentation on github pages
- parse containers function
- googe site verification
- site logo and icon attribution
- sitemap
- reference ontology retrieval
- term types for all predicates
- restored error check feature on diagram, including error classes

### Changed

- hand-made XSD reference to XSD namespace inside `rdflib`
- no unique literals option to store true flag, setting no unique literals as the default behavior
- `file_path` argument in `conver_drawio_ttl` function to `input_path`

### Removed

- hand-made XSD reference
- do not check option
- not literal IDs

### Fixed

- exact match functionality not outputting all desired properties (label and SKOS exact match)
- non-bunny-eared data type string output
- prefixes not being imported from file

## [0.8.4] - 2025-07-20

**NOTE:** The changes listed here are a catch-all between this version and all prior releases. We haven't kept a good changelog until today, so we apologize for the broad statements to keep this document section brief.

### Added

- application CLI
- support for converting directly to `.ttl` files from draw.io and vice versa
- support for literals and literal annotations (language and datatype)
- term matching via reference ontologies
- ability to add reference ontologies
- unique literal ID generation option
- support for annotation types
- classes-only option for drawing layouts
- ability to write prefixes
- tree-splitting for dealing with multiple inheritance
- stratified term category (includes definitions, annotations, etc.) for prioritizing in the layout
- match suppression with star keys
- alias support with parenthetic notation
- README instructions on CLI and scripting for new package implementation

### Changed

- programmming paradigm, from an clunky OOP-based approach to a hybrid functional approach
- File structure, adopting file conventions in functional programming
- all prior functionality implementations except those expressly mentioned in the remove section
- choosing more general category of terms to draw in the tree layout (stratified) versus just rank terms (subclass and type)
- shape definitions from native classes to dataclasses
- rendering shapes directly from dataclasses instead of through manual prop generation
- computing arrow directions dynamically based on shape angle instead of static case-based matching
- example scripts

### Removed

- All functions built under the OOP-based software
- shape-extent-based area diagram reading
- circle-based (organic) layouts
- straight-arrow and curve template files
- error detection in diagram reads
- defer-layout option
