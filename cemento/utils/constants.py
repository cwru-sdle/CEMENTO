from enum import Enum


class RDFFormatException(BaseException):
    def __init__(self, message="Error when parsing RDF format"):
        self.message = message
        super().__init__(self.message)


class RDFFormat(Enum):
    TURTLE = "turtle"
    XML = "xml"
    JSONLD = "json-ld"
    NT = "nt"
    N3 = "n3"

    @classmethod
    def from_input(cls, input_format: str):
        format_mapping = {
            "turtle": RDFFormat.TURTLE,
            "ttl": RDFFormat.TURTLE,
            "turtle2": RDFFormat.TURTLE,
            "xml": RDFFormat.XML,
            "pretty-xml": RDFFormat.XML,
            "json-ld": RDFFormat.JSONLD,
            "ntriples": RDFFormat.NT,
            "nt": RDFFormat.NT,
            "nt11": RDFFormat.NT,
            "n3": RDFFormat.N3,
        }

        if input_format not in format_mapping.keys():
            raise ValueError(
                f"Cannot find specified format, options are: {', '.join(format_mapping.keys())}"
            )

        return format_mapping[input_format]

    @classmethod
    def from_ext(cls, file_ext: str):
        format_mapping = {
            ".ttl": RDFFormat.TURTLE,
            ".xml": RDFFormat.XML,
            ".jsonld": RDFFormat.JSONLD,
            ".nt": RDFFormat.NT,
            ".n3": RDFFormat.N3,
        }

        if file_ext not in format_mapping.keys():
            raise ValueError(
                f"Cannot find a format for the given file extension, supported file extensions are: {', '.join(format_mapping.keys())}"
            )

        return format_mapping[file_ext]

    @staticmethod
    def get_valid_file_extensions() -> list[str]:
        return [".ttl", ".xml", ".jsonld", ".nt", ".n3"]

    @staticmethod
    def get_valid_rdf_formats() -> list[str]:
        return [
            "turtle",
            "ttl",
            "turtle2",
            "xml",
            "pretty-xml",
            "json-ld",
            "ntriples",
            "nt",
            "nt11",
            "n3",
        ]


class NullTermError(ValueError):
    pass


valid_collection_types = {
    "owl:unionOf",
    "owl:intersectionOf",
    "owl:complementOf",
    "mds:tripleSyntaxSugar",
}
