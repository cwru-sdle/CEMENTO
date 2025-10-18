from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import atan2, pi
from typing import NamedTuple

from more_itertools import flatten
from rdflib import URIRef


class TermCase(Enum):
    PASCAL_CASE = "pascal"
    CAMEL_CASE = "camel_case"


class DiagramKey(Enum):
    TERM_ID = "term_id"
    LABEL = "label"


class ShapeType(Enum):
    LITERAL = "literal"
    CLASS = "class"
    INSTANCE = "instance"
    UKNOWN = "shape"


class TreeFolding(Enum):
    FOLD = 1
    NO_FOLD = 0


# TODO: move to config.ini and parse with a function in preprocessing
SHAPE_WIDTH = 200
SHAPE_HEIGHT = 80
FILL_COLOR = "#f2f3f4"
STROKE_COLOR = "#000000"
x_padding = 10
y_padding = 20


def get_timestamp_str():
    return f"{datetime.now():%Y-%m-%dT%H:%M:%S.%fZ}"


class DiagramObject:
    pass


@dataclass
class DiagramInfo(DiagramObject):
    diagram_name: int
    diagram_id: int
    modify_date: str = field(default_factory=get_timestamp_str)
    grid_dx: int = 1600
    grid_dy: int = 850
    grid_size: int = 10
    page_width: int = 1100
    page_height: int = 850
    diagram_content: int = None
    template_key: str = "scaffold"


class ConnectorType(Enum):
    RANK_CONNECTOR = "rank"
    PROPERTY_CONNECTOR = "property"


@dataclass
class Connector(DiagramObject):
    connector_id: str
    source_id: str
    target_id: str
    connector_label_id: str
    connector_val: str
    rel_x_pos: float = 0
    rel_y_pos: float = 0
    start_pos_x: float = 0.5
    start_pos_y: float = 0
    end_pos_x: float = 0.5
    end_pos_y: float = 1
    is_dashed: bool = 0
    is_curved: bool = 0
    template_key: str = "connector"

    @staticmethod
    def center_coordinates(
        x_coordinate: float,
        y_coordinate: float,
        shape_height=SHAPE_HEIGHT,
        shape_width=SHAPE_WIDTH,
    ) -> tuple[float, float]:
        return (x_coordinate + shape_width / 2, y_coordinate + shape_height / 2)

    @staticmethod
    def compute_dynamic_position(
        source_shape_x: float,
        source_shape_y: float,
        target_shape_x: float,
        target_shape_y: float,
    ) -> tuple[float, float, float, float]:
        crit_angle = abs(atan2(SHAPE_HEIGHT, SHAPE_WIDTH))
        positions = [(source_shape_x, source_shape_y), (target_shape_x, target_shape_y)]
        (source_shape_x, source_shape_y), (target_shape_x, target_shape_y) = map(
            lambda coords: Connector.center_coordinates(*coords), positions
        )
        angle = atan2(target_shape_y - source_shape_y, target_shape_x - source_shape_x)

        if -crit_angle <= angle <= crit_angle:
            return (1, 0.5, 0, 0.5)
        elif crit_angle < angle <= pi - crit_angle:
            return (0.5, 1, 0.5, 0)
        elif angle > pi - crit_angle or angle < -(pi - crit_angle):
            return (0, 0.5, 1, 0.5)
        elif -(pi - crit_angle) <= angle < -crit_angle:
            return (0.5, 0, 0.5, 1)
        else:
            return (0, 0, 0, 0)

    def resolve_position(
        self,
        source_shape_pos: tuple[float, float],
        target_shape_pos: tuple[float, float],
        strat_only: bool = False,
        horizontal_tree: bool = False,
    ) -> None:

        if not strat_only:
            self.start_pos_x, self.start_pos_y, self.end_pos_x, self.end_pos_y = (
                Connector.compute_dynamic_position(*source_shape_pos, *target_shape_pos)
            )

        if horizontal_tree:
            temp = self.start_pos_x
            self.start_pos_x = self.start_pos_y
            self.start_pos_y = temp

            temp = self.end_pos_x
            self.end_pos_x = self.end_pos_y
            self.end_pos_y = temp


@dataclass
class GhostConnector(Connector):

    is_curved: bool = 1
    is_dashed: bool = 1


@dataclass
class Shape:
    shape_id: str
    shape_content: str
    x_pos: float
    y_pos: float
    shape_width: int
    shape_height: int
    fill_color: str = FILL_COLOR
    stroke_color: str = STROKE_COLOR
    tree_folding: int = TreeFolding.FOLD.value
    template_key: str = "shape"


@dataclass
class LiteralShape(Shape):
    template_key: str = "literal"


@dataclass
class ClassShape(Shape):
    template_key: str = "class"


@dataclass
class InstanceShape(Shape):
    template_key: str = "instance"


@dataclass
class Label(Shape):
    # TODO: move to constants
    shape_width: int = 100
    shape_height: int = 40
    fill_color: str = "none"
    stroke_color: str = "none"
    tree_folding: int = TreeFolding.NO_FOLD.value


@dataclass
class Line(DiagramObject):
    line_id: str
    start_pos_x: float
    start_pos_y: float
    end_pos_x: float
    end_pos_y: float
    line_width: int = 50
    line_height: int = 50
    template_key: str = "line"


class NxEdge(NamedTuple):
    subj: any
    obj: any
    pred: any


class NxStringEdge(NxEdge):
    subj: str
    obj: str
    pred: str


# Errors


class WrongFileFormatError(Exception):
    def __init__(self, message="Wrong file format provided"):
        self.message = message
        super().__init__(self.message)


class BadDiagramError(Exception):
    def __init__(self, output_file_path):
        self.message = f"The diagram has several syntax issues that needs to be resolved. Please refer to generated file at {output_file_path}. Problematic elements are in red! Refer to earlier output for list of errors."
        super().__init__(self.message)


class DisconnectedTermError(Exception):
    def __init__(self, term_id, term_content):
        if term_content is None or not term_content.strip():
            self.message = f"Term with id {term_id} is not connected to any other term."
        else:
            self.message = f"Term with content: {term_content}, is not connected to any other term."
        super().__init__(self.message)


class DisconnectedEdgeError(Exception):
    def __init__(self, message):
        super().__init__(message)


class MissingParentEdgeError(DisconnectedEdgeError):
    def __init__(self, edge_id, edge_content, child_content):
        if edge_content is None or not edge_content.strip():
            self.message = (
                f"Edge with id: {edge_id} does not have a source (parent) connected."
            )
        elif child_content:
            self.message = f"Edge with id: {edge_id}, and content: {edge_content} and with child: {child_content}, does not have a source (parent) connected."
        else:
            self.message = f"Edge with id: {edge_id}, and content: {edge_content}, does not have a source (parent) connected."
        super().__init__(self.message)


class MissingChildEdgeError(DisconnectedEdgeError):
    def __init__(self, edge_id, edge_content, parent_content):
        if edge_content is None or not edge_content.strip():
            self.message = (
                f"Edge with id: {edge_id} does not have a source (parent) connected."
            )
        elif parent_content:
            self.message = f"Edge with id: {edge_id}, and content: {edge_content} and with parent: {parent_content}, does not have a source (parent) connected."
        else:
            self.message = f"Edge with id: {edge_id}, and content: {edge_content}, does not have a source (parent) connected."
        super().__init__(self.message)


class BidirectionalEdgeError(Exception):
    def __init__(self, edge_id, edge_content, parent_content, child_content):
        message_start = f"Edge with id: {edge_id}"
        message_end = (
            " is bidirectional. Consider using regular arrows to avoid confusion."
        )

        if edge_content is not None and edge_content.strip():
            message_start = f"{message_start} and content: {edge_content}"

        if parent_content:
            message_start = f"{message_start} and connected to term: {parent_content},"

        if child_content:
            message_start = f"{message_start} as well as term: {child_content},"

        self.message = f"{message_start} {message_end}"
        super().__init__(self.message)


class InvertedEdgeError(Exception):
    def __init__(self, edge_id, edge_content, parent_content, child_content):
        message_start = f"Edge with id: {edge_id}"
        message_end = " is inverted! Please make sure to use end-arrows only for ontology diagrams."

        if edge_content is not None and edge_content.strip():
            message_start = f"{message_start} and content: {edge_content}"

        if parent_content:
            message_start = f"{message_start} and connected to term: {parent_content},"

        if child_content:
            message_start = f"{message_start} as well as term: {child_content},"

        self.message = f"{message_start} {message_end}"
        super().__init__(self.message)


class FloatingEdgeError(DisconnectedEdgeError):
    def __init__(self, edge_id, edge_content):
        if edge_content is None or not edge_content:
            self.message = f"Edge with id {edge_id} does not have anything connected."
        else:
            self.message = f"Edge with id: {edge_id}, and content: {edge_content}, does not have anything connected."
        super().__init__(self.message)


class CircularEdgeError(DisconnectedEdgeError):
    def __init__(self, edge_id, edge_content):
        if edge_content is None or not edge_content.strip():
            self.message = (
                f"Edge with id {edge_id} is only connected to itself. Please ignore."
            )
        else:
            self.message = f"Edge with id {edge_id}, and content: {edge_content}, is only connected to itself. Please ignore."
        super().__init__(self.message)


class BlankLabelError(Exception):
    def __init__(self, message):
        super().__init__(message)


class BlankTermLabelError(BlankLabelError):
    def __init__(self, term_id):
        self.message = f"Term with id {term_id} does not have a label."
        super().__init__(self.message)


class BlankEdgeLabelError(BlankLabelError):
    def __init__(self, edge_id, connected_terms):
        if not connected_terms:
            self.message = f"Edge with id {edge_id} does not have a label."
        else:
            self.message = f"Edge with id: {edge_id}, connected to {' and '.join(connected_terms)}, does not have a label."
        super().__init__(self.message)


class BaseContainerError(Exception):
    def __init__(self, message):
        super().__init__(message)


class NestedSyntaxSugarError(BaseContainerError):
    def __init__(self, container_id, container_value, member_ids, member_values):
        self.header = f"The container with id: {container_id} and content: {container_value} and members with ids: {member_ids} and corresponding values: {member_values}"
        self.message = f"{self.header} nested an unlabelled collection which resolves to an undefined collection type."
        super().__init__(self.message)


class ContainerSubjectError(BaseContainerError):
    def __init__(self, container_id, container_value, member_ids, member_values):
        self.header = f"The container with id: {container_id} and content: {container_value} and members with ids: {member_ids} and corresponding values: {member_values}"
        self.message = f"{self.header} is used as a subject! Container subjects are not supported yet."
        super().__init__(self.message)


class FloatingContainerError(BaseContainerError):
    def __init__(self, container_id, container_value, member_ids, member_values):
        self.header = f"The container with id: {container_id} and content: {container_value} and members with ids: {member_ids} and corresponding values: {member_values}"
        self.message = f"{self.header} does not have any connections."
        super().__init__(self.message)


def get_namespace_terms(*namespaces) -> set[URIRef]:
    namespace_terms = map(dir, namespaces)
    return set(filter(lambda term: isinstance(term, URIRef), flatten(namespace_terms)))
