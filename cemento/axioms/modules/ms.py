from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef


class MS(DefinedNamespace):

    _fail = True

    exactly: URIRef
    Asymmetric: URIRef
    minLength: URIRef
    Ontology: URIRef
    IntroTerm: URIRef
    Functional: URIRef
    of: URIRef
    Is: URIRef
    DifferentFrom: URIRef
    some: URIRef
    Or: URIRef
    value: URIRef
    that: URIRef
    o: URIRef
    onProperty: URIRef
    max: URIRef
    InverseFunctional: URIRef
    Symmetric: URIRef
    pattern: URIRef
    minExclusive: URIRef
    Irreflexive: URIRef
    And: URIRef
    subPropertyChain: URIRef
    Single: URIRef
    maxExclusive: URIRef
    Transitive: URIRef
    Not: URIRef
    SameAs: URIRef
    length: URIRef
    maxLength: URIRef
    maxInclusive: URIRef
    min: URIRef
    minInclusive: URIRef
    langRange: URIRef
    onClass: URIRef
    Self: URIRef
    only: URIRef
    belongsTo: URIRef
    equivalentTo: URIRef
    Reflexive: URIRef
    forWhich: URIRef
    RemoveIterNumber: URIRef

    _NS = Namespace("https://cwrusdle.bitbucket.io/mds/manchester-syntax/")
