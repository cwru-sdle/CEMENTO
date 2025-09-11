
from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef

class MS(DefinedNamespace):

    _fail = True

    max: URIRef
    Irreflexive: URIRef
    Reflexive: URIRef
    belongsTo: URIRef
    DifferentFrom: URIRef
    maxLength: URIRef
    only: URIRef
    minExclusive: URIRef
    exactly: URIRef
    length: URIRef
    maxExclusive: URIRef
    minLength: URIRef
    minInclusive: URIRef
    some: URIRef
    Ontology: URIRef
    Or: URIRef
    value: URIRef
    Asymmetric: URIRef
    langRange: URIRef
    Functional: URIRef
    IntroTerm: URIRef
    Transitive: URIRef
    that: URIRef
    maxInclusive: URIRef
    pattern: URIRef
    Not: URIRef
    SameAs: URIRef
    min: URIRef
    onClass: URIRef
    Symmetric: URIRef
    InverseFunctional: URIRef
    Self: URIRef
    And: URIRef
    of: URIRef

    _NS = Namespace("https://cwrusdle.bitbucket.io/mds/manchester-syntax/")