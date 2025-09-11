
from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef

class MS(DefinedNamespace):

    _fail = True

    pattern: URIRef
    Irreflexive: URIRef
    max: URIRef
    of: URIRef
    length: URIRef
    minLength: URIRef
    maxExclusive: URIRef
    langRange: URIRef
    that: URIRef
    maxInclusive: URIRef
    Functional: URIRef
    Symmetric: URIRef
    maxLength: URIRef
    minExclusive: URIRef
    only: URIRef
    onClass: URIRef
    minInclusive: URIRef
    Not: URIRef
    belongsTo: URIRef
    Transitive: URIRef
    Reflexive: URIRef
    Asymmetric: URIRef
    DifferentFrom: URIRef
    Or: URIRef
    some: URIRef
    Self: URIRef
    Ontology: URIRef
    SameAs: URIRef
    value: URIRef
    And: URIRef
    IntroTerm: URIRef
    exactly: URIRef
    InverseFunctional: URIRef
    min: URIRef

    _NS = Namespace("https://cwrusdle.bitbucket.io/mds/manchester-syntax/")