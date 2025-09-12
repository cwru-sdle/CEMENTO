
from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef

class MS(DefinedNamespace):

    _fail = True

    that: URIRef
    SameAs: URIRef
    pattern: URIRef
    exactly: URIRef
    Ontology: URIRef
    max: URIRef
    Transitive: URIRef
    some: URIRef
    Reflexive: URIRef
    DifferentFrom: URIRef
    only: URIRef
    Symmetric: URIRef
    belongsTo: URIRef
    of: URIRef
    IntroTerm: URIRef
    equivalentTo: URIRef
    maxInclusive: URIRef
    min: URIRef
    langRange: URIRef
    InverseFunctional: URIRef
    Or: URIRef
    Irreflexive: URIRef
    onClass: URIRef
    Self: URIRef
    minLength: URIRef
    minInclusive: URIRef
    Functional: URIRef
    value: URIRef
    minExclusive: URIRef
    maxLength: URIRef
    maxExclusive: URIRef
    And: URIRef
    Not: URIRef
    Asymmetric: URIRef
    length: URIRef

    _NS = Namespace("https://cwrusdle.bitbucket.io/mds/manchester-syntax/")