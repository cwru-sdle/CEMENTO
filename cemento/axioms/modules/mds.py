from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef

class MDS(DefinedNamespace):

    _fail = True

    tripleSyntaxSugar: URIRef
    Ontology: URIRef
    hasCollectionMember: URIRef
    enumeration: URIRef

    _NS = Namespace("https://cwrusdle.bitbucket.io/mds/")
