************************
Axioms and Restrictions
************************

This page provides an overview of axioms and restrictions that can be visually described with CEMENTO.

CEMENTO allows a visually coherent introduction of full OWL DL axioms and restrictions with draw.io native objects. Users do not need to download custom pallettes or patterns online.

We strongly recommend going through the :doc:`Drawing Basics </user-guide-pages/drawing-basics>` tutorial before reading this guide.

A Walkthrough
=============

The subsequent diagram demonstrates the visual syntax CEMENTO uses for writing OWL DL axioms and restrictions.  This file is also available as an ``examples`` folder under the file name ``happy-example-2.drawio``.

    **NOTE:** CEMENTO only currently supports diagram-to-RDF conversion for graphs with axioms. The example file also excludes nested collections section to avoid errors about disconnected lists.

.. iframe:: https://viewer.diagrams.net?#Uhttps%3A%2F%2Fraw.githubusercontent.com%2FGabbyton%2FCEMENTO%2Frefs%2Fheads%2Fmaster%2Ffigures%2Fdo-not-input-this-happy-example-2-explainer.drawio
    :height: auto
    :width: 100%
    :aspectratio: 1.77

Having trouble? Download the figure above as an :download:`svg image <https://raw.githubusercontent.com/Gabbyton/CEMENTO/refs/heads/master/figures/happy-example-2-explainer.drawio.svg>` or :download:`draw.io diagram <https://raw.githubusercontent.com/Gabbyton/CEMENTO/refs/heads/master/figures/do-not-input-this-happy-example-2-explainer.drawio>`.

In Case You Missed It
=====================

The diagram above goes through key axioms and restriction constructs that can be defined visually in CEMENTO diagrams. In case you didn't catch those features, the following diagrams goes through key concepts and capabilities:


* **Nested Collections.**
    OWL collections for defining unions, intersections and complements are notorious for their convoluted syntax. For CEMENTO, all you need is to use the list object to create your collection. The headers can be owl:intersectionOf, owl:unionOf, owl:unionOf, owl:datatypeComplementOf, and owl:oneOf.

* **Domains and Ranges.**
    You can now use nested collections for domains, ranges and other properties. CEMENTO enables users to visually describe complex classes and their combinations.

* **Equivalence and Disjointness.**
    Users can declare whether two classes, instances, and properties are the incompatible or equivalent.

* **Datatypes and data type ranges**
    CEMENTO extensively supports the declaration and description of custom datatypes. Datatypes can also connect to collections for more complex restrictions on types. Data type facets are also supported via the Manchester-syntax (MS) inspired bracket notation.

* **Property Restrictions**
    A user now able to define property restrictions, including property chains, existential restrictions, cardinalities and the like. CEMENTO uses a hybrid turtle-Manchester syntax for full visual axiomatic descriptions.

* **Logical recombination**
    CEMENTO allows users to explicitly nest and define restrictions via logical branches or streams. Each branch is an anonymous restriction triple that are connected by arrows to denote facets.

Non-supported Axioms and Restrictions
=====================================

* **NOT constructs**
    Negation statements are NOT supported. Our reasons are two-fold:
        1. Negation statements are explicit and it contradicts with a declarative philosophy of ontology creation.
        2. Negation on object properties cannot be represented by arrows because draw.io does not allow arrows that connect to/from the middle of arrows.

* **Language range**
    This was not supported because: Language range constructs use the MS bracket construct for datatype facets but interpret them differently, and language annotations aren't extensively supported in CEMENTO.

* **Nested restrictions**
    Users cannot simply connect one ``owl:Restriction`` to another. This is because nested restrictions are generally discouraged when a named class can simplify the logic. Nesting can also lead to self-contradictions which requires a reasoner to catch.