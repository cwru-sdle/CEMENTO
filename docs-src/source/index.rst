.. CEMENTO documentation master file, created by
   sphinx-quickstart on Mon Jul 21 10:58:46 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
    :google-site-verification: IR9RTySb3FPwmbsK5FVfOqHoVuhW1P2WqIq9n8hWxhg

.. image:: /_static/logo.svg
    :width: 250px
    :class: homepage-logo

***********
CEMENTO
***********

.. toctree::
   :maxdepth: 1
   :hidden:

   quickstart
   user-guide
   modules
   faqs
   about
   changelog

**Version:** |release|

**Useful links**:
`Source Repository <https://github.com/cwru-sdle/CEMENTO>`_ |
`Issue Tracker <https://github.com/cwru-sdle/CEMENTO/issues>`_ |
`MDS-Onto Website <https://cwrusdle.bitbucket.io>`_ |
`PyPI Page <https://pypi.org/project/cemento/>`_



``CEMENTO`` converts your draw.io ontology diagrams into turtle files, and vice versa. ``CEMENTO`` can also:

- Match and substitute for terms in ontologies YOU provide.
- Create coherent tree-based layouts for visualizing ontology class and instance relationships (even with multiple inheritance).
- Support URI prefixes (via binding), literal annotations and property definitions.
- Collect domain and range instances as used in your ontology *(class inference coming soon)*.
- Point you to your diagram errors by highlighting faulty entities. Your errors show all at once.
- Support for multiple pages in a draw.io file, for when you want to organize terms your way.

.. grid:: 2
   
    .. grid-item-card::
        :img-top: _static/running_person.svg

        Quick Start
        ^^^^^^^^^^^

        You just want to convert files? Check out our quick start guide and get yourself converting ontology diagrams and ``.ttl`` files immediately.

        +++

        .. button-ref:: quickstart
            :expand:
            :color: dark
            :click-parent:

            To Quick Start

    .. grid-item-card::
        :img-top: _static/book.svg

        Guide
        ^^^^^

        A detailed guide for using the CLI and the scripting tools.

        +++

        .. button-ref:: user-guide
            :expand:
            :color: dark
            :click-parent:

            To the User Guide

    .. grid-item-card::
        :img-top: _static/more.svg

        API Reference
        ^^^^^^^^^^^^^

        The full documentation for all things CEMENTO.

        +++

        .. button-ref:: modules
            :expand:
            :color: dark
            :click-parent:

            To the API Reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`