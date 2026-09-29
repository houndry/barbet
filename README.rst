.. image:: https://github.com/houndry/barbet/blob/main/docs/images/barbet-banner.jpg?raw=true

.. start-badges

|pypi badge| |testing badge| |coverage badge| |docs badge| |black badge| |torchapp badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/barbet?color=blue
   :alt: PyPI - Version
   :target: https://pypi.org/project/barbet/

.. |testing badge| image:: https://github.com/houndry/barbet/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/houndry/barbet/actions

.. |docs badge| image:: https://github.com/houndry/barbet/actions/workflows/docs.yml/badge.svg
    :target: https://houndry.github.io/barbet
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/09aad5114164b54daabe1f5efd02a009/raw/coverage-badge.json
    :target: https://houndry.github.io/barbet/coverage/

.. |torchapp badge| image:: https://img.shields.io/badge/torch-app-B1230A.svg
    :target: https://rbturnbull.github.io/torchapp/
    
.. end-badges

.. start-quickstart

Installation
==================================

Install using pip:

.. code-block:: bash

    pip install barbet


Usage
==================================

See the options for making inferences with the command:

.. code-block:: bash

    barbet --help


Run
==================================

.. code-block:: bash

    barbet --input GCA_000006945.2.fna --output-dir outputs

Or using the large model:

.. code-block:: bash

    barbet --input GCA_000006945.2.fna --output-dir outputs-large --large

The predicted lineage and the probability at each rank are written to ``barbet-predictions.csv`` in the output directory.


Additional outputs
==================================

Top predictions per rank
------------------------

To also write the top N taxa at each rank for each genome:

.. code-block:: bash

    barbet --input genomes/ --output-dir outputs --output-predictions top-predictions.tsv --num-predictions 5

By default the alternatives at each rank are the children of the top prediction at the rank above.
The TSV has the columns ``name``, ``rank``, ``taxon``, ``probability`` and ``prediction_number``,
where ``probability`` is the probability of the full lineage down to that taxon (as in ``barbet-predictions.csv``).

Add ``--global-predictions`` to instead rank every taxon at each rank across the whole taxonomy.
Taxa that are the only child of their parent are ranked with their parent's probability. The TSV then has the columns:

- ``joint_probability``: the probability of the full lineage down to the taxon
- ``local_probability``: the probability of the taxon given its parent (1.0 for an only child)
- ``prediction_number``: the position of the taxon in the ranking at that rank
- ``in_predicted_lineage``: whether the taxon is in the lineage reported in ``barbet-predictions.csv``

Context vectors
---------------

To write the mean-pooled context vector for each genome (the input to the classification layer, averaged over all gene stacks):

.. code-block:: bash

    barbet --input genomes/ --output-dir outputs --context-vector-file context-vectors.tsv.gz

The TSV has the columns ``name`` and ``context_vector`` (comma-separated values) and is gzipped if the path ends in ``.gz``.

For both options, a bare filename is written to ``--output-dir``, a directory gets a default filename
(``barbet-top-predictions.tsv`` or ``barbet-context-vectors.tsv``), and any other path is used as given.


Training
==================================

You can train the model on releases from GTDB or your own custom dataset.
See the instructions in the documentation for `preprocessing <https://houndry.github.io/barbet/preprocessing.html>`_ and `training <https://houndry.github.io/barbet/training.html>`_.

.. end-quickstart


Credits
==================================

.. start-credits

`Robert Turnbull <https://robturnbull.com>`_, Mar Quiroga, Gabriele Marini, Torsten Seemann, Wytamma Wirth

For more information contact: <wytamma.wirth@unimelb.edu.au>

Created using torchapp (https://github.com/rbturnbull/torchapp).

.. end-credits

