.. _installation:

Installation
############

Requirements
============

The only requirements are

- Python (3.9 or later)
- numpy_ (1.22 or later)

PyQG can also conveniently store model output data as an xarray dataset. The feature (which is used in some of the examples in this documentation) requires xarray_.

.. _numpy:  https://www.numpy.org/
.. _xarray: https://xarray.pydata.org/en/stable/


Instructions
============

Installing with Conda
^^^^^^^^^^^^^^^^^^^^^

We suggest that you install pyqg using conda. To install pyqg with conda,

.. code-block:: bash

    $ conda install -c conda-forge pyqg


Installing from PyPI
^^^^^^^^^^^^^^^^^^^^

To install from `PyPI <https://pypi.org/project/pyqg/>`__:

.. code-block:: bash

    $ python -m pip install pyqg


Installing from source
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ git clone https://github.com/pyqg/pyqg.git

Then install pyqg locally on your system:

.. code-block:: bash

    $ cd pyqg && python -m pip install .
