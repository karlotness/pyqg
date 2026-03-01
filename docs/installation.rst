.. _installation:

Installation
############

Requirements
============

The only requirements are

- Python (3.9 or later)
- numpy_ (2.0 or later)

PyQG can also conveniently store model output data as an xarray dataset. The feature (which is used in some of the examples in this documentation) requires xarray_.

.. _numpy:  https://www.numpy.org/
.. _xarray: https://xarray.pydata.org/en/stable/


Instructions
============

Installing from PyPI
^^^^^^^^^^^^^^^^^^^^

PyQG can be installed from `PyPI <https://pypi.org/project/pyqg/>`__
while optionally requesting extra dependencies which enable improved
performance or additional functionality.

.. tab:: Recommended

   .. code-block:: bash

      $ python -m pip install pyqg[recommended]

   This will install PyQG along with recommended dependencies for
   improved performance. In particular, this will also install `PyFFTW
   <https://github.com/pyFFTW/pyFFTW>`__.

.. tab:: Complete

   .. code-block:: bash

      $ python -m pip install pyqg[complete]

   This installs the package will all optional dependencies, enabling
   all functionality including xarray_ support.

.. tab:: Minimal

   .. code-block:: bash

      $ python -m pip install pyqg
      # *OR* use:
      $ python -m pip install pyqg[]

   This installs a PyQG with most functionality and minimal
   dependencies, although with perhaps reduced performance.

Installing with Conda
^^^^^^^^^^^^^^^^^^^^^

PyQG can also be installed using conda:

.. code-block:: bash

    $ conda install -c conda-forge pyqg

The Conda installation provides all available functionality.

Installing from source
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ git clone https://github.com/pyqg/pyqg.git

Then install pyqg locally on your system:

.. code-block:: bash

    $ cd pyqg && python -m pip install .

If desired, extras can also be specified similarly to installations
from PyPI:

.. code-block:: bash

    $ cd pyqg && python -m pip install .[recommended]
