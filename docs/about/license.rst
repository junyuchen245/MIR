License and author
==================

License
-------

MIR is released under the `Apache License 2.0 <https://opensource.org/license/apache-2-0>`__. Vendored
third-party code (for example ``MIR.models.SITReg``) keeps its original license, included alongside the
code where the upstream project provides one. Ported methods (LessNet, Fourier-Net, ConvexAdam,
MultiMorph) are credited on their model pages and in :doc:`publications`.

Author
------

MIR is developed by `Junyu Chen <https://junyuchen245.github.io>`__ at Johns Hopkins University, with
contributions from the co-authors of the papers it implements.

Contributing
------------

Issues and pull requests are welcome on `GitHub <https://github.com/junyuchen245/MIR>`__. Please run the
unit tests before submitting:

.. code-block:: bash

   python3.8 -m pytest tests -q
