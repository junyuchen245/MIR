Contributing and development
============================

This page is for people changing MIR itself. It covers the development setup, code style, the stability
rules the package follows, and the checks every change has to pass.

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/junyuchen245/MIR.git
   cd MIR
   python3.8 -m pip install -e .[dev]      # pytest, ruff, pre-commit
   pre-commit install                      # run lint and formatting on every commit

Repository layout
-----------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Path
     - Contents
   * - ``src/MIR/models``
     - Registration networks, configuration factories (``configs_*.py``), spatial transformers and
       integrators (``registration_utils.py``), LDDMM (``lddmm.py``), affine registration, atlas building.
   * - ``src/MIR/image_similarity``, ``src/MIR/deformation_regularizer``, ``src/MIR/accuracy_measures``
     - Losses, regularizers and evaluation metrics.
   * - ``src/MIR/utils``, ``src/MIR/data_augmentation.py``, ``src/MIR/statistical_analysis``
     - Training, visualization, I/O and analysis helpers.
   * - ``tests``
     - Unit and regression tests (CPU, about a minute).
   * - ``tools``
     - Compatibility checks used when refactoring (see below).
   * - ``tutorials``
     - Runnable training and inference scripts. They are users of the package, not part of it.

**Vendored code.** ``models/SITReg``, ``models/fireants``, ``models/convexAdam``, ``intensity_normalization``
and ``random_image_generation/perlin_numpy`` are third-party projects kept close to upstream. They are
excluded from linting and formatting (``pyproject.toml``) and should only receive minimal, clearly marked fixes.

Code style
----------

- **Lint and format** with `ruff <https://docs.astral.sh/ruff/>`__; the configuration is in ``pyproject.toml``:

  .. code-block:: bash

     ruff check src/MIR tests tools
     ruff format src/MIR tests tools

- **Docstrings** follow the Google convention (``Args:``, ``Returns:``, ``Raises:``) and state tensor shapes
  explicitly, e.g. ``[B, 3, H, W, D]`` for a displacement field in voxels. They are rendered in the
  :doc:`API reference <../api>`, so describe what the code does, including return-tuple order.
- **Python 3.8 compatibility.** The tutorials run with ``python3.8``: use ``typing`` forms in annotations
  (``Optional[int]``, ``List[int]``, ``Tuple[...]``), not ``int | None`` or ``list[int]``.
- **Type hints** on public functions and methods wherever the type is clear.
- **Device handling.** Never hard-code ``.cuda()`` or ``'cuda:0'``: create tensors on the input's device
  (``x.device``) or move cached tensors with ``.to(x.device)`` in ``forward``. All losses and models must run
  on CPU, which is also how the tests run.

API stability
-------------

The tutorials, downstream projects and published checkpoints depend on the package in detail, so MIR treats
its whole module tree as public:

- Do not move, rename or remove modules, classes, functions, methods or parameters. Tutorials import deep
  paths such as ``MIR.models.registration_utils`` or ``MIR.models.VFA.Decoder``.
- New parameters go at the end of a signature and have defaults that preserve the old behaviour.
- Keep ``state_dict`` parameter names and shapes unchanged: pretrained weights are loaded by name with
  ``strict=True``. If a buffer becomes non-persistent, drop the stale key in ``_load_from_state_dict`` (see
  ``SpatialTransformer``) so older checkpoints keep loading.
- Changes in numerical output are bug fixes or explicit opt-in features, never side effects of refactoring.

Checks
------

Run the unit tests and the linter for every change:

.. code-block:: bash

   python3.8 -m pytest tests -q
   ruff check src/MIR tests tools && ruff format --check src/MIR tests tools

The test suite includes ``tests/test_public_api.py``, which compares every first-party module against
``tests/data/public_api.json``: removing a name or changing a parameter fails CI. After an intended,
backwards-compatible addition, refresh the reference with ``python3.8 tools/api_snapshot.py --write-test-data``.

Refactoring also has to pass the compatibility checks in ``tools/``. Record the references **before** making
changes, then compare after each step:

.. code-block:: bash

   # before: record the public API and reference outputs
   python3.8 tools/api_snapshot.py   --write /tmp/api.json
   python3.8 tools/golden_outputs.py --write /tmp/golden.json

   # after each change: all three must pass
   tools/check_all.sh /tmp/api.json /tmp/golden.json

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What it guarantees
   * - ``tools/api_snapshot.py``
     - Every module still imports, and no public name was removed or had its parameters (names, kinds,
       defaults) changed. Type annotations may be added freely.
   * - ``tools/golden_outputs.py``
     - Fixed-seed CPU forward passes of every model family, every loss, regularizer and metric produce the same
       numbers, every model has the same ``state_dict`` names and shapes, and the pretrained checkpoints found
       under ``tutorials/*/pretrained_wts`` load with ``strict=True`` and reproduce their outputs.
   * - ``tools/ast_equal.py``
     - A formatting or documentation-only edit changed no code: compares syntax trees, optionally ignoring
       docstrings and type annotations (``--code-only``).

Adding a model
--------------

1. Put the network in ``src/MIR/models/<Name>.py`` and, if it is configurable, a factory in
   ``configs_<Name>.py`` returning an ``ml_collections.ConfigDict``.
2. Export it from ``src/MIR/models/__init__.py`` and add it to ``__all__``.
3. Add a CPU test in ``tests/`` that builds a small instance and runs a forward pass, and a case in
   ``tools/golden_outputs.py``.
4. Document it: a page under ``docs/models/``, an entry in :doc:`../api`, and a runnable tutorial folder.

Building the documentation
--------------------------

.. code-block:: bash

   python3.8 -m pip install -e .[docs]
   python3.8 -m sphinx -b html docs docs/_build/html

The API pages are generated from the docstrings at build time (``docs/api`` is not committed).
