installation
************

using pip
=========
The most easy way to install autotab is using ``pip``
::
    pip install autotab

However, if you are interested in optimizing pipeline for deep learning models, you can
choose to install tensorflow as well by using ``all`` option
::
    pip install autotab[all]

For list of all options see :ref:`installation_options` options

using github link
=================
You can use github link for install autotab.
::
    python -m pip install git+https://github.com/Sara-Iftikhar/AutoTab.git

The latest code however (possibly with less bugs and more features) can be insalled from ``dev`` branch instead
::
    python -m pip install git+https://github.com/Sara-Iftikhar/AutoTab.git@dev

To install the latest branch (`dev`) with all requirements use ``all`` keyword
::
    python -m pip install "AutoTab[all] @ git+https://github.com/Sara-Iftikhar/AutoTab.git@dev"

using setup.py file
===================
go to folder where repository is downloaded
::
    python setup.py install

.. _installation_options:

installation options
=====================
The ``all`` option will install tensorflow 2.7 version along with autotab and h5py.