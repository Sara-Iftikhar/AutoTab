installation
************

using pip
=========
The most easy way to install autotab is using ``pip``
::
    pip install autotab

installing a specific version
=============================
If you want to install a specific version of ai4water using pip, just specify the version
as below
::
    pip install autotab==0.12

using github link
=================
You can use github link for install autotab.
::
    python -m pip install git+https://github.com/Sara-Iftikhar/AutoTab.git

The latest code however (possibly with less bugs and more features) can be installed from ``dev`` branch instead
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
If you are interested in optimizing pipeline for deep learning models, you can
choose to install tensorflow as well by using ``all`` option
::
    pip install autotab[all]

The ``all`` option will install tensorflow 2.7 version along with autotab and h5py.
