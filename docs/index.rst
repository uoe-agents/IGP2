.. igp2 documentation master file, created by
   sphinx-quickstart on Mon Apr 24 16:59:35 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Interpretable Goal-based Prediction and Planning (IGP2)
=======================================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   installation
   first_steps
   custom_scenarios
   goal_recognition
   road_layout
   configuration_file
   new_behaviour
   overwrite_mcts
   commandline_options
   useful_links
   api/modules
   api/module_graph
   changelog

Welcome to the Interpretable Goal-based Prediction and Planning (IGP2) documentation.

This home page contains an index of contents of the full documentation of IGP2.
If you are new to using IGP2, we recommend starting with the below three steps, however, feel free to read the documentation in any order that suits you.

1. Install IGP2: You should check out the :ref:`installation` page.
2. Start using IGP2: You can refer to the :ref:`first_steps` page to run IGP2.
3. Create your own scenario: You can create your own scenario by following the :ref:`custom_scenarios` page.
4. Refer to the API: If you would like to develop your own code using IGP2 then you can consult the :ref:`IGP2 API <api>`.

Support
-------

If you have any issues with installing or running the code, then feel free to use the `GitHub Issues <https://github.com/uoe-agents/IGP2/issues>`_ page to ask for help. We will endeavour to answer questions as soon as possible.

Bug reports
^^^^^^^^^^^

If you encounter a bug with IGP2 please put in a ticket to the GitHub issues page with your systems setup information,  steps about how to reproduce your bug, and the debug log from your running of IGP2.

Getting started
---------------

:ref:`installation` - This page guides you through all the necessary steps to get up and running with IGP2.

:ref:`first_steps` - This page guides you through how to run your first driving scenario with IGP2.


Components
^^^^^^^^^^

:ref:`goal_recognition` - The goal recognition module of IGP2 can be run in standalone mode without simulating for motion planning. This page tells you how to do just that.

:ref:`custom_scenarios` - IGP2 comes with several pre-defined scenarios, but this page will guide you through how you can create your own scenarios.

:ref:`road_layout` - IGP2 relies on the OpenDrive 1.6 standard for road layout definition, however there are some important additional steps you should take when creating your own maps for IGP2. This page will walk you through those steps.

:ref:`configuration_file` - This page gives full documentation of all fields that can appear in configuration files of IGP2.

Extending IGP2
^^^^^^^^^^^^^^

:ref:`Add your own macro actions and maneuvers <new_behaviour>`: You can add new agent behaviour to IGP2 by defining new macro actions and maneuvers.

:ref:`Overwrite MCTS <overwrite_mcts>`: You can also overwrite how MCTS is run to add custom planning behaviour.

Resources
^^^^^^^^^

:ref:`commandline_options` - A list of commandline options to be used with `scripts/run.py`.

:ref:`IGP2 API <api>` - Directory of the API of IGP2 organised according to its package hierarchy.

:ref:`module_graph` - Browse the unnecessarily complicated module dependency graph of IGP2.

:ref:`useful_links` - A collection of useful links relevant to working with IGP2.

:ref:`changelog` - List of versions and changes over the past.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
