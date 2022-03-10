.. automl documentation master file, created on Sun Jan  9 11:57:43 2022.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

    auto_examples/index

.. module:: automl


OptimizePipeline
==================================

.. autoclass:: automl.OptimizePipeline
   :members:
        __init__,
        fit,
        add_model,
        update_model_space,
        remove_model,
        change_child_iteration,
        get_best_metric,
        get_best_metric_iteration,
        get_best_pipeline_by_metric,
        get_best_pipeline_by_model,
        baseline_results,
        dumbbell_plot,
        taylor_plot,
        _build_model,
        compare_models,
        cleanup,
        post_fit,
        bfe_all_best_models,
        bfe_best_model_from_scratch,
        bfe_model_from_scratch,
        be_best_model_from_config,
        from_config,
        from_config_file,
        config,
        report,
        save_results,
   :undoc-members:
   :show-inheritance:
