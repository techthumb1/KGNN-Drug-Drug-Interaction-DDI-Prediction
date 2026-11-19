def run_pgexplainer(*args, **kwargs):
    raise NotImplementedError(
        "PGExplainer requires a trained and evaluated "
        "explanation model. Use /graph_data for "
        "recorded source relationships."
    )