from .index import register_index
from .predict import register_predict
from .batch import register_batch
from .explain import register_explain
from .graph import register_graph_routes
from .check import register_check



def init_routes(app):
    register_index(app)
    register_predict(app)
    register_batch(app)
    register_explain(app)
    register_graph_routes(app)
    register_check(app)
