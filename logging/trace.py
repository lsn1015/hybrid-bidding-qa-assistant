# logging/trace.py
import uuid
import contextvars

trace_id_var = contextvars.ContextVar("trace_id", default=None)

def new_trace():
    trace_id = str(uuid.uuid4())
    trace_id_var.set(trace_id)
    return trace_id

def get_trace():
    return trace_id_var.get()


