# Minimal stub for the Streamlit library used in tests.
# Provides only the attributes accessed by the annotation app.

def cache_data(*, show_spinner=False):
    """Dummy decorator that returns the original function unchanged.
    It mimics ``st.cache_data`` used in the app.
    """
    def decorator(func):
        return func
    return decorator

class _DummyNamespace:
    def __getattr__(self, name):
        # Return a callable that does nothing and returns None.
        def _dummy(*args, **kwargs):
            return None
        return _dummy

# The app does ``import streamlit as st`` and then accesses ``st`` attributes.
st = _DummyNamespace()
