"""Controller layer: widget input, caching and orchestration.

The only layer that talks to both ``models`` and ``views``. Uses
``st.cache_resource`` / ``st.cache_data`` for loaded artifacts and results.
"""
