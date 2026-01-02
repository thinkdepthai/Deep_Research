"""Provider namespace for pluggable search backends.

Modules in this package can self-register search providers with
`search_factory.register_provider` or expose a PROVIDER object that
implements the SearchProvider protocol.
"""
