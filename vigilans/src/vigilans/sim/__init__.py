"""The observation-level simulator. Knows the truth, so the pipeline may never import it.

Only :mod:`vigilans.sources.sim` may import this package (spec §6); an architecture test
enforces that. The RF-level simulator that exercises adapters belongs to Audiens.
"""
