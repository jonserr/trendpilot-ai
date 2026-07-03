"""Optional database layer for prediction-history logging.

Enabled only when the DATABASE_URL environment variable is set. All entry
points are fail-soft: inference never depends on database availability.
"""
