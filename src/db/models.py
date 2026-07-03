"""SQLAlchemy models for prediction-history logging."""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    model_name: Mapped[str] = mapped_column(String(255))
    model_version: Mapped[str] = mapped_column(String(255))
    endpoint: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str] = mapped_column(String(64), index=True)
    entity_context = mapped_column(JSON, nullable=True)
    horizon: Mapped[int] = mapped_column(Integer, default=1)
    request_payload = mapped_column(JSON)
    scenario_adjustments = mapped_column(JSON, nullable=True)
    prediction = mapped_column(JSON)
    feature_columns = mapped_column(JSON)
    warnings = mapped_column(JSON)
    duration_ms: Mapped[float] = mapped_column(Float)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "endpoint": self.endpoint,
            "entity_id": self.entity_id,
            "entity_context": self.entity_context,
            "horizon": self.horizon,
            "request_payload": self.request_payload,
            "scenario_adjustments": self.scenario_adjustments,
            "prediction": self.prediction,
            "feature_columns": self.feature_columns,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
        }
