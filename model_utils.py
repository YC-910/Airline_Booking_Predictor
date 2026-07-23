from __future__ import annotations

import numpy as np
import pandas as pd


REGRESSION_FEATURES = [
    "airline_id",
    "country",
    "airline_tier",
    "service_rating",
    "cabin_class",
    "seat_type",
    "flight_duration_hours",
    "booking_channel",
    "meals_included",
    "payment_type",
    "passenger_age",
    "lead_time_days",
    "departure_month",
    "is_weekend",
    "is_public_holiday",
    "load_factor_flight",
    "competitor_avg_fare_rm",
    "distance_km",
    "promo_applied",
    "events_index_destination",
    "departure_day_of_week",
    "departure_quarter",
    "route_speed_kmh",
    "demand_pressure",
    "lead_time_group",
    "last_minute_booking",
    "long_haul_flight"
]


CLASSIFICATION_FEATURES = [
    "airline_id",
    "country",
    "airline_tier",
    "service_rating",
    "cabin_class",
    "flight_duration_hours",
    "booking_channel",
    "passenger_age",
    "lead_time_days",
    "departure_month",
    "is_weekend",
    "is_public_holiday",
    "load_factor_flight",
    "distance_km",
    "events_index_destination",
    "departure_day_of_week",
    "departure_quarter",
    "route_speed_kmh",
    "demand_pressure",
    "lead_time_group",
    "last_minute_booking",
    "long_haul_flight"
]


def engineer_airline_features(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Create the same engineered features used during model training.
    """

    engineered = data.copy()

    engineered["departure_date"] = pd.to_datetime(
        engineered["departure_date"],
        errors="coerce"
    )

    if engineered["departure_date"].isna().any():
        raise ValueError(
            "A valid departure date is required."
        )

    engineered["departure_month"] = (
        engineered["departure_date"].dt.month
    )

    engineered["is_weekend"] = (
        engineered[
            "departure_date"
        ].dt.dayofweek >= 5
    ).astype(int)

    engineered["departure_day_of_week"] = (
        engineered[
            "departure_date"
        ].dt.day_name()
    )

    engineered["departure_quarter"] = (
        "Q"
        + engineered[
            "departure_date"
        ].dt.quarter.astype(str)
    )

    duration = pd.to_numeric(
        engineered["flight_duration_hours"],
        errors="coerce"
    )

    distance = pd.to_numeric(
        engineered["distance_km"],
        errors="coerce"
    )

    if (duration <= 0).any():
        raise ValueError(
            "Flight duration must be greater than zero."
        )

    if (distance <= 0).any():
        raise ValueError(
            "Distance must be greater than zero."
        )

    engineered["route_speed_kmh"] = (
        distance / duration
    )

    engineered["demand_pressure"] = (
        pd.to_numeric(
            engineered["load_factor_flight"],
            errors="coerce"
        )
        * pd.to_numeric(
            engineered[
                "events_index_destination"
            ],
            errors="coerce"
        )
    )

    engineered["lead_time_group"] = pd.cut(
        pd.to_numeric(
            engineered["lead_time_days"],
            errors="coerce"
        ),
        bins=[
            -np.inf,
            7,
            30,
            90,
            np.inf
        ],
        labels=[
            "Last Minute (0-7 Days)",
            "Short Lead (8-30 Days)",
            "Medium Lead (31-90 Days)",
            "Long Lead (91+ Days)"
        ]
    )

    engineered["last_minute_booking"] = (
        pd.to_numeric(
            engineered["lead_time_days"],
            errors="coerce"
        ) <= 7
    ).astype(int)

    engineered["long_haul_flight"] = np.where(
        distance >= 3000,
        "Long Haul",
        "Short or Medium Haul"
    )

    return engineered
