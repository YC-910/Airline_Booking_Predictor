from __future__ import annotations

from datetime import date
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from model_utils import (
    CLASSIFICATION_FEATURES,
    REGRESSION_FEATURES,
    engineer_airline_features
)


APP_DIRECTORY = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Airline Booking Predictor",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# SMALL AMOUNT OF STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1120px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stForm"] {
            border: 1px solid #E4E7EC;
            border-radius: 16px;
            padding: 1rem 1.15rem 1.2rem;
            background: #FFFFFF;
        }

        [data-testid="stMetric"] {
            border: 1px solid #E4E7EC;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: #FFFFFF;
        }

        .useful-note {
            padding: 0.9rem 1rem;
            border: 1px solid #E4E7EC;
            border-radius: 12px;
            background: #F9FAFB;
            color: #475467;
            line-height: 1.5;
        }

        /* Remove top-right Streamlit toolbar */
        [data-testid="stToolbar"],
        [data-testid="stToolbarActions"],
        [data-testid="stHeaderActionElements"],
        [data-testid="stAppDeployButton"] {
            display: none !important;
        }

        /* Remove the three-dot menu */
        #MainMenu {
            visibility: hidden !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS, METADATA AND RANKINGS
# ============================================================

@st.cache_resource
def load_assets():
    regression_model = joblib.load(
        APP_DIRECTORY
        / "best_ticket_price_model.joblib"
    )

    classification_model = joblib.load(
        APP_DIRECTORY
        / "best_delay_status_model.joblib"
    )

    with open(
        APP_DIRECTORY
        / "deployment_metadata.json",
        "r",
        encoding="utf-8"
    ) as metadata_file:
        metadata = json.load(
            metadata_file
        )

    regression_ranking = pd.read_csv(
        APP_DIRECTORY
        / "regression_model_ranking.csv"
    )

    classification_ranking = pd.read_csv(
        APP_DIRECTORY
        / "classification_model_ranking.csv"
    )

    return (
        regression_model,
        classification_model,
        metadata,
        regression_ranking,
        classification_ranking
    )


(
    regression_model,
    classification_model,
    metadata,
    regression_ranking,
    classification_ranking
) = load_assets()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def numeric_default(
    column: str
) -> float:
    return float(
        metadata[
            "numeric_defaults"
        ][column]["median"]
    )


def get_model_name(
    task: str
) -> str:
    key_options = {
        "regression": [
            "best_regression_model",
            "regression_model"
        ],
        "classification": [
            "best_classification_model",
            "classification_model"
        ]
    }

    for key in key_options[task]:
        value = metadata.get(key)

        if value:
            return str(value)

    return "Unavailable"


def get_model_row(
    ranking_table: pd.DataFrame,
    selected_model: str
) -> pd.Series:
    matching_rows = ranking_table[
        ranking_table["Model"].astype(str)
        == selected_model
    ]

    if not matching_rows.empty:
        return matching_rows.iloc[0]

    return ranking_table.iloc[0]


def create_useful_guidance(
    predicted_price: float,
    competitor_fare: float,
    disruption_risk: float
) -> tuple[str, str, str, str]:
    if competitor_fare > 0:
        price_difference_percentage = (
            (
                predicted_price
                - competitor_fare
            )
            / competitor_fare
            * 100
        )
    else:
        price_difference_percentage = 0.0

    if price_difference_percentage > 10:
        fare_level = "warning"
        fare_message = (
            "The predicted fare is more than 10% above the "
            "competitor fare. Review whether the airline offers "
            "enough service or schedule value to support the price."
        )
    elif price_difference_percentage < -10:
        fare_level = "info"
        fare_message = (
            "The predicted fare is more than 10% below the "
            "competitor fare. The price may be competitive, but "
            "the airline should also check whether revenue is being "
            "left on the table."
        )
    else:
        fare_level = "success"
        fare_message = (
            "The predicted fare is within 10% of the competitor "
            "fare, suggesting that it is broadly aligned with the "
            "market reference."
        )

    if disruption_risk >= 0.60:
        operation_level = "error"
        operation_message = (
            "High disruption risk: prioritise operational monitoring, "
            "review staffing or aircraft availability, and prepare "
            "proactive passenger communication."
        )
    elif disruption_risk >= 0.40:
        operation_level = "warning"
        operation_message = (
            "Moderate disruption risk: monitor the flight and prepare "
            "contingency communication if conditions worsen."
        )
    else:
        operation_level = "success"
        operation_message = (
            "Lower disruption risk: normal monitoring may be suitable, "
            "although the prediction is not a guarantee."
        )

    return (
        fare_level,
        fare_message,
        operation_level,
        operation_message
    )


def show_message(
    message_level: str,
    message: str
) -> None:
    display_functions = {
        "success": st.success,
        "info": st.info,
        "warning": st.warning,
        "error": st.error
    }

    display_functions[
        message_level
    ](message)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("✈️ Airline Booking Predictor")

st.write(
    "Enter one booking profile to estimate the ticket price "
    "and the probability of an **On-Time**, **Delayed**, or "
    "**Cancelled** flight."
)

st.caption(
    "All fields below are used by at least one of the two models. "
    "Advanced fields already contain typical values, so they can "
    "be left unchanged when the information is unavailable."
)


# ============================================================
# PREDICTION FORM
# ============================================================

with st.form(
    "prediction_form",
    clear_on_submit=False
):
    st.subheader("1. Main booking details")

    left_column, right_column = st.columns(2)

    with left_column:
        airline_id = st.selectbox(
            "Airline ID",
            metadata[
                "categories"
            ]["airline_id"]
        )

        country = st.selectbox(
            "Country",
            metadata[
                "categories"
            ]["country"]
        )

        airline_tier = st.selectbox(
            "Airline tier",
            metadata[
                "categories"
            ]["airline_tier"]
        )

        cabin_class = st.selectbox(
            "Cabin class",
            metadata[
                "categories"
            ]["cabin_class"]
        )

        departure_date = st.date_input(
            "Departure date",
            value=date.today()
        )

        distance_km = st.number_input(
            "Route distance (km)",
            min_value=1.0,
            value=float(
                round(
                    numeric_default(
                        "distance_km"
                    ),
                    1
                )
            ),
            step=10.0
        )

        flight_duration_hours = st.number_input(
            "Flight duration (hours)",
            min_value=0.10,
            value=float(
                round(
                    numeric_default(
                        "flight_duration_hours"
                    ),
                    2
                )
            ),
            step=0.10
        )

    with right_column:
        passenger_age = st.number_input(
            "Passenger age",
            min_value=0,
            max_value=100,
            value=int(
                round(
                    numeric_default(
                        "passenger_age"
                    )
                )
            ),
            step=1
        )

        booking_channel = st.selectbox(
            "Booking channel",
            metadata[
                "categories"
            ]["booking_channel"]
        )

        lead_time_days = st.number_input(
            "Days booked before departure",
            min_value=0,
            value=int(
                round(
                    numeric_default(
                        "lead_time_days"
                    )
                )
            ),
            step=1,
            help=(
                "For example, enter 5 when the ticket is booked "
                "five days before departure."
            )
        )

        competitor_avg_fare_rm = st.number_input(
            "Competitor average fare (RM)",
            min_value=0.0,
            value=float(
                round(
                    numeric_default(
                        "competitor_avg_fare_rm"
                    ),
                    2
                )
            ),
            step=10.0
        )

        load_factor_percent = st.slider(
            "Expected seat occupancy (%)",
            min_value=0,
            max_value=100,
            value=int(
                round(
                    numeric_default(
                        "load_factor_flight"
                    )
                    * 100
                )
            ),
            step=1,
            help=(
                "The expected percentage of seats occupied "
                "on the flight."
            )
        )

        events_index_destination = st.slider(
            "Destination congestion or event index",
            min_value=0,
            max_value=100,
            value=int(
                round(
                    numeric_default(
                        "events_index_destination"
                    )
                )
            ),
            step=1,
            help=(
                "Higher values represent stronger event activity "
                "or congestion at the destination."
            )
        )

    with st.expander(
        "2. Advanced details",
        expanded=False
    ):
        advanced_1, advanced_2, advanced_3 = (
            st.columns(3)
        )

        with advanced_1:
            service_rating = st.slider(
                "Service rating",
                min_value=1.0,
                max_value=5.0,
                value=float(
                    round(
                        numeric_default(
                            "service_rating"
                        ),
                        1
                    )
                ),
                step=0.1
            )

            seat_type = st.selectbox(
                "Seat type",
                metadata[
                    "categories"
                ]["seat_type"]
            )

        with advanced_2:
            meals_included = st.selectbox(
                "Meals included",
                metadata[
                    "categories"
                ]["meals_included"]
            )

            payment_type = st.selectbox(
                "Payment type",
                metadata[
                    "categories"
                ]["payment_type"]
            )

        with advanced_3:
            is_public_holiday = st.checkbox(
                "Public holiday"
            )

            promo_applied = st.checkbox(
                "Promotion applied"
            )

    submitted = st.form_submit_button(
        "Predict ticket price and flight status",
        type="primary",
        use_container_width=True
    )


# ============================================================
# MAKE THE PREDICTION
# ============================================================

if submitted:
    try:
        load_factor_flight = (
            load_factor_percent / 100
        )

        booking_record = pd.DataFrame({
            "airline_id": [airline_id],
            "country": [country],
            "airline_tier": [airline_tier],
            "service_rating": [
                service_rating
            ],
            "cabin_class": [
                cabin_class
            ],
            "seat_type": [seat_type],
            "flight_duration_hours": [
                flight_duration_hours
            ],
            "booking_channel": [
                booking_channel
            ],
            "meals_included": [
                meals_included
            ],
            "payment_type": [
                payment_type
            ],
            "passenger_age": [
                passenger_age
            ],
            "lead_time_days": [
                lead_time_days
            ],
            "departure_date": [
                pd.Timestamp(
                    departure_date
                )
            ],
            "is_public_holiday": [
                int(
                    is_public_holiday
                )
            ],
            "load_factor_flight": [
                load_factor_flight
            ],
            "competitor_avg_fare_rm": [
                competitor_avg_fare_rm
            ],
            "distance_km": [
                distance_km
            ],
            "promo_applied": [
                int(
                    promo_applied
                )
            ],
            "events_index_destination": [
                events_index_destination
            ]
        })

        engineered_record = (
            engineer_airline_features(
                booking_record
            )
        )

        regression_input = (
            engineered_record[
                REGRESSION_FEATURES
            ]
        )

        classification_input = (
            engineered_record[
                CLASSIFICATION_FEATURES
            ]
        )

        predicted_price = float(
            regression_model.predict(
                regression_input
            )[0]
        )

        predicted_status = str(
            classification_model.predict(
                classification_input
            )[0]
        )

        probabilities = (
            classification_model
            .predict_proba(
                classification_input
            )[0]
        )

        probability_table = pd.DataFrame({
            "Flight Status": (
                classification_model.classes_
            ),
            "Probability": probabilities
        }).sort_values(
            by="Probability",
            ascending=False
        ).reset_index(
            drop=True
        )

        probability_lookup = dict(
            zip(
                classification_model.classes_,
                probabilities
            )
        )

        predicted_confidence = float(
            probability_lookup.get(
                predicted_status,
                probability_table[
                    "Probability"
                ].iloc[0]
            )
        )

        disruption_risk = float(
            probability_lookup.get(
                "Delayed",
                0.0
            )
            + probability_lookup.get(
                "Cancelled",
                0.0
            )
        )

        route_speed = float(
            engineered_record[
                "route_speed_kmh"
            ].iloc[0]
        )

        if (
            route_speed < 100
            or route_speed > 1200
        ):
            st.warning(
                "The distance and duration produce an unusual "
                f"average route speed of {route_speed:,.0f} km/h. "
                "Check these two inputs before relying on the result."
            )

        (
            fare_level,
            fare_message,
            operation_level,
            operation_message
        ) = create_useful_guidance(
            predicted_price=predicted_price,
            competitor_fare=(
                competitor_avg_fare_rm
            ),
            disruption_risk=(
                disruption_risk
            )
        )

        result_summary = pd.DataFrame({
            "Predicted_Ticket_Price_RM": [
                round(
                    predicted_price,
                    2
                )
            ],
            "Predicted_Delay_Status": [
                predicted_status
            ],
            "Prediction_Confidence_Percentage": [
                round(
                    predicted_confidence
                    * 100,
                    2
                )
            ],
            "Disruption_Risk_Percentage": [
                round(
                    disruption_risk
                    * 100,
                    2
                )
            ],
            "Competitor_Fare_RM": [
                competitor_avg_fare_rm
            ],
            "Lead_Time_Days": [
                lead_time_days
            ],
            "Route_Distance_KM": [
                distance_km
            ],
            "Flight_Duration_Hours": [
                flight_duration_hours
            ]
        })

        st.session_state[
            "prediction_result"
        ] = {
            "predicted_price": (
                predicted_price
            ),
            "predicted_status": (
                predicted_status
            ),
            "predicted_confidence": (
                predicted_confidence
            ),
            "disruption_risk": (
                disruption_risk
            ),
            "probability_table": (
                probability_table
            ),
            "fare_level": fare_level,
            "fare_message": fare_message,
            "operation_level": (
                operation_level
            ),
            "operation_message": (
                operation_message
            ),
            "result_summary": (
                result_summary
            )
        }

    except Exception as error:
        st.error(
            "The prediction could not be completed: "
            + str(error)
        )


# ============================================================
# DISPLAY USEFUL RESULTS
# ============================================================

result = st.session_state.get(
    "prediction_result"
)

if result:
    st.divider()

    st.subheader("Prediction result")

    result_1, result_2, result_3 = (
        st.columns(3)
    )

    result_1.metric(
        "Estimated ticket price",
        (
            f"RM "
            f"{result['predicted_price']:,.2f}"
        )
    )

    result_2.metric(
        "Most likely flight status",
        result[
            "predicted_status"
        ]
    )

    result_3.metric(
        "Confidence in selected status",
        (
            f"{result['predicted_confidence']:.2%}"
        )
    )

    st.metric(
        "Combined disruption risk",
        (
            f"{result['disruption_risk']:.2%}"
        ),
        help=(
            "This is the combined probability of the flight "
            "being Delayed or Cancelled."
        )
    )

    st.subheader("Useful interpretation")

    show_message(
        result[
            "fare_level"
        ],
        result[
            "fare_message"
        ]
    )

    show_message(
        result[
            "operation_level"
        ],
        result[
            "operation_message"
        ]
    )

    probability_column, table_column = (
        st.columns([1.3, 1])
    )

    with probability_column:
        st.subheader(
            "Probability by flight status"
        )

        chart_data = (
            result[
                "probability_table"
            ]
            .set_index(
                "Flight Status"
            )[["Probability"]]
        )

        st.bar_chart(
            chart_data
        )

    with table_column:
        st.subheader(
            "Probability details"
        )

        probability_display = (
            result[
                "probability_table"
            ].copy()
        )

        probability_display[
            "Probability (%)"
        ] = (
            probability_display[
                "Probability"
            ]
            .mul(100)
            .round(2)
        )

        probability_display = (
            probability_display[
                [
                    "Flight Status",
                    "Probability (%)"
                ]
            ]
        )

        probability_display.index = range(
            1,
            len(
                probability_display
            ) + 1
        )

        probability_display.index.name = (
            "Rank"
        )

        st.dataframe(
            probability_display,
            use_container_width=True
        )

    prediction_csv = (
        result[
            "result_summary"
        ]
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        "Download prediction summary",
        data=prediction_csv,
        file_name=(
            "airline_booking_prediction.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# COMPACT MODEL INFORMATION
# ============================================================

with st.expander(
    "Model performance and interpretation"
):
    regression_model_name = (
        get_model_name(
            "regression"
        )
    )

    classification_model_name = (
        get_model_name(
            "classification"
        )
    )

    regression_row = get_model_row(
        regression_ranking,
        regression_model_name
    )

    classification_row = get_model_row(
        classification_ranking,
        classification_model_name
    )

    st.markdown(
        f"**Ticket-price model:** "
        f"{regression_model_name}"
    )

    regression_metrics = st.columns(4)

    if "MAE" in regression_row:
        regression_metrics[0].metric(
            "MAE",
            f"RM {float(regression_row['MAE']):,.2f}"
        )

    if "RMSE" in regression_row:
        regression_metrics[1].metric(
            "RMSE",
            f"RM {float(regression_row['RMSE']):,.2f}"
        )

    if "MAPE" in regression_row:
        regression_metrics[2].metric(
            "MAPE",
            f"{float(regression_row['MAPE']):.2f}%"
        )

    if "R2" in regression_row:
        regression_metrics[3].metric(
            "R²",
            f"{float(regression_row['R2']):.2f}"
        )

    st.caption(
        "MAE shows the average fare error. RMSE gives more "
        "weight to large errors. MAPE expresses the average "
        "error as a percentage."
    )

    st.markdown(
        f"**Delay-status model:** "
        f"{classification_model_name}"
    )

    classification_metrics = st.columns(4)

    if "Accuracy" in classification_row:
        classification_metrics[0].metric(
            "Accuracy",
            (
                f"{float(classification_row['Accuracy']):.2%}"
            )
        )

    if "Balanced_Accuracy" in classification_row:
        classification_metrics[1].metric(
            "Balanced accuracy",
            (
                f"{float(classification_row['Balanced_Accuracy']):.2%}"
            )
        )

    if "F1_Macro" in classification_row:
        classification_metrics[2].metric(
            "Macro F1",
            (
                f"{float(classification_row['F1_Macro']):.2%}"
            )
        )

    if "ROC_AUC_OVR_Macro" in classification_row:
        classification_metrics[3].metric(
            "ROC-AUC",
            (
                f"{float(classification_row['ROC_AUC_OVR_Macro']):.2%}"
            )
        )

    st.caption(
        "The Cancelled class is smaller than the other classes, "
        "so Macro F1 and balanced accuracy are useful alongside "
        "ordinary accuracy."
    )

    st.markdown(
        """
        <div class="useful-note">
            The application is an academic decision-support tool.
            Its outputs are predictions rather than guaranteed
            prices or flight outcomes.
        </div>
        """,
        unsafe_allow_html=True
    )
