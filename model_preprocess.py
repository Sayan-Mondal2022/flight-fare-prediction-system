# flight_preprocess.py
import pandas as pd
import numpy as np

# =========================
# Preprocessing Function
# =========================
def preprocess_custom_input(input_df, feature_columns):
    """
    Preprocess custom input exactly like training data
    """
    processed_df = input_df.copy()

    # Class encoding
    class_priority = {
        'Economy': 1, 
        'Premium Economy': 2, 
        'First': 3, 
        'Business': 4
    }
    processed_df['Class_Encoded'] = processed_df['Class'].map(class_priority)

    # Stops encoding
    stops_priority = {
        '2+-stop': 0, 
        '1-stop': 1, 
        'non-stop': 2
    }
    processed_df['Stops_Encoded'] = processed_df['Total_stops'].map(stops_priority)

    # Arrival time encoding
    arrival_priority = {
        'Before 6 AM': 0, 
        'After 6 PM': 1, 
        '6 AM - 12 PM': 2, 
        '12 PM - 6 PM': 3
    }
    processed_df['Arrival_Encoded'] = processed_df['Arrival'].map(arrival_priority)

    # Departure time encoding
    departure_priority = {
        'Before 6 AM': 0, 
        'After 6 PM': 1, 
        '6 AM - 12 PM': 2, 
        '12 PM - 6 PM': 3
    }
    processed_df['Departure_Encoded'] = processed_df['Departure'].map(departure_priority)

    # Journey day encoding
    days_priority = {
        'Tuesday': 0, 'Wednesday': 0, 
        'Saturday': 1, 
        'Monday': 2, 'Thursday': 2, 
        'Friday': 3, 'Sunday': 3
    }
    processed_df['Days_Encoded'] = processed_df['Journey_day'].map(days_priority)

    # Drop original columns
    columns_to_drop = ['Journey_day', 'Class', 'Total_stops', 'Arrival', 'Departure']
    processed_df = processed_df.drop([col for col in columns_to_drop if col in processed_df.columns], axis=1)

    # Add missing training columns
    for col in feature_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    # Reorder columns
    processed_df = processed_df[feature_columns]

    return processed_df

# Example usage:
# feature_columns = [...]  # List of feature columns used in training
