import pandas as pd
from datetime import datetime
from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    df_prepared = prepare_data(df, categorical)
    
    print(f"Number of rows in prepared dataframe: {len(df_prepared)}")

    expected_data = [
        {
            'PULocationID': '-1',
            'DOLocationID': '-1',
            'tpep_pickup_datetime': dt(1, 1),
            'tpep_dropoff_datetime': dt(1, 10),
            'duration': 9.0
        },
        {
            'PULocationID': '1',
            'DOLocationID': '1',
            'tpep_pickup_datetime': dt(1, 2),
            'tpep_dropoff_datetime': dt(1, 10),
            'duration': 8.0
        }
    ]

    expected_df = pd.DataFrame(expected_data)

    # Compare the prepared DataFrame with the expected DataFrame
    pd.testing.assert_frame_equal(df_prepared.reset_index(drop=True), expected_df.reset_index(drop=True))

    # Additional assertions
    assert len(df_prepared) == 2, f"Expected 2 rows, but got {len(df_prepared)}"
    assert (df_prepared['duration'] >= 1).all() and (df_prepared['duration'] <= 60).all(), "Duration should be between 1 and 60 minutes"
    assert df_prepared['PULocationID'].dtype == 'object' and df_prepared['DOLocationID'].dtype == 'object', "Categorical columns should be of type 'object' (string)"