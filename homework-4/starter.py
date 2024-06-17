import os
import pickle
import pandas as pd
import argparse

# Function to read data
def read_data(filename):
    categorical = ['PULocationID', 'DOLocationID']
    df = pd.read_parquet(filename)
    
    # Calculate trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    # Filter out trips with unrealistic durations
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Fill missing values and convert to string
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df, categorical

# Main function
def main(year, month):
    # Load the model and DictVectorizer
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    # Read the data
    filename = f'data/yellow_tripdata_{year}-{month:02d}.parquet'
    df, categorical = read_data(filename)

    # Convert categorical columns to dictionary format
    dicts = df[categorical].to_dict(orient='records')

    # Transform the data using the DictVectorizer
    X_val = dv.transform(dicts)

    # Predict using the model
    y_pred = model.predict(X_val)

    # Create an artificial ride_id column
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Create a DataFrame with the results
    df_result = df[['ride_id']].copy()
    df_result['predicted_duration'] = y_pred

    # Save the results as a parquet file
    output_file = f'predictions_{year}_{month:02d}.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    # Get the size of the output file
    file_size = os.path.getsize(output_file)
    print(f"The size of the output file is {file_size} bytes.")

    # Print the mean predicted duration
    mean_duration = y_pred.mean()
    print(f"The mean predicted duration is {mean_duration:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict trip durations.')
    parser.add_argument('--year', type=int, required=True, help='Year of the trip data')
    parser.add_argument('--month', type=int, required=True, help='Month of the trip data')
    args = parser.parse_args()
    
    main(args.year, args.month)