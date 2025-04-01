import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType, StringType, DoubleType
import pandas as pd
import os
from datetime import datetime

from src.tasks import (
    full_function,
    filter_corrupted_data,
    create_timeslot_status_df,
    calculate_criticality,
    filter_by_threshold,
    join_with_stations,
    sort_and_save_results
)

@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("testing")
        .getOrCreate()
    )
    spark.udf.register("full", full_function)
    yield spark
    spark.stop()

@pytest.fixture
def register_data(spark):
    """Create sample register data for testing."""
    schema = StructType([
        StructField("station", IntegerType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("used_slots", IntegerType(), True),
        StructField("free_slots", IntegerType(), True)
    ])
    
    # Create sample data including normal and corrupted rows
    data = [
        # Normal rows for station 1
        (1, datetime(2023, 1, 1, 12, 0, 0), 3, 2),  # Sunday at 12:00
        (1, datetime(2023, 1, 1, 12, 30, 0), 5, 0),  # Sunday at 12:30 (full)
        (1, datetime(2023, 1, 1, 13, 0, 0), 3, 2),   # Sunday at 13:00
        # Normal rows for station 2
        (2, datetime(2023, 1, 2, 14, 0, 0), 2, 3),   # Monday at 14:00
        (2, datetime(2023, 1, 2, 14, 30, 0), 5, 0),  # Monday at 14:30 (full)
        (2, datetime(2023, 1, 2, 14, 59, 0), 5, 0),   # Monday at 14:59 (full)
        # Corrupted row - both free_slots and used_slots are 0
        (3, datetime(2023, 1, 3, 16, 0, 0), 0, 0)
    ]
    
    return spark.createDataFrame(data, schema)

@pytest.fixture
def stations_data(spark):
    """Create sample stations data for testing."""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("name", StringType(), True)
    ])
    
    data = [
        (1, 30.5, 50.5, "Station 1"),
        (2, 31.0, 51.0, "Station 2"),
        (3, 31.5, 51.5, "Station 3")
    ]
    
    return spark.createDataFrame(data, schema)

def test_full_function():
    """Test the full_function returns correct values."""
    assert full_function(0) == 1, "Should return 1 when free_slots is 0"
    assert full_function(1) == 0, "Should return 0 when free_slots is greater than 0"
    assert full_function(5) == 0, "Should return 0 when free_slots is greater than 0"

def test_filter_corrupted_data(spark, register_data):
    """Test filtering of corrupted data."""
    filtered_df = filter_corrupted_data(register_data)
    
    # Check that corrupted rows are removed
    assert filtered_df.count() == 6, "Should filter out the corrupted row"
    
    # Check that no rows have both used_slots and free_slots equal to 0
    corrupted_count = filtered_df.filter("used_slots = 0 AND free_slots = 0").count()
    assert corrupted_count == 0, "Should not contain rows with both used_slots = 0 and free_slots = 0"

def test_create_timeslot_status_df(spark, register_data):
    """Test creating timeslot status DataFrame."""
    # First filter out corrupted data
    filtered_df = filter_corrupted_data(register_data)
    
    # Then create timeslot status DataFrame
    timeslot_df = create_timeslot_status_df(filtered_df)
    
    # Check the schema
    assert "station" in timeslot_df.columns, "Should contain station column"
    assert "dayofweek" in timeslot_df.columns, "Should contain dayofweek column"
    assert "hour" in timeslot_df.columns, "Should contain hour column"
    assert "fullstatus" in timeslot_df.columns, "Should contain fullstatus column"
    
    # Check the data
    data = timeslot_df.collect()
    assert len(data) == 6, "Should have the same number of rows as filtered data"

def test_calculate_criticality(spark, register_data):
    """Test calculation of criticality."""
    # Prepare data
    filtered_df = filter_corrupted_data(register_data)
    timeslot_df = create_timeslot_status_df(filtered_df)
    
    # Calculate criticality
    criticality_df = calculate_criticality(timeslot_df)
    
    # Check the schema
    assert "station" in criticality_df.columns, "Should contain station column"
    assert "dayofweek" in criticality_df.columns, "Should contain dayofweek column"
    assert "hour" in criticality_df.columns, "Should contain hour column"
    assert "criticality" in criticality_df.columns, "Should contain criticality column"
    
    # Check criticality calculation (for example, station 2 has 2/3 fullstatus=1)
    station2_data = criticality_df.filter("station = 2").collect()
    assert len(station2_data) > 0, "Should have data for station 2"
    
    # Station 2 has 2 full slots out of 3 (66.7% criticality)
    for row in station2_data:
        if row["hour"] == 14 or row["hour"] == 15:
            assert abs(row["criticality"] - 0.667) < 0.1, "Criticality should be approximately 0.667 for station 2"

def test_filter_by_threshold(spark, register_data):
    """Test filtering by threshold."""
    # Prepare data
    filtered_df = filter_corrupted_data(register_data)
    timeslot_df = create_timeslot_status_df(filtered_df)
    criticality_df = calculate_criticality(timeslot_df)
    
    # Filter by threshold
    threshold = 0.5
    critical_df = filter_by_threshold(criticality_df, threshold)
    
    # Check filtering
    for row in critical_df.collect():
        assert row["criticality"] > threshold, f"All criticality values should be greater than {threshold}"

def test_join_with_stations(spark, register_data, stations_data):
    """Test joining with stations data."""
    # Prepare data
    filtered_df = filter_corrupted_data(register_data)
    timeslot_df = create_timeslot_status_df(filtered_df)
    criticality_df = calculate_criticality(timeslot_df)
    critical_df = filter_by_threshold(criticality_df, 0.25)
    
    # Join with stations
    joined_df = join_with_stations(critical_df, stations_data)
    
    # Check joined data
    assert "longitude" in joined_df.columns, "Should contain longitude column from stations"
    assert "latitude" in joined_df.columns, "Should contain latitude column from stations"
    
    # Verify the join was correct
    for row in joined_df.collect():
        assert row["station"] == row["id"], "Station ID should match in the joined data"

def test_sort_and_save_results(spark, register_data, stations_data, tmp_path):
    """Test sorting and saving results."""
    # Prepare data
    filtered_df = filter_corrupted_data(register_data)
    timeslot_df = create_timeslot_status_df(filtered_df)
    criticality_df = calculate_criticality(timeslot_df)
    critical_df = filter_by_threshold(criticality_df, 0.25)
    joined_df = join_with_stations(critical_df, stations_data)

    output_path = tmp_path / "output"
    
    # Sort and save
    sort_and_save_results(joined_df, str(output_path.resolve()))
    
    # Check output file existence
    output_files = [f for f in os.listdir(str(output_path.resolve())) if f.endswith(".csv")]
    assert len(output_files) > 0, "Should create at least one CSV file"
    
    # Load the saved data to check sorting
    saved_df = spark.read.csv(f"{str(output_path.resolve())}", header=True, inferSchema=True)
    
    # Convert to pandas for easier assertion
    pandas_df = saved_df.toPandas()
    
    # Check that data is sorted by criticality descending
    if len(pandas_df) > 1:
        for i in range(len(pandas_df) - 1):
            if pandas_df.iloc[i]["criticality"] == pandas_df.iloc[i+1]["criticality"]:
                # If criticality is the same, check station ID sorting
                if pandas_df.iloc[i]["station"] == pandas_df.iloc[i+1]["station"]:
                    # If station ID is also the same, check dayofweek sorting
                    if pandas_df.iloc[i]["dayofweek"] == pandas_df.iloc[i+1]["dayofweek"]:
                        # If dayofweek is also the same, check hour sorting
                        assert pandas_df.iloc[i]["hour"] <= pandas_df.iloc[i+1]["hour"], "Hours should be sorted in ascending order"
                    else:
                        # Compare dayofweek
                        # Note: We would need a mapping from dayofweek strings to numbers to properly check this
                        pass
                else:
                    assert pandas_df.iloc[i]["station"] <= pandas_df.iloc[i+1]["station"], "Station IDs should be sorted in ascending order"
            else:
                assert pandas_df.iloc[i]["criticality"] >= pandas_df.iloc[i+1]["criticality"], "Criticality should be sorted in descending order"

