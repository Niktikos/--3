from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, 
    date_format, 
    hour, 
    udf, 
    avg, 
    when, 
    count
)
from pyspark.sql.types import IntegerType
import os

def full_function(free_slots):
    """
     Визначає, чи станція знаходиться в заповненому стані на основі кількості вільних слотів.
    
    Аргументи:
        free_slots: Кількість вільних слотів на станції
        
    Повертає:
        1, якщо станція заповнена (free_slots = 0), 0 в іншому випадку
    """
    return 1 if free_slots == 0 else 0

def read_register_data(spark, register_path):
    """
    Зчитує файл register.csv у DataFrame.
    
    Аргументи:
        spark: SparkSession
        input_path: Шлях до файлу register.csv
        
    Повертає:
        DataFrame з даними реєстру
    """
    return spark.read.csv(register_path, header=True, sep='\t', 
                           inferSchema=True)

def filter_corrupted_data(df):
    """
      Відфільтровує пошкоджені дані, де і free_slots, і used_slots дорівнюють 0.
    
    Аргументи:
        df: DataFrame з даними реєстру
        
    Повертає:
        DataFrame з видаленими пошкодженими даними
    """
    return df.filter("used_slots > 0 OR free_slots > 0")

def create_timeslot_status_df(filtered_df):
    """
    Створює DataFrame з полями: station, dayofweek, hour та fullstatus.
    
    Аргументи:
        df: DataFrame з очищеними даними реєстру
        
    Повертає:
        DataFrame з інформацією про часовий інтервал та статус
    """
    return (filtered_df
            .select(
                col("station"),
                date_format("timestamp", "EEEE").alias("dayofweek"),
                hour("timestamp").alias("hour"),
                udf(full_function, IntegerType())("free_slots").alias("fullstatus")
            ))

def calculate_criticality(timeslot_df):
    """
    Обчислює критичність для кожної групи (station, dayofweek, hour).
    Критичність визначається як середнє значення fullstatus.
    
    Аргументи:
        df: DataFrame з інформацією про часовий інтервал та статус
        
    Повертає:
        DataFrame з обчисленою критичністю для кожної групи
    """
    return (timeslot_df
            .groupBy("station", "dayofweek", "hour")
            .agg(avg("fullstatus").alias("criticality")))

def filter_by_threshold(criticality_df, threshold):
    """
    Відфільтровує рядки, де критичність перевищує вказаний поріг.
    
    Аргументи:
        df: DataFrame з обчисленою критичністю
        threshold: Мінімальний поріг критичності (за замовчуванням: 0.25)
        
    Повертає:
        DataFrame лише з рядками, що перевищують поріг
    """
    return criticality_df.filter(f"criticality > {threshold}")

def read_stations_data(spark, stations_path):
    """
    Зчитує файл stations.csv у DataFrame.
    
    Аргументи:
        spark: SparkSession
        input_path: Шлях до файлу stations.csv
        
    Повертає:
        DataFrame з даними станцій
    """
    return spark.read.csv(stations_path, header=True, sep='\t', 
                           inferSchema=True)

def join_with_stations(critical_df, stations_df):
    """
    Об'єднує критичні часові інтервали з таблицею станцій для отримання координат станцій.
    
    Аргументи:
        critical_df: DataFrame з критичними часовими інтервалами
        stations_df: DataFrame з даними станцій
        
    Повертає:
        Об'єднаний DataFrame з координатами станцій
    """
    return critical_df.join(
        stations_df, 
        critical_df.station == stations_df.id
    ).select(
        stations_df.id,  # Keep the original id column
        critical_df.station,  # Keep the original station column
        critical_df.dayofweek, 
        critical_df.hour, 
        critical_df.criticality,
        stations_df.longitude, 
        stations_df.latitude
    )

def sort_and_save_results(joined_df, output_path):
    """
      Сортує результати та зберігає їх у CSV-файл.
    
    Порядок сортування:
    1. Критичність (за спаданням)
    2. ID станції (за зростанням)
    3. День тижня (за зростанням)
    4. Година (за зростанням)
    
    Аргументи:
        df: Об'єднаний DataFrame з критичними часовими інтервалами та інформацією про станції
        output_path: Шлях для збереження результатів
        
    Повертає:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Sort DataFrame
    sorted_df = joined_df.orderBy(
        col("criticality").desc(), 
        col("station").asc(), 
        col("dayofweek").asc(), 
        col("hour").asc()
    )
    
    # Save to CSV
    sorted_df.write.csv(
        output_path, 
        mode='overwrite', 
        header=True, 
        sep=','
    )
