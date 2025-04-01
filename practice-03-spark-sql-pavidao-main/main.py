import os
import json
import shutil
from src.spark import start_spark
from src.tasks import (
    full_function,
    read_register_data,
    filter_corrupted_data,
    create_timeslot_status_df,
    calculate_criticality,
    filter_by_threshold,
    read_stations_data,
    join_with_stations,
    sort_and_save_results
)

if __name__ == "__main__":
    # Ініціалізація Spark сесії
    spark, logger = start_spark(number_cores=2, memory_gb=1)
    
    # Реєстрація UDF для визначення, чи станція заповнена
    spark.udf.register("full", full_function)
    
    # Встановлення шляхів
    register_path = "input/register.csv"
    stations_path = "input/stations.csv"
    output_path = "output"
    
    # Встановлення порогу критичності
    threshold = 0.25
    
    logger.warn("Зчитування даних реєстру...")
    register_df = read_register_data(spark, register_path)
    
    logger.warn("Фільтрація пошкоджених даних...")
    filtered_df = filter_corrupted_data(register_df)
    
    logger.warn("Створення DataFrame з часовими інтервалами та статусами...")
    timeslot_df = create_timeslot_status_df(filtered_df)
    
    logger.warn("Обчислення критичності...")
    criticality_df = calculate_criticality(timeslot_df)
    
    logger.warn(f"Фільтрація за порогом {threshold}...")
    critical_df = filter_by_threshold(criticality_df, threshold)
    
    logger.warn("Зчитування даних станцій...")
    stations_df = read_stations_data(spark, stations_path)
    
    logger.warn("Об'єднання з даними станцій...")
    joined_df = join_with_stations(critical_df, stations_df)
    
    logger.warn("Сортування та збереження результатів...")
    
    # Очищення вихідної директорії, якщо вона існує
    if os.path.exists(output_path) and os.path.isdir(output_path):
        shutil.rmtree(output_path)
    
    # Збереження результатів
    sort_and_save_results(joined_df, output_path)
    
    logger.warn("Завдання успішно виконано!")
    
    # Зупинка Spark сесії
    spark.stop()
