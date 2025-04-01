"""
A module for starting Spark Jobs
"""
from pyspark.sql import SparkSession

from .logger import Log4py


def start_spark(
        number_cores: int = 2,
        memory_gb: int = 4,
        app_name: str = "sample_job",
        spark_config: dict[str, str] = {}) -> tuple[SparkSession, Log4py]:
    """
    Start Spark Context on the local node, get logger and load config if present.
    
    Args:
        number_cores (int): Number of workers to use
        memory_gb (int): Memory to use for running Spark
        app_name (str): Name of Spark app.
        spark_config (dict[str,str]): Dictionary of config key-value pairs.
    
    Returns:
        A tuple of references to the Spark Session and logger.
    """
    # Create a Spark Session object
    builder = (
        SparkSession
        .builder
        .appName(app_name)
        .master(f"local[{number_cores}]")
        .config("spark.driver.memory", f"{memory_gb}g")
    )

    for k, v in spark_config.items():
        builder.config(k, v)

    spark = builder.getOrCreate()
    logger = Log4py(spark)

    logger.warn("Context Started.")

    return spark, logger
