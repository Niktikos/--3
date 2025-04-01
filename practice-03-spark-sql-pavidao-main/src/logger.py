"""
Apache Spark uses log4j (v2) package under the hood.
Therefore, to properly pass logging messages and unify them across multiple APIs,
we'll need to use it instead of Python's usual logging package for Python applications.

In this code we do the following:

* Get the LogManager object via py4j bridge
* Configure the optional custom_prefix
* Return the initialized logger object
"""

from pyspark.sql import SparkSession
from typing import Optional


class Log4py():
    """
    Wrapper class for Log4j.
    """

    def __init__(self,
                 spark: SparkSession,
                 custom_prefix: Optional[str] = "") -> None:
        config = spark.conf
        if not custom_prefix:
            app_id = config.get('spark.app.id')
            app_name = config.get('spark.app.name')
            custom_prefix = f"<{app_name} {app_id}>"

        log4j = spark._jvm.org.apache.log4j
        self._logger = log4j.LogManager.getLogger(custom_prefix)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warn(self, message: str) -> None:
        self._logger.warn(message)
