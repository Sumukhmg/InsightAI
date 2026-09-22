"""InsightAI DuckDB In-Memory Analytical Database
Provides high-performance analytical SQL query execution over retail datasets.
"""

from typing import Any, Dict, List, Optional
import duckdb
import pandas as pd


class DatabaseManager:
    """Manages an in-process DuckDB analytical connection and registered views."""
    
    def __init__(self):
        self.conn = duckdb.connect(database=":memory:")
        self._registered_tables = set()

    def register_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Registers a pandas DataFrame as a queryable DuckDB relation."""
        self.conn.register(table_name, df)
        self._registered_tables.add(table_name)

    def query(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Executes an analytical SQL query and returns results as a pandas DataFrame."""
        try:
            if params:
                return self.conn.execute(sql, params).df()
            return self.conn.execute(sql).df()
        except Exception as e:
            raise RuntimeError(f"DuckDB Query Execution Error: {e}\nSQL: {sql}")

    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """Returns the schema (column names and data types) for a registered table."""
        if table_name not in self._registered_tables:
            return []
        schema_df = self.query(f"DESCRIBE {table_name}")
        return schema_df.to_dict(orient="records")

    def run_fast_kpis(self, table_name: str = "cleaned_data") -> Dict[str, Any]:
        """Calculates executive KPIs in a single fast SQL scan."""
        sql = f"""
        SELECT
            COUNT(*) AS total_records,
            COALESCE(SUM(Revenue), 0) AS total_revenue,
            COUNT(DISTINCT Invoice) AS total_orders,
            COUNT(DISTINCT "Customer ID") AS total_customers,
            COUNT(DISTINCT StockCode) AS total_products,
            COALESCE(SUM(Quantity), 0) AS total_quantity,
            COALESCE(AVG(Revenue), 0) AS avg_item_revenue,
            COALESCE(SUM(Revenue) / NULLIF(COUNT(DISTINCT Invoice), 0), 0) AS avg_order_value
        FROM {table_name}
        """
        res = self.query(sql).iloc[0].to_dict()
        return {
            "total_records": int(res["total_records"]),
            "total_revenue": float(res["total_revenue"]),
            "total_orders": int(res["total_orders"]),
            "total_customers": int(res["total_customers"]),
            "total_products": int(res["total_products"]),
            "total_quantity": int(res["total_quantity"]),
            "avg_item_revenue": float(res["avg_item_revenue"]),
            "avg_order_value": float(res["avg_order_value"])
        }

    def close(self):
        """Closes the DuckDB database connection."""
        self.conn.close()


# Singleton database instance
_db_instance: Optional[DatabaseManager] = None


def get_database(df: Optional[pd.DataFrame] = None) -> DatabaseManager:
    """Returns the global DatabaseManager instance, registering the DataFrame if provided."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    if df is not None:
        _db_instance.register_dataframe("cleaned_data", df)
    return _db_instance
