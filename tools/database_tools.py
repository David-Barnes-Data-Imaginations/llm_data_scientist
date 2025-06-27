from smolagents import Tool
from sqlalchemy import create_engine, text


class DatabaseConnect(Tool):
    name = "DatabaseConnect"
    description = "Establish database connection to the turtle games database"
    inputs = {}  # No inputs required
    output_type = "string"

def __init__(self, sandbox=None):
    super().__init__()
    self.sandbox = sandbox

def forward(self) -> str:

    """
        Returns:
            sqlalchemy.engine.Engine: A SQLAlchemy engine object connected to the database.

        Example usage:
        engine = get_db_connection()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM table_name"))
        """
        try:
            engine = create_engine('sqlite:///data/tg_database.db')
            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as e:
            raise Exception(f"Failed to connect to database: {str(e)}")

class QuerySales(Tool):
    name = "QuerySales"
    description = "Query sales data from the database."

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, platform: str = None):
        """
        Args:
            platform (str, optional): Filter results by platform name. If None, returns all records.

        Returns:
            list: List of sales records with columns:
                - Product (INTEGER): Product identifier
                - Ranking (INTEGER): Sales ranking
                - Platform (TEXT): Gaming platform
                - Year (REAL): Release year
                - Genre (TEXT): Game genre
                - Publisher (TEXT): Game publisher
                - NA_Sales (REAL): North American sales in millions
                - EU_Sales (REAL): European sales in millions
                - Global_Sales (REAL): Global sales in millions

        Example input:
        platform = "Wii" # Returns only the Wii platform sales
        platform = None # Returns all sales records
        """
        import pandas as pd
        engine = DatabaseConnect()
        with engine.connect() as conn:
            if platform:
                result = conn.execute(pd.read_sql_query("SELECT * FROM tg_sales_table WHERE Platform = :platform"),
                                    {"platform": platform})
            else:
                result = conn.execute(pd.read_sql_query("SELECT * FROM tg_sales_table"))
            return result.fetchall()

class QueryReviews(Tool):
    name = "QueryReviews"
    description = "Query review data from the database."

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, platform: str = None):
        """
        Args:
            platform (str, optional): Filter results by platform name. If None, returns all records.

        Returns:
            list: List of review records with columns:
                - gender (TEXT): Reviewer's gender
                - age (INTEGER): Reviewer's age
                - remuneration (k£) (REAL): Reviewer's income in thousands of pounds
                - spending_score (1-100) (INTEGER): Customer spending score
                - loyalty_points (INTEGER): Customer loyalty points
                - education (TEXT): Reviewer's education level
                - language (TEXT): Reviewer's language
                - platform (TEXT): Gaming platform
                - product (INTEGER): Product identifier
                - review (TEXT): Full review text
                - summary (TEXT): Review summary

        Example input:
        platform = "PS4" # Returns only PS4 platform reviews
        platform = None # Returns all review records
        """
        import pandas as pd
        engine = DatabaseConnect()
        with engine.connect() as conn:
            if platform:
                result = conn.execute(pd.read_sql_query("SELECT * FROM tg_reviews_table WHERE Platform = :platform"),
                                    {"platform": platform})
            else:
                result = conn.execute(pd.read_sql_query("SELECT * FROM tg_reviews_table"))
            return result.fetchall()
