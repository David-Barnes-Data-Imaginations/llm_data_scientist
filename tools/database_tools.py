from smolagents import Tool
from sqlalchemy import create_engine, text

class DatabaseConnect(Tool):
    name = "database_connect"
    description = "Establish database connection to the turtle games database and test connectivity"
    inputs = {}
    output_type = "string"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self) -> str:
        try:
            engine = create_engine('sqlite:///data/tg_database.db')
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return "Successfully connected to database: sqlite:///data/tg_database.db"
        except Exception as e:
            return f"Failed to connect to database: {str(e)}"

class QuerySales(Tool):
    name = "query_sales"
    description = """Query sales data from the database with flexible filtering and column selection.
    
    Available columns: Product, Ranking, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, Global_Sales
    """
    inputs = {
        "columns": {"type": "string", "description": "Comma-separated list of columns to return (e.g., 'Platform,Global_Sales') or '*' for all columns", "optional": True, "nullable": True},
        "where_column": {"type": "string", "description": "Column name to filter by (e.g., 'Platform', 'Genre', 'Year')", "optional": True, "nullable": True},
        "where_value": {"type": "string", "description": "Value to filter for in the where_column", "optional": True, "nullable": True},
        "limit": {"type": "integer", "description": "Maximum number of records to return", "optional": True, "nullable": True},
        "order_by": {"type": "string", "description": "Column to sort by (e.g., 'Global_Sales DESC')", "optional": True, "nullable": True}
    }

    output_type = "string"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, columns: str = "*", where_column: str = None, where_value: str = None,
                limit: int = None, order_by: str = None) -> str:

        """
        Query sales data with flexible filtering and column selection.

        Args:
            columns: Columns to return, comma-separated or '*' for all
            where_column: Column name to filter by
            where_value: Value to filter for
            limit: Maximum number of records
            order_by: Column to sort by (can include ASC/DESC)

        Example usage:
        - query_sales(columns="Platform,Global_Sales", where_column="Platform", where_value="Wii")
        - query_sales(columns="*", where_column="Year", where_value="2009", limit=10)
        - query_sales(columns="Genre,Global_Sales", order_by="Global_Sales DESC", limit=5)
        """

        try:
            engine = create_engine('sqlite:///data/tg_database.db')

            valid_columns = ["Product", "Ranking", "Platform", "Year", "Genre", "Publisher",
                         "NA_Sales", "EU_Sales", "Global_Sales"]

             # Validate and clean column names
            column_mapping = {
                "Product": "\"product\"",
                "Platform": "\"platform\"",
                "Year": "\"year\"",
                "Genre": "\"genre\"",
                "Publisher": "\"publisher\"",
                "NA_Sales": "\"na_sales\"",
                "EU_Sales": "\"eu_sales\"",
                "Global_Sales": "\"global_sales\""
            }

            if columns == "*":
                select_clause = "*"
            else:
                # Clean and validate column names
                requested_cols = [col.strip() for col in columns.split(",")]
                mapped_cols = []
                for col in requested_cols:
                    if col in column_mapping:
                        mapped_cols.append(column_mapping[col])
                    elif col in valid_columns:
                        mapped_cols.append(col)
                    elif f'"{col}"' in valid_columns:
                        mapped_cols.append(f'"{col}"')
                    else:
                        return f"Invalid column: {col}. Valid columns: gender, age, remuneration, spending_score, loyalty_points, education, language, platform, product, review, summary"

                select_clause = ", ".join(mapped_cols)

            # Build query
            query = f"SELECT {select_clause} FROM tg_reviews_table"
            params = {}

            # Add WHERE clause if specified
            if where_column and where_value:
                # Map column name if needed
                mapped_where_col = column_mapping.get(where_column, where_column)
                if mapped_where_col not in valid_columns and where_column not in [col.strip('"') for col in valid_columns]:
                    return f"Invalid where_column: {where_column}"

                query += f" WHERE {mapped_where_col} = :where_value"
                params["where_value"] = where_value

            # Add ORDER BY clause if specified
            if order_by:
                order_parts = order_by.split()
                mapped_order_col = column_mapping.get(order_parts[0], order_parts[0])
                query += f" ORDER BY {mapped_order_col}"
                if len(order_parts) > 1:
                    query += f" {order_parts[1]}"

            # Add LIMIT if specified
            if limit:
                query += f" LIMIT {limit}"

            # Execute query
            with engine.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                columns_returned = result.keys()

                if not rows:
                    return "No review data found for the specified criteria"

                # Format results as string
                output = f"Found {len(rows)} review records:\n"
                output += f"Columns: {list(columns_returned)}\n\n"

                # Show sample of results
                sample_size = min(5, len(rows))  # Fewer for reviews as they can be long
                for i, row in enumerate(rows[:sample_size]):
                    row_dict = dict(zip(columns_returned, row))
                    # Truncate long review text for display
                    if 'review' in row_dict and row_dict['review']:
                        row_dict['review'] = row_dict['review'][:100] + "..." if len(str(row_dict['review'])) > 100 else row_dict['review']
                    output += f"Row {i+1}: {row_dict}\n"

                if len(rows) > sample_size:
                    output += f"\n... and {len(rows) - sample_size} more rows"

                return output

        except Exception as e:
            return f"Error querying review data: {str(e)}"

class QueryReviews(Tool):
    name = "query_reviews"
    description = """Query review data from the database with flexible filtering and column selection.
    
    Available columns: gender, age, remuneration (k£), spending_score (1-100), loyalty_points, 
    education, language, platform, product, review, summary
    """
    inputs = {
        "columns": {"type": "string", "description": "Comma-separated list of columns to return (e.g., 'Platform,Global_Sales') or '*' for all columns", "optional": True, "nullable": True},
        "where_column": {"type": "string", "description": "Column name to filter by (e.g., 'Platform', 'Genre', 'Year')", "optional": True, "nullable": True},
        "where_value": {"type": "string", "description": "Value to filter for in the where_column", "optional": True, "nullable": True},
        "limit": {"type": "integer", "description": "Maximum number of records to return", "optional": True, "nullable": True},
        "order_by": {"type": "string", "description": "Column to sort by (e.g., 'Global_Sales DESC')", "optional": True, "nullable": True}
    }

    output_type = "string"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, columns: str = "*", where_column: str = None, where_value: str = None,
                limit: int = None, order_by: str = None) -> str:
        """
        Query review data with flexible filtering and column selection.

        Example usage:
        - query_reviews(columns="platform,age,review", where_column="platform", where_value="PS4")
        - query_reviews(columns="*", where_column="age", where_value="25", limit=5)
        - query_reviews(columns="education,spending_score", order_by="spending_score DESC")
        """
        try:
            engine = create_engine('sqlite:///data/tg_database.db')

            # Note: SQLite column names with spaces/special chars need quotes
            valid_columns = ["gender", "age", "\"remuneration (k£)\"", "\"spending_score (1-100)\"",
                             "loyalty_points", "education", "language", "platform", "product",
                             "review", "summary"]

            # For user-friendly input, map common names
            column_mapping = {
                "remuneration": "\"remuneration (k£)\"",
                "spending_score": "\"spending_score (1-100)\"",
                "remuneration (k£)": "\"remuneration (k£)\"",
                "spending_score (1-100)": "\"spending_score (1-100)\""
            }

            if columns == "*":
                select_clause = "*"
            else:
                # Clean and validate column names
                requested_cols = [col.strip() for col in columns.split(",")]
                mapped_cols = []
                for col in requested_cols:
                    if col in column_mapping:
                        mapped_cols.append(column_mapping[col])
                    elif col in valid_columns:
                        mapped_cols.append(col)
                    elif f'"{col}"' in valid_columns:
                        mapped_cols.append(f'"{col}"')
                    else:
                        return f"Invalid column: {col}. Valid columns: gender, age, remuneration, spending_score, loyalty_points, education, language, platform, product, review, summary"

                select_clause = ", ".join(mapped_cols)

            # Build query
            query = f"SELECT {select_clause} FROM tg_reviews_table"
            params = {}

            # Add WHERE clause if specified
            if where_column and where_value:
                # Map column name if needed
                mapped_where_col = column_mapping.get(where_column, where_column)
                if mapped_where_col not in valid_columns and where_column not in [col.strip('"') for col in valid_columns]:
                    return f"Invalid where_column: {where_column}"

                query += f" WHERE {mapped_where_col} = :where_value"
                params["where_value"] = where_value

            # Add ORDER BY clause if specified
            if order_by:
                order_parts = order_by.split()
                mapped_order_col = column_mapping.get(order_parts[0], order_parts[0])
                query += f" ORDER BY {mapped_order_col}"
                if len(order_parts) > 1:
                    query += f" {order_parts[1]}"

            # Add LIMIT if specified
            if limit:
                query += f" LIMIT {limit}"

            # Execute query
            with engine.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                columns_returned = result.keys()

                if not rows:
                    return "No review data found for the specified criteria"

                # Format results as string
                output = f"Found {len(rows)} review records:\n"
                output += f"Columns: {list(columns_returned)}\n\n"

                # Show sample of results
                sample_size = min(5, len(rows))  # Fewer for reviews as they can be long
                for i, row in enumerate(rows[:sample_size]):
                    row_dict = dict(zip(columns_returned, row))
                    # Truncate long review text for display
                    if 'review' in row_dict and row_dict['review']:
                        row_dict['review'] = row_dict['review'][:100] + "..." if len(str(row_dict['review'])) > 100 else row_dict['review']
                    output += f"Row {i+1}: {row_dict}\n"

                if len(rows) > sample_size:
                    output += f"\n... and {len(rows) - sample_size} more rows"

                return output

        except Exception as e:
            return f"Error querying review data: {str(e)}"

