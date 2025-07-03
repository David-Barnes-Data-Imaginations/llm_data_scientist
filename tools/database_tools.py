from smolagents import Tool
from sqlalchemy import create_engine, text
from src.client.telemetry import TelemetryManager
from langfuse import observe, get_client


class DatabaseConnect(Tool):
    name = "DatabaseConnect"
    description = "Establish database connection to the turtle games database and test connectivity"
    inputs = {}
    output_type = "string"
    help_notes = """ 
    DatabaseConnect: 
    A tool that establishes and tests a connection to the Turtle Games SQLite database.
    Use this to verify database connectivity before attempting to query data.
    This is useful as a first step when working with database data to ensure the database is accessible.

    Example usage: 

    connection_status = DatabaseConnect().forward()
    print(connection_status)  # "Successfully connected to database: sqlite:///data/tg_database.db"
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    @observe
    def forward(self) -> str:
        telemetry = TelemetryManager()
        langfuse = get_client()
        trace = telemetry.start_trace("database_connect", {
            "database": "sqlite:///data/tg_database.db"
        })

        try:
            engine = create_engine('sqlite:///data/tg_database.db')

            telemetry.log_event(trace, "processing", {
                "step": "connecting_to_database"
            })

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            # Log success
            telemetry.log_event(trace, "success", {
                "message": "Successfully connected to database"
            })

            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            return "Successfully connected to database: sqlite:///data/tg_database.db"

        except Exception as e:
            # Log error
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            return f"Failed to connect to database: {str(e)}"
        finally:
            # Always finish the trace
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            telemetry.finish_trace(trace)

class QuerySales(Tool):
    name = "QuerySales"
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
    help_notes = """ 
    QuerySales: 
    A tool that queries the sales data from the Turtle Games database with flexible filtering and column selection.
    Use this to retrieve and analyze sales data across different platforms, genres, years, and regions.

    Available columns: Product, Ranking, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, Global_Sales

    Example usage: 

    # Get sales data for Wii platform
    wii_sales = QuerySales().forward(columns="Platform,Global_Sales", where_column="Platform", where_value="Wii")

    # Get top 10 games from 2009 by sales
    top_2009_games = QuerySales().forward(columns="*", where_column="Year", where_value="2009", limit=10)

    # Get top 5 genres by global sales
    top_genres = QuerySales().forward(columns="Genre,Global_Sales", order_by="Global_Sales DESC", limit=5)
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
    @observe
    def forward(self, columns: str = "*", where_column: str = None, where_value: str = None,
                limit: int = None, order_by: str = None) -> str:
        telemetry = TelemetryManager()
        langfuse = get_client()
        trace = telemetry.start_trace("query_sales", {
            "columns": columns,
            "where_column": where_column,
            "where_value": where_value,
            "limit": limit,
            "order_by": order_by
        })
        try:
            engine = create_engine('sqlite:///data/tg_database.db')

            telemetry.log_event(trace, "processing", {
                "step": "initializing_query",
                "database": "sqlite:///data/tg_database.db"
            })

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
                telemetry.log_event(trace, "processing", {
                    "step": "validating_columns",
                    "requested_columns": columns
                })

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
                        telemetry.log_event(trace, "error", {
                            "error_type": "ValidationError",
                            "error_message": f"Invalid column: {col}"
                        })
                        langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
                        return f"Invalid column: {col}. Valid columns: gender, age, remuneration, spending_score, loyalty_points, education, language, platform, product, review, summary"

                select_clause = ", ".join(mapped_cols)

            # Build query
            query = f"SELECT {select_clause} FROM tg_reviews_table"
            params = {}

            telemetry.log_event(trace, "processing", {
                "step": "building_query",
                "base_query": query
            })

            # Add WHERE clause if specified
            if where_column and where_value:
                telemetry.log_event(trace, "processing", {
                    "step": "adding_where_clause",
                    "where_column": where_column,
                    "where_value": where_value
                })

                # Map column name if needed
                mapped_where_col = column_mapping.get(where_column, where_column)
                if mapped_where_col not in valid_columns and where_column not in [col.strip('"') for col in valid_columns]:
                    telemetry.log_event(trace, "error", {
                        "error_type": "ValidationError",
                        "error_message": f"Invalid where_column: {where_column}"
                    })
                    langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
                    return f"Invalid where_column: {where_column}"

                query += f" WHERE {mapped_where_col} = :where_value"
                params["where_value"] = where_value

            # Add ORDER BY clause if specified
            if order_by:
                telemetry.log_event(trace, "processing", {
                    "step": "adding_order_by",
                    "order_by": order_by
                })

                order_parts = order_by.split()
                mapped_order_col = column_mapping.get(order_parts[0], order_parts[0])
                query += f" ORDER BY {mapped_order_col}"
                if len(order_parts) > 1:
                    query += f" {order_parts[1]}"

            # Add LIMIT if specified
            if limit:
                telemetry.log_event(trace, "processing", {
                    "step": "adding_limit",
                    "limit": limit
                })

                query += f" LIMIT {limit}"

            telemetry.log_event(trace, "processing", {
                "step": "executing_query",
                "final_query": query
            })

            # Execute query
            with engine.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                columns_returned = result.keys()

                if not rows:
                    telemetry.log_event(trace, "success", {
                        "result": "no_data_found"
                    })
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

                telemetry.log_event(trace, "success", {
                    "rows_found": len(rows),
                    "columns_returned": str(list(columns_returned))
                })

                langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
                return output

        except Exception as e:
            # Log error
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            return f"Error querying review data: {str(e)}"
        finally:
            # Always finish the trace
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            telemetry.finish_trace(trace)

class QueryReviews(Tool):
    name = "QueryReviews"
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
    help_notes = """ 
    QueryReviews: 
    A tool that queries customer review data from the Turtle Games database with flexible filtering and column selection.
    Use this to retrieve and analyze customer feedback, demographics, and spending patterns.

    Available columns: gender, age, remuneration (k£), spending_score (1-100), loyalty_points, 
    education, language, platform, product, review, summary

    Example usage: 

    # Get reviews for PS4 platform
    ps4_reviews = QueryReviews().forward(columns="platform,age,review", where_column="platform", where_value="PS4")

    # Get reviews from 25-year-old customers
    young_adult_reviews = QueryReviews().forward(columns="*", where_column="age", where_value="25", limit=5)

    # Get reviews sorted by spending score (highest first)
    high_spender_reviews = QueryReviews().forward(columns="education,spending_score", order_by="spending_score DESC")
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    @observe
    def forward(self, columns: str = "*", where_column: str = None, where_value: str = None,
                limit: int = None, order_by: str = None) -> str:
        telemetry = TelemetryManager()
        langfuse = get_client()
        trace = telemetry.start_trace("query_reviews", {
            "columns": columns,
            "where_column": where_column,
            "where_value": where_value,
            "limit": limit,
            "order_by": order_by
        })
        """
        Query review data with flexible filtering and column selection.

        Example usage:
        - query_reviews(columns="platform,age,review", where_column="platform", where_value="PS4")
        - query_reviews(columns="*", where_column="age", where_value="25", limit=5)
        - query_reviews(columns="education,spending_score", order_by="spending_score DESC")
        """
        try:
            engine = create_engine('sqlite:///data/tg_database.db')

            telemetry.log_event(trace, "processing", {
                "step": "initializing_query",
                "database": "sqlite:///data/tg_database.db"
            })

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
                telemetry.log_event(trace, "processing", {
                    "step": "validating_columns",
                    "requested_columns": columns
                })

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
                        telemetry.log_event(trace, "error", {
                            "error_type": "ValidationError",
                            "error_message": f"Invalid column: {col}"
                        })
                        langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
                        return f"Invalid column: {col}. Valid columns: gender, age, remuneration, spending_score, loyalty_points, education, language, platform, product, review, summary"

                select_clause = ", ".join(mapped_cols)

            # Build query
            query = f"SELECT {select_clause} FROM tg_reviews_table"
            params = {}

            telemetry.log_event(trace, "processing", {
                "step": "building_query",
                "base_query": query
            })

            # Add WHERE clause if specified
            if where_column and where_value:
                telemetry.log_event(trace, "processing", {
                    "step": "adding_where_clause",
                    "where_column": where_column,
                    "where_value": where_value
                })

                # Map column name if needed
                mapped_where_col = column_mapping.get(where_column, where_column)
                if mapped_where_col not in valid_columns and where_column not in [col.strip('"') for col in valid_columns]:
                    telemetry.log_event(trace, "error", {
                        "error_type": "ValidationError",
                        "error_message": f"Invalid where_column: {where_column}"
                    })
                    langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
                    return f"Invalid where_column: {where_column}"

                query += f" WHERE {mapped_where_col} = :where_value"
                params["where_value"] = where_value

            # Add ORDER BY clause if specified
            if order_by:
                telemetry.log_event(trace, "processing", {
                    "step": "adding_order_by",
                    "order_by": order_by
                })

                order_parts = order_by.split()
                mapped_order_col = column_mapping.get(order_parts[0], order_parts[0])
                query += f" ORDER BY {mapped_order_col}"
                if len(order_parts) > 1:
                    query += f" {order_parts[1]}"

            # Add LIMIT if specified
            if limit:
                telemetry.log_event(trace, "processing", {
                    "step": "adding_limit",
                    "limit": limit
                })

                query += f" LIMIT {limit}"

            telemetry.log_event(trace, "processing", {
                "step": "executing_query",
                "final_query": query
            })

            # Execute query
            with engine.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                columns_returned = result.keys()

                if not rows:
                    telemetry.log_event(trace, "success", {
                        "result": "no_data_found"
                    })
                    langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
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

                telemetry.log_event(trace, "success", {
                    "rows_found": len(rows),
                    "columns_returned": str(list(columns_returned))
                })

                langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
                return output

        except Exception as e:
            # Log error
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            return f"Error querying review data: {str(e)}"

        finally:
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            telemetry.finish_trace(trace)
