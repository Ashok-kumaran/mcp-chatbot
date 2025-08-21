"""
Provides a suite of MCP tools for agents/chatbots to query and analyze
a default SAP HANA table with intelligent insights.
"""

import os
import json
import csv
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from difflib import get_close_matches
from dotenv import load_dotenv
from hdbcli import dbapi
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Environment & MCP setup
# ---------------------------------------------------------------------
load_dotenv()
mcp = FastMCP("SAPHANA_DB")

HANA_HOST = os.getenv("HANA_HOST")
HANA_PORT = int(os.getenv("HANA_PORT", "443"))
HANA_USER = os.getenv("HANA_USER")
HANA_PASS = os.getenv("HANA_PASS")
HANA_SCHEMA = os.getenv("HANA_SCHEMA")
HANA_TABLE_NAME = os.getenv("HANA_TABLE_NAME")
HANA_CERTIFICATE = os.getenv("HANA_CERTIFICATE")

# ---------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------
try:
    conn = dbapi.connect(
        address=HANA_HOST,
        port=HANA_PORT,
        user=HANA_USER,
        password=HANA_PASS,
        encrypt=True,
        sslValidateCertificate=True if HANA_CERTIFICATE else False,
        sslCertificate=HANA_CERTIFICATE if HANA_CERTIFICATE else None
    )
    logger.info(
        f"✅ Connected to HANA schema '{HANA_SCHEMA}', table '{HANA_TABLE_NAME}' "
        f"on {HANA_HOST}:{HANA_PORT}"
    )
except Exception as e:
    logger.error(f"❌ DB connection failed: {e}")
    exit(1)

# ---------------------------------------------------------------------
# Globals (schema cache)
# ---------------------------------------------------------------------
VALID_TABLES: set = set()
VALID_COLUMNS: set = set()
COLUMN_TYPES: Dict[str, str] = {}

# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------
@dataclass
class QueryInsight:
    summary: str
    key_findings: List[str]
    data_quality_notes: List[str]
    recommendations: List[str]
    raw_data: Any

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def validate_column(col: str):
    """Validate a column name against the loaded schema cache."""
    if col is None:
        raise ValueError("Column name is required.")
    if col.upper() not in VALID_COLUMNS:
        suggestion = get_close_matches(col.upper(), VALID_COLUMNS, n=1)
        error_msg = f"Invalid column: '{col}'"
        if suggestion:
            error_msg += f". Did you mean '{suggestion[0]}'?"
        raise ValueError(error_msg)

def load_valid_tables():
    """Load valid table names from HANA into VALID_TABLES."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT DISTINCT TABLE_NAME
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = ?
            """,
            (HANA_SCHEMA,),
        )
        global VALID_TABLES
        VALID_TABLES = set(row[0] for row in cursor.fetchall())
        logger.debug(f"Loaded {len(VALID_TABLES)} valid tables.")
    finally:
        cursor.close()

def load_valid_columns():
    """Load columns and types for default table into VALID_COLUMNS and COLUMN_TYPES."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE_NAME
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?
            """,
            (HANA_SCHEMA, HANA_TABLE_NAME),
        )
        global VALID_COLUMNS, COLUMN_TYPES
        results = cursor.fetchall()
        VALID_COLUMNS = set(row[0].upper() for row in results)
        COLUMN_TYPES = {row[0].upper(): row[1] for row in results}
        logger.debug(f"Loaded {len(VALID_COLUMNS)} valid columns.")
    finally:
        cursor.close()

# ---------------------------------------------------------------------
# DataAnalyzer class (intelligent analysis)
# ---------------------------------------------------------------------
class DataAnalyzer:
    """Intelligent data analysis and response generation."""

    @staticmethod
    def analyze_data_sample(data: List[Dict], table_name: str, columns: List[str]) -> QueryInsight:
        if not data:
            return QueryInsight(
                summary=f"No data found in table {table_name}",
                key_findings=["Table appears to be empty"],
                data_quality_notes=["Consider checking data loading processes"],
                recommendations=["Verify data ingestion pipelines"],
                raw_data=data,
            )

        findings = []
        quality_notes = []
        recommendations = []

        row_count = len(data)
        findings.append(f"Retrieved {row_count} records from {table_name}")

        # Missing values
        null_counts = {}
        for col in columns:
            null_count = sum(1 for row in data if row.get(col) is None or row.get(col) == "")
            if null_count > 0:
                null_counts[col] = null_count

        if null_counts:
            findings.append(f"Data quality: {len(null_counts)} columns have missing values")
            for col, count in list(null_counts.items())[:3]:
                quality_notes.append(
                    f"Column '{col}' has {count} null/empty values ({count/row_count*100:.1f}%)"
                )
        else:
            findings.append("Data quality: No missing values detected in sample")

        # Value distribution insights for first 3 columns
        for col in columns[:3]:
            values = [row.get(col) for row in data if row.get(col) is not None]
            unique_values = len(set(values))
            if unique_values == len(values) and len(values) > 0:
                findings.append(f"Column '{col}' contains unique values (possibly an ID field)")
            elif unique_values < len(values) * 0.5:
                findings.append(
                    f"Column '{col}' has high duplication ({unique_values} unique out of {len(values)})"
                )

        # Recommendations
        if row_count < 100:
            recommendations.append("Consider increasing the sample size for better analysis")
        if null_counts:
            recommendations.append("Review data quality and consider data cleansing processes")
        recommendations.append("Use filter_data or search_data for more targeted analysis")

        summary = (
            f"Analysis of {table_name}: {row_count} records across {len(columns)} columns "
            f"with {'good' if not null_counts else 'mixed'} data quality"
        )

        return QueryInsight(summary, findings, quality_notes, recommendations, data)

    @staticmethod
    def analyze_search_results(results: List[Dict], search_term: str, columns_searched: List[str]) -> QueryInsight:
        if not results:
            return QueryInsight(
                summary=f"No matches found for '{search_term}'",
                key_findings=[f"Search term '{search_term}' not found in {len(columns_searched)} text columns"],
                data_quality_notes=[],
                recommendations=[
                    "Try a broader search term",
                    "Check spelling variations",
                    "Use get_distinct_values to see available values",
                ],
                raw_data=results,
            )

        findings = [
            f"Found {len(results)} records matching '{search_term}'",
            f"Searched across {len(columns_searched)} columns: {', '.join(columns_searched[:3])}{'...' if len(columns_searched) > 3 else ''}",
        ]

        match_distribution: Dict[str, int] = {}
        for result in results:
            for col in columns_searched:
                value = str(result.get(col, ""))
                if search_term.lower() in value.lower():
                    match_distribution[col] = match_distribution.get(col, 0) + 1

        if match_distribution:
            top_match_col = max(match_distribution, key=match_distribution.get)
            findings.append(f"Most matches in column '{top_match_col}' ({match_distribution[top_match_col]} records)")

        recommendations = [
            "Use filter_data for more precise filtering",
            "Consider using get_column_stats for numerical analysis",
        ]

        return QueryInsight(
            summary=f"Search for '{search_term}' returned {len(results)} matches",
            key_findings=findings,
            data_quality_notes=[],
            recommendations=recommendations,
            raw_data=results,
        )

# ---------------------------------------------------------------------
# Response formatter
# ---------------------------------------------------------------------
def format_response(insight: QueryInsight) -> Dict:
    return {
        "analysis": {
            "summary": insight.summary,
            "key_findings": insight.key_findings,
            "data_quality_notes": insight.data_quality_notes,
            "recommendations": insight.recommendations,
        },
        "raw_data": insight.raw_data,
        "metadata": {
            "analysis_timestamp": datetime.now().isoformat(),
            "data_sample_size": len(insight.raw_data) if isinstance(insight.raw_data, list) else 1,
        },
    }

# ---------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------

@mcp.tool()
def refresh_schema_cache() -> Dict:
    """Reloads the table and column cache from HANA"""
    load_valid_tables()
    load_valid_columns()
    return {"status": "Schema cache refreshed", "tables": len(VALID_TABLES), "columns": len(VALID_COLUMNS)}

@mcp.tool()
def get_schema() -> Dict:
    """Retrieve the database schema for SAP HANA with intelligent analysis"""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE_NAME, LENGTH, IS_NULLABLE
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?
            ORDER BY POSITION
            """,
            (HANA_SCHEMA, HANA_TABLE_NAME),
        )
        results = cursor.fetchall()
    finally:
        cursor.close()

    schema_info = []
    data_types: Dict[str, int] = {}
    nullable_count = 0

    for table_name, column_name, data_type, length, nullable in results:
        schema_info.append(
            {"name": column_name, "type": data_type.lower(), "length": length, "nullable": nullable == "YES"}
        )
        data_types[data_type] = data_types.get(data_type, 0) + 1
        if nullable == "YES":
            nullable_count += 1

    findings = [
        f"Table '{HANA_TABLE_NAME}' has {len(schema_info)} columns",
        f"Data types: {dict(data_types)}",
        f"{nullable_count} columns allow NULL values",
    ]

    recommendations = [
        "Use get_data to preview actual data",
        "Use get_column_stats for numerical columns",
        "Use get_distinct_values for categorical analysis",
    ]

    insight = QueryInsight(
        summary=f"Schema analysis for {HANA_TABLE_NAME}",
        key_findings=findings,
        data_quality_notes=[],
        recommendations=recommendations,
        raw_data={"table": HANA_TABLE_NAME, "columns": schema_info},
    )

    return format_response(insight)

@mcp.tool()
def get_columns() -> Dict:
    """Get column names with intelligent metadata"""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?
            ORDER BY POSITION
            """,
            (HANA_SCHEMA, HANA_TABLE_NAME),
        )
        results = cursor.fetchall()
    finally:
        cursor.close()

    columns = []
    text_columns = []
    numeric_columns = []
    date_columns = []

    for col_name, data_type, nullable in results:
        columns.append(col_name)
        if data_type in ["VARCHAR", "NVARCHAR", "TEXT", "NTEXT"]:
            text_columns.append(col_name)
        elif data_type in ["INTEGER", "DECIMAL", "DOUBLE", "REAL"]:
            numeric_columns.append(col_name)
        elif data_type in ["DATE", "TIMESTAMP", "TIME"]:
            date_columns.append(col_name)

    findings = [
        f"Total columns: {len(columns)}",
        f"Text columns: {len(text_columns)} (good for search_data)",
        f"Numeric columns: {len(numeric_columns)} (good for get_column_stats)",
        f"Date columns: {len(date_columns)} (good for filter_by_date_range)",
    ]

    recommendations = []
    if text_columns:
        recommendations.append(f"Use search_data on: {', '.join(text_columns[:3])}")
    if numeric_columns:
        recommendations.append(f"Use get_column_stats on: {', '.join(numeric_columns[:3])}")
    if date_columns:
        recommendations.append(f"Use filter_by_date_range on: {', '.join(date_columns[:3])}")

    insight = QueryInsight(
        summary=f"Column analysis for {HANA_TABLE_NAME}",
        key_findings=findings,
        data_quality_notes=[],
        recommendations=recommendations,
        raw_data={
            "table": HANA_TABLE_NAME,
            "all_columns": columns,
            "text_columns": text_columns,
            "numeric_columns": numeric_columns,
            "date_columns": date_columns,
        },
    )

    return format_response(insight)

@mcp.tool()
def get_data(columns: Optional[str] = None, limit: int = 10) -> Dict:
    """Retrieve and analyze sample data from the default HANA table"""
    cursor = conn.cursor()
    try:
        col_clause = "*"
        selected_columns = None

        if columns:
            col_list = [col.strip() for col in columns.split(",")]
            # Validate requested columns
            for c in col_list:
                validate_column(c)
            col_clause = ", ".join(f'"{col}"' for col in col_list)
            selected_columns = col_list

        query = f'SELECT {col_clause} FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}" LIMIT {limit}'
        cursor.execute(query)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
    except dbapi.Error as e:
        raise ValueError(str(e))
    finally:
        cursor.close()

    data = [dict(zip(col_names, row)) for row in rows]
    insight = DataAnalyzer.analyze_data_sample(data, HANA_TABLE_NAME, col_names)
    return format_response(insight)

@mcp.tool()
def get_table_summary() -> Dict:
    """Get comprehensive table summary with insights"""
    cursor = conn.cursor()
    try:
        cursor.execute(f'SELECT COUNT(*) FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"')
        count = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*), 
                   SUM(CASE WHEN DATA_TYPE_NAME IN ('VARCHAR', 'NVARCHAR', 'TEXT') THEN 1 ELSE 0 END) as text_cols,
                   SUM(CASE WHEN DATA_TYPE_NAME IN ('INTEGER', 'DECIMAL', 'DOUBLE') THEN 1 ELSE 0 END) as numeric_cols,
                   SUM(CASE WHEN DATA_TYPE_NAME IN ('DATE', 'TIMESTAMP') THEN 1 ELSE 0 END) as date_cols
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?
            """,
            (HANA_SCHEMA, HANA_TABLE_NAME),
        )

        col_stats = cursor.fetchone()
        total_cols, text_cols, numeric_cols, date_cols = col_stats
    finally:
        cursor.close()

    findings = [
        f"Table contains {count:,} total records",
        f"Column breakdown: {total_cols} total ({text_cols} text, {numeric_cols} numeric, {date_cols} date)",
    ]

    if count == 0:
        findings.append("⚠️ Table is empty")
        recommendations = ["Check data loading processes", "Verify table population"]
    elif count < 1000:
        findings.append("Small dataset - suitable for complete analysis")
        recommendations = ["Use get_data without limits", "Consider get_column_stats on all numeric columns"]
    elif count < 100000:
        findings.append("Medium dataset - use sampling for exploration")
        recommendations = ["Use get_data with appropriate limits", "Use search_data for targeted queries"]
    else:
        findings.append("Large dataset - recommend filtered analysis")
        recommendations = ["Use filter_data for subset analysis", "Use aggregate_data for summaries"]

    quality_notes = []
    if text_cols == 0:
        quality_notes.append("No text columns available for search operations")
    if numeric_cols == 0:
        quality_notes.append("No numeric columns available for statistical analysis")
    if date_cols == 0:
        quality_notes.append("No date columns available for time-based filtering")

    insight = QueryInsight(
        summary=f"Table summary: {count:,} rows across {total_cols} columns",
        key_findings=findings,
        data_quality_notes=quality_notes,
        recommendations=recommendations,
        raw_data={
            "table": HANA_TABLE_NAME,
            "row_count": count,
            "column_stats": {"total": total_cols, "text": text_cols, "numeric": numeric_cols, "date": date_cols},
        },
    )

    return format_response(insight)

@mcp.tool()
def run_sql(query: str) -> Dict:
    """Run custom SQL with intelligent result analysis"""
    if not query.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    # Basic validation - attempt to detect column names and ensure they exist in cache
    sql_functions = {"COUNT", "MAX", "MIN", "AVG", "SUM", "NOW", "UPPER", "LOWER"}
    tokens = query.replace(",", " ").replace("(", " ").replace(")", " ").split()
    potential_columns = [
        t.strip('"') for t in tokens
        if t.isidentifier()
        and t.upper() not in {"SELECT", "FROM", "WHERE", "AND", "OR", "LIMIT", "LIKE", "IN", "IS", "NOT", "NULL", "BETWEEN", "AS", "GROUP", "BY", "ORDER", "DESC", "ASC", "DISTINCT"}
        and t.upper() not in sql_functions
        and t.strip('"') not in VALID_TABLES
    ]

    for col in potential_columns:
        if col.upper() not in VALID_COLUMNS:
            suggestion = get_close_matches(col.upper(), VALID_COLUMNS, n=1)
            if suggestion:
                raise ValueError(f"Invalid column name: '{col}'. Did you mean '{suggestion[0]}'?")
            else:
                raise ValueError(f"Invalid column name: '{col}'")

    cursor = conn.cursor()
    try:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except dbapi.Error as e:
        raise ValueError(str(e))
    finally:
        cursor.close()

    data = [dict(zip(columns, row)) for row in rows]

    findings = [
        "Query executed successfully",
        f"Returned {len(data)} rows with {len(columns)} columns",
    ]

    query_lower = query.lower()
    if "group by" in query_lower:
        findings.append("Aggregation query detected - results show grouped data")
    if "order by" in query_lower:
        findings.append("Results are sorted as specified")
    if "where" in query_lower:
        findings.append("Filtered query - results match specified conditions")

    recommendations = ["Use export_query_results to save these results", "Modify the query to explore different aspects of the data"]

    if len(data) == 0:
        recommendations.append("No results found - try broadening your WHERE conditions")
    elif len(data) > 1000:
        recommendations.append("Large result set - consider adding LIMIT for better performance")

    insight = QueryInsight(
        summary=f"SQL query returned {len(data)} rows",
        key_findings=findings,
        data_quality_notes=[],
        recommendations=recommendations,
        raw_data={"query": query, "columns": columns, "results": data},
    )

    return format_response(insight)

@mcp.tool()
def search_data(
    search_term: str,
    columns: Optional[str] = None,
    limit: int = 20,
    case_sensitive: bool = False
) -> Dict:
    """Smart search with intelligent result analysis"""
    cursor = conn.cursor()
    try:
        if not columns:
            cursor.execute(
                """
                SELECT COLUMN_NAME 
                FROM SYS.TABLE_COLUMNS 
                WHERE SCHEMA_NAME = ? AND TABLE_NAME = ? 
                AND DATA_TYPE_NAME IN ('VARCHAR', 'NVARCHAR', 'TEXT', 'NTEXT')
                """,
                (HANA_SCHEMA, HANA_TABLE_NAME),
            )
            text_columns = [row[0] for row in cursor.fetchall()]
        else:
            text_columns = [col.strip() for col in columns.split(",")]
            for c in text_columns:
                validate_column(c)

        search_conditions = []
        for col in text_columns:
            if case_sensitive:
                search_conditions.append(f'"{col}" LIKE ?')
            else:
                search_conditions.append(f'UPPER("{col}") LIKE UPPER(?)')

        where_clause = " OR ".join(search_conditions)
        search_pattern = f"%{search_term}%"

        query = f'''
            SELECT * FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"
            WHERE {where_clause}
            LIMIT {limit}
        '''

        params = [search_pattern] * len(text_columns)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
    except dbapi.Error as e:
        raise ValueError(str(e))
    finally:
        cursor.close()

    results = [dict(zip(col_names, row)) for row in rows]
    insight = DataAnalyzer.analyze_search_results(results, search_term, text_columns)

    response = format_response(insight)
    response["raw_data"]["search_metadata"] = {
        "search_term": search_term, 
        "columns_searched": text_columns, 
        "case_sensitive": case_sensitive
    }

    return response

@mcp.tool()
def filter_data(
    filters: str,  # JSON string of filters
    sort_by: Optional[str] = None,
    sort_order: str = "ASC",
    limit: int = 50,
) -> Dict:
    """Filter data with intelligent result analysis"""
    try:
        filter_dict = json.loads(filters)
    except json.JSONDecodeError:
        raise ValueError("Filters must be valid JSON")

    cursor = conn.cursor()
    try:
        conditions = []
        params: List[Any] = []

        for column, value in filter_dict.items():
            validate_column(column)

            if isinstance(value, list):
                placeholders = ",".join(["?" for _ in value])
                conditions.append(f'"{column}" IN ({placeholders})')
                params.extend(value)
            elif isinstance(value, dict) and "operator" in value:
                op = value["operator"].upper()
                if op in [">", "<", ">=", "<=", "=", "!="]:
                    conditions.append(f'"{column}" {op} ?')
                    params.append(value["value"])
                elif op == "BETWEEN":
                    conditions.append(f'"{column}" BETWEEN ? AND ?')
                    params.extend([value["min"], value["max"]])
                elif op == "LIKE":
                    conditions.append(f'"{column}" LIKE ?')
                    params.append(f"%{value['value']}%")
                else:
                    raise ValueError(f"Unsupported operator: {op}")
            else:
                conditions.append(f'"{column}" = ?')
                params.append(value)

        base_query = f'SELECT * FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"'
        if conditions:
            where_clause = " AND ".join(conditions)
            query = f"{base_query} WHERE {where_clause}"
            count_query = f'SELECT COUNT(*) FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}" WHERE {where_clause}'
        else:
            query = base_query
            count_query = f'SELECT COUNT(*) FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"'

        if sort_by:
            validate_column(sort_by)
            sort_order = sort_order.upper()
            if sort_order not in ["ASC", "DESC"]:
                sort_order = "ASC"
            query += f' ORDER BY "{sort_by}" {sort_order}'

        cursor.execute(count_query, params)
        total_matches = cursor.fetchone()[0]

        query += f" LIMIT {limit}"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]

    except dbapi.Error as e:
        raise ValueError(str(e))
    finally:
        cursor.close()

    results = [dict(zip(col_names, row)) for row in rows]

    findings = [
        f"Filter conditions matched {total_matches:,} total records",
        f"Displaying {len(results)} records" + (f" (limited from {total_matches})" if total_matches > limit else ""),
    ]

    if total_matches == 0:
        findings.append("No records match the specified criteria")
        recommendations = [
            "Try relaxing some filter conditions",
            "Use get_distinct_values to see available values",
            "Check for typos in filter values",
        ]
    else:
        filter_count = len(filter_dict)
        findings.append(f"Applied {filter_count} filter condition{'s' if filter_count > 1 else ''}")
        recommendations = [
            "Use export_query_results to save filtered data",
            "Use get_column_stats on filtered results for analysis",
        ]
        if total_matches > limit:
            recommendations.append(f"Consider adding more filters or increasing limit (showing {limit} of {total_matches})")

    if sort_by:
        findings.append(f"Results sorted by '{sort_by}' in {sort_order.lower()}ending order")

    insight = QueryInsight(
        summary=f"Filter operation returned {total_matches:,} matching records",
        key_findings=findings,
        data_quality_notes=[],
        recommendations=recommendations,
        raw_data={
            "filters_applied": filter_dict,
            "total_matches": total_matches,
            "displayed_count": len(results),
            "sort_by": sort_by,
            "sort_order": sort_order,
            "results": results,
        },
    )

    return format_response(insight)

@mcp.tool()
def filter_by_date_range(
    date_column: str,
    start_date: str,
    end_date: str,
    date_format: str = "YYYY-MM-DD",
    limit: int = 100
) -> Dict:
    """Date-based filtering with intelligent analysis"""
    date_format_py = date_format.replace("YYYY", "%Y").replace("MM", "%m").replace("DD", "%d")

    try:
        parsed_start = datetime.strptime(start_date, date_format_py).date()
        parsed_end = datetime.strptime(end_date, date_format_py).date()
    except ValueError:
        raise ValueError("Invalid date or format. Check 'date_format' and try again.")

    validate_column(date_column)

    cursor = conn.cursor()
    try:
        count_query = f'''
            SELECT COUNT(*) FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"
            WHERE "{date_column}" BETWEEN ? AND ?
        '''
        cursor.execute(count_query, (parsed_start.isoformat(), parsed_end.isoformat()))
        total_matches = cursor.fetchone()[0]

        query = f'''
            SELECT * FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"
            WHERE "{date_column}" BETWEEN ? AND ?
            ORDER BY "{date_column}" ASC
            LIMIT {limit}
        '''
        cursor.execute(query, (parsed_start.isoformat(), parsed_end.isoformat()))
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
    except dbapi.Error as e:
        raise ValueError(str(e))
    finally:
        cursor.close()

    results = [dict(zip(col_names, row)) for row in rows]
    date_range_days = (parsed_end - parsed_start).days

    findings = [
        f"Date range: {parsed_start} to {parsed_end} ({date_range_days} days)",
        f"Found {total_matches:,} records in this date range",
        f"Displaying {len(results)} records" + (f" (limited from {total_matches})" if total_matches > limit else ""),
    ]

    recommendations = []
    if total_matches == 0:
        findings.append("No records found in the specified date range")
        recommendations.extend([
            "Try expanding the date range",
            "Use get_column_stats to understand the date distribution",
            "Verify date format and column values",
        ])
    else:
        records_per_day = total_matches / max(date_range_days, 1)
        findings.append(
            f"{'Sparse' if records_per_day < 1 else 'Dense'} data: ~{records_per_day:.2f} records/day"
        )
        recommendations.extend([
            "Use aggregate_data to analyze patterns by date periods",
            "Consider export_query_results for time series analysis",
        ])
        if total_matches > limit:
            recommendations.append("Large result set - consider narrowing date range or increasing limit")

    if results:
        first_date = results[0].get(date_column)
        last_date = results[-1].get(date_column)
        if first_date and last_date:
            findings.append(f"Data spans from {first_date} to {last_date}")

    insight = QueryInsight(
        summary=f"Date range filter on '{date_column}' returned {total_matches:,} records",
        key_findings=findings,
        data_quality_notes=[],
        recommendations=recommendations,
        raw_data={
            "date_column": date_column,
            "start_date": parsed_start.isoformat(),
            "end_date": parsed_end.isoformat(),
            "date_range_days": date_range_days,
            "total_matches": total_matches,
            "displayed_count": len(results),
            "results": results,
        },
    )

    return format_response(insight)

@mcp.tool()
def aggregate_data(
    group_by: str,
    aggregations: str,  # JSON string of aggregations
    having_filter: Optional[str] = None,
    limit: int = 100,
) -> Dict:
    """Perform aggregations with intelligent insights"""
    try:
        agg_dict = json.loads(aggregations)
    except json.JSONDecodeError:
        raise ValueError("Aggregations must be valid JSON")

    cursor = conn.cursor()
    try:
        validate_column(group_by)
        valid_functions = ["COUNT", "SUM", "AVG", "MIN", "MAX"]

        select_parts = [f'"{group_by}"']
        for column, function in agg_dict.items():
            validate_column(column)
            func = function.upper()
            if func not in valid_functions:
                raise ValueError(f"Invalid function: '{function}'. Use: {valid_functions}")
            select_parts.append(f'{func}("{column}") as {func.lower()}_{column}')

        query = f'''
            SELECT {", ".join(select_parts)}
            FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"
            GROUP BY "{group_by}"
        '''
        if having_filter:
            query += f" HAVING {having_filter}"
        query += f' ORDER BY "{group_by}" LIMIT {limit}'

        cursor.execute(query)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
    except dbapi.Error as e:
        raise ValueError(str(e))
    finally:
        cursor.close()

    results = [dict(zip(col_names, row)) for row in rows]

    findings = [f"Aggregated results grouped by '{group_by}'"]
    if results:
        findings.append(f"Returned {len(results)} groups")
        if len(results) == 1 and any(k.startswith("count_") for k in results[0].keys()):
            count_val = list(results[0].values())[1]
            findings.append(f"Total matching records: {count_val:,}")
    else:
        findings.append("No matching records found")

    insight = QueryInsight(
        summary=f"Aggregation on '{group_by}' returned {len(results)} groups",
        key_findings=findings,
        data_quality_notes=[],
        recommendations=["Consider exporting results for further analysis"],
        raw_data={
            "group_by": group_by,
            "aggregations": agg_dict,
            "having_filter": having_filter,
            "results": results,
        },
    )

    return format_response(insight)

@mcp.tool()
def get_distinct_values(
    column: str,
    limit: int = 50,
    search_filter: Optional[str] = None,
) -> Dict:
    """Analyze unique values with intelligent insights"""
    validate_column(column)

    cursor = conn.cursor()
    try:
        base_query = f'SELECT DISTINCT "{column}", COUNT(*) as frequency FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"'

        if search_filter:
            query = f'{base_query} WHERE UPPER("{column}") LIKE UPPER(?) GROUP BY "{column}" ORDER BY frequency DESC, "{column}" LIMIT {limit}'
            cursor.execute(query, (f"%{search_filter}%",))
        else:
            query = f'{base_query} GROUP BY "{column}" ORDER BY frequency DESC, "{column}" LIMIT {limit}'
            cursor.execute(query)

        results = cursor.fetchall()
        values_with_freq = [{"value": row[0], "frequency": row[1]} for row in results if row[0] is not None]

        count_query = f'SELECT COUNT(DISTINCT "{column}") FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}" WHERE "{column}" IS NOT NULL'
        if search_filter:
            count_query += f' AND UPPER("{column}") LIKE UPPER(?)'
            cursor.execute(count_query, (f"%{search_filter}%",))
        else:
            cursor.execute(count_query)
        total_distinct = cursor.fetchone()[0]
    except dbapi.Error as e:
        raise ValueError(str(e))
    finally:
        cursor.close()

    findings = [
        f"Column '{column}' has {total_distinct} distinct values" + (f" matching '{search_filter}'" if search_filter else ""),
        f"Showing top {len(values_with_freq)} values by frequency",
    ]

    if values_with_freq:
        most_common = values_with_freq[0]
        findings.append(f"Most frequent value: '{most_common['value']}' ({most_common['frequency']} occurrences)")
        total_freq = sum(v["frequency"] for v in values_with_freq)
        top_freq = most_common["frequency"]
        if total_freq and top_freq / total_freq > 0.5:
            findings.append(f"Distribution is skewed - top value represents {top_freq/total_freq:.1%} of data")
        else:
            findings.append("Distribution appears relatively balanced")

    recommendations = ["Use filter_data to analyze specific values", "Use these values in WHERE clauses for targeted queries"]
    if total_distinct > limit:
        recommendations.append(f"Only showing top {limit} of {total_distinct} values - use search_filter to narrow down")

    quality_notes = []
    if len(values_with_freq) == 1 and total_distinct == 1:
        quality_notes.append("Column contains only a single value - limited analytical value")
    elif total_distinct < 10:
        quality_notes.append("Low cardinality column - suitable for categorical analysis")
    elif total_distinct > 1000:
        quality_notes.append("High cardinality column - consider grouping or binning for analysis")

    insight = QueryInsight(
        summary=f"Distinct value analysis for column '{column}'",
        key_findings=findings,
        data_quality_notes=quality_notes,
        recommendations=recommendations,
        raw_data={
            "column": column,
            "total_distinct_count": total_distinct,
            "displayed_count": len(values_with_freq),
            "values": values_with_freq,
            "search_filter": search_filter,
        },
    )

    return format_response(insight)

@mcp.tool()
def get_data_types() -> Dict:
    """Comprehensive column metadata with recommendations"""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT 
                COLUMN_NAME,
                DATA_TYPE_NAME,
                LENGTH,
                SCALE,
                IS_NULLABLE,
                DEFAULT_VALUE,
                COMMENTS
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?
            ORDER BY POSITION
            """,
            (HANA_SCHEMA, HANA_TABLE_NAME),
        )

        columns_info = []
        type_distribution: Dict[str, int] = {}
        nullable_count = 0
        has_defaults = 0

        for row in cursor.fetchall():
            col_info = {
                "name": row[0],
                "data_type": row[1],
                "length": row[2],
                "scale": row[3],
                "nullable": row[4] == "YES",
                "default_value": row[5],
                "comments": row[6],
            }
            columns_info.append(col_info)
            type_distribution[row[1]] = type_distribution.get(row[1], 0) + 1
            if row[4] == "YES":
                nullable_count += 1
            if row[5] is not None:
                has_defaults += 1
    finally:
        cursor.close()

    total_columns = len(columns_info)
    findings = [
        f"Table has {total_columns} columns with {len(type_distribution)} different data types",
        f"Data type distribution: {dict(type_distribution)}",
        f"{nullable_count} columns allow NULL values ({nullable_count/total_columns:.1%})" if total_columns else "No columns detected",
        f"{has_defaults} columns have default values",
    ]

    text_cols = [col for col in columns_info if col["data_type"] in ["VARCHAR", "NVARCHAR", "TEXT"]]
    numeric_cols = [col for col in columns_info if col["data_type"] in ["INTEGER", "DECIMAL", "DOUBLE"]]
    date_cols = [col for col in columns_info if col["data_type"] in ["DATE", "TIMESTAMP"]]

    recommendations = []
    if text_cols:
        recommendations.append(f"Text analysis: Use search_data on {[c['name'] for c in text_cols[:3]]}")
    if numeric_cols:
        recommendations.append(f"Statistical analysis: Use get_column_stats on {[c['name'] for c in numeric_cols[:3]]}")
    if date_cols:
        recommendations.append(f"Time analysis: Use filter_by_date_range on {[c['name'] for c in date_cols[:3]]}")

    quality_notes = []
    if nullable_count > total_columns * 0.8 if total_columns else False:
        quality_notes.append("High percentage of nullable columns - expect missing data")
    if len([col for col in columns_info if col["data_type"] == "VARCHAR" and (col["length"] or 0) > 1000]) > 0:
        quality_notes.append("Large text columns detected - may contain unstructured data")

    insight = QueryInsight(
        summary=f"Data type analysis: {total_columns} columns across {len(type_distribution)} data types",
        key_findings=findings,
        data_quality_notes=quality_notes,
        recommendations=recommendations,
        raw_data={
            "table": HANA_TABLE_NAME,
            "columns": columns_info,
            "summary_stats": {
                "total_columns": total_columns,
                "type_distribution": type_distribution,
                "nullable_count": nullable_count,
                "defaults_count": has_defaults,
                "text_columns": [c["name"] for c in text_cols],
                "numeric_columns": [c["name"] for c in numeric_cols],
                "date_columns": [c["name"] for c in date_cols],
            },
        },
    )

    return format_response(insight)

@mcp.tool()
def get_column_stats(column: str) -> Dict:
    """Get statistical analysis for a specific column"""
    validate_column(column)

    cursor = conn.cursor()
    try:
        # Basic stats for all column types
        cursor.execute(f'''
            SELECT 
                COUNT("{column}") as count,
                COUNT(DISTINCT "{column}") as distinct_count
            FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"
            WHERE "{column}" IS NOT NULL
        ''')
        basic_stats = cursor.fetchone()
        stats = {
            "count": basic_stats[0],
            "distinct_count": basic_stats[1]
        }

        # Try numeric statistics
        try:
            cursor.execute(f'''
                SELECT 
                    MIN("{column}") as min_value,
                    MAX("{column}") as max_value,
                    AVG(CAST("{column}" AS DECIMAL)) as average
                FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"
                WHERE "{column}" IS NOT NULL
            ''')
            numeric_stats = cursor.fetchone()
            if numeric_stats[0] is not None:
                stats.update({
                    "min_value": numeric_stats[0],
                    "max_value": numeric_stats[1],
                    "average": numeric_stats[2]
                })
        except dbapi.Error:
            # Not a numeric column
            pass

        # Get top values
        cursor.execute(f'''
            SELECT "{column}", COUNT(*) as frequency
            FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"
            WHERE "{column}" IS NOT NULL
            GROUP BY "{column}"
            ORDER BY frequency DESC
            LIMIT 10
        ''')
        top_values = [{"value": row[0], "frequency": row[1]} for row in cursor.fetchall()]
        stats["top_values"] = top_values

    except dbapi.Error as e:
        raise ValueError(str(e))
    finally:
        cursor.close()

    # Analyze the statistics
    findings = []
    quality_notes = []
    recommendations = []

    count = stats.get("count", 0)
    distinct_count = stats.get("distinct_count", 0)

    findings.append(f"Column '{column}' contains {count:,} non-null values")

    if distinct_count:
        uniqueness_ratio = distinct_count / count if count else 0
        findings.append(f"Uniqueness: {distinct_count:,} distinct values ({uniqueness_ratio:.1%})")
        if uniqueness_ratio > 0.95:
            findings.append("High uniqueness suggests this might be an identifier column")
        elif uniqueness_ratio < 0.1:
            findings.append("Low uniqueness indicates many repeated values")

    if stats.get("min_value") is not None:
        min_val = stats.get("min_value")
        max_val = stats.get("max_value")
        avg_val = stats.get("average")

        findings.append(f"Range: {min_val} to {max_val}")
        if avg_val is not None:
            findings.append(f"Average: {avg_val}")

        if min_val < 0 and column.lower() in ["amount", "price", "quantity", "count"]:
            quality_notes.append("Negative values detected in what appears to be a positive measure")

        if max_val is not None and min_val is not None and max_val > (min_val * 1000 if min_val != 0 else max_val):
            quality_notes.append("Large value range detected - check for outliers")

    if "top_values" in stats:
        top_values = stats["top_values"][:5]
        top_values_summary = [f"{v['value']} ({v['frequency']}x)" for v in top_values]
        findings.append(f"Top values: {top_values_summary}")
        if top_values and top_values[0]["frequency"] > count * 0.5:
            quality_notes.append(f"Single value dominates dataset: '{top_values[0]['value']}'")

    recommendations.append("Use filter_data to analyze specific value ranges")
    recommendations.append("Consider get_distinct_values for categorical analysis")

    insight = QueryInsight(
        summary=f"Statistical analysis of column '{column}' with {count:,} values",
        key_findings=findings,
        data_quality_notes=quality_notes,
        recommendations=recommendations,
        raw_data=stats,
    )

    return format_response(insight)

@mcp.tool()
def export_query_results(
    query: Optional[str] = None,
    format: str = "csv",
    include_headers: bool = True,
) -> Dict:
    """Export with format-specific insights"""
    # Use default query if none provided
    if not query:
        query = f'SELECT * FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"'
    
    if not query.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    cursor = conn.cursor()
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]

        # Create exports directory if it doesn't exist
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{HANA_TABLE_NAME}_{timestamp}"

        if format.lower() == "csv":
            filepath = os.path.join(export_dir, f"{filename}.csv")
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if include_headers:
                    writer.writerow(col_names)
                for row in rows:
                    writer.writerow(row)
            export_data = f"Data exported to {filepath}"

        elif format.lower() == "json":
            filepath = os.path.join(export_dir, f"{filename}.json")
            export_data = [dict(zip(col_names, row)) for row in rows]
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            export_data = f"Data exported to {filepath}"

        else:  # summary format
            export_data = {
                "query": query,
                "columns": col_names,
                "row_count": len(rows),
                "sample_rows": [dict(zip(col_names, row)) for row in rows[:5]]
            }

        insight = QueryInsight(
            summary=f"Export completed: {len(rows):,} records in {format.upper()} format",
            key_findings=[
                f"✅ Successfully exported {len(rows):,} records",
                f"Export format: {format.upper()}",
                f"✅ File saved as: {filepath if format.lower() in ['csv', 'json'] else 'N/A'}",
                f"Columns included: {len(col_names)}"
            ],
            data_quality_notes=[],
            recommendations=[
                f"Data exported in {format} format",
                "Check the 'exports' folder for your file"
            ],
            raw_data=export_data
        )

        return format_response(insight)

    except dbapi.Error as e:
        raise ValueError(str(e))
    finally:
        cursor.close()

# ---------------------------------------------------------------------
# Run MCP Server
# ---------------------------------------------------------------------
if __name__ == "__main__":
    load_valid_tables()
    load_valid_columns()
    mcp.run(transport="stdio")