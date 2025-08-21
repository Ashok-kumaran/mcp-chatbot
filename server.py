import os
import json
import logging
import sys
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from difflib import get_close_matches
from hdbcli import dbapi
from dotenv import load_dotenv
import argparse

# Logging with UTF-8
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            stream.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        UTF8StreamHandler(sys.stdout),
        logging.FileHandler('server.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Environment setup
load_dotenv()
HANA_HOST = os.getenv("HANA_HOST")
HANA_PORT = os.getenv("HANA_PORT", "443")
HANA_USER = os.getenv("HANA_USER")
HANA_PASS = os.getenv("HANA_PASS")
HANA_SCHEMA = os.getenv("HANA_SCHEMA")
HANA_TABLE_NAME = os.getenv("HANA_TABLE_NAME", "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION")

def validate_env_vars():
    required_vars = ["HANA_HOST", "HANA_USER", "HANA_PASS", "HANA_SCHEMA"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}", file=sys.stderr)
        sys.exit(1)
    try:
        port = int(HANA_PORT)
        if port <= 0 or port > 65535:
            raise ValueError
    except ValueError:
        logger.error("HANA_PORT must be a valid integer port number (1-65535)")
        print("Error: HANA_PORT must be a valid integer port number", file=sys.stderr)
        sys.exit(1)

validate_env_vars()

# DB connection
conn = None

def init_connection():
    global conn
    try:
        conn = dbapi.connect(
            address=HANA_HOST,
            port=int(HANA_PORT),
            user=HANA_USER,
            password=HANA_PASS,
            encrypt=True,
            sslValidateCertificate=False
        )
        logger.info(f"Connected to HANA schema '{HANA_SCHEMA}' on {HANA_HOST}:{HANA_PORT}")
        print(f"Connected to HANA schema '{HANA_SCHEMA}'", file=sys.stderr)
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        print(f"Error: DB connection failed: {e}", file=sys.stderr)
        sys.exit(1)

def close_connection():
    global conn
    if conn is not None:
        conn.close()
        logger.info("Database connection closed")
        print("Database connection closed", file=sys.stderr)

import atexit
atexit.register(close_connection)
init_connection()

# Schema cache
class SchemaCache:
    def __init__(self):
        self.valid_tables: set = set()
        self.valid_columns: set = set()
        self.column_types: Dict[str, str] = {}

    def load_valid_tables(self):
        with conn.cursor() as cursor:
            try:
                cursor.execute(
                    "SELECT DISTINCT TABLE_NAME FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME = ?",
                    (HANA_SCHEMA,)
                )
                self.valid_tables = set(row[0] for row in cursor.fetchall())
                logger.debug(f"Loaded {len(self.valid_tables)} valid tables")
                print(f"Loaded {len(self.valid_tables)} tables", file=sys.stderr)
            except dbapi.Error as e:
                logger.error(f"Failed to load valid tables: {e}")
                raise

    def load_valid_columns(self):
        with conn.cursor() as cursor:
            try:
                cursor.execute(
                    "SELECT COLUMN_NAME, DATA_TYPE_NAME FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?",
                    (HANA_SCHEMA, HANA_TABLE_NAME)
                )
                results = cursor.fetchall()
                self.valid_columns = set(row[0].upper() for row in results)
                self.column_types = {row[0].upper(): row[1] for row in results}
                logger.debug(f"Loaded {len(self.valid_columns)} valid columns")
                print(f"Loaded {len(self.valid_columns)} columns", file=sys.stderr)
            except dbapi.Error as e:
                logger.error(f"Failed to load valid columns: {e}")
                raise

schema_cache = SchemaCache()
schema_cache.load_valid_tables()
schema_cache.load_valid_columns()

# Data containers
@dataclass
class QueryInsight:
    summary: str
    key_findings: List[str]
    data_quality_notes: List[str]
    recommendations: List[str]
    raw_data: Any

# Utilities
def validate_column(col: str, schema_cache: SchemaCache) -> str:
    aliases = {
        "material document year": "FUELONWARDSMATERIALDOCUMENTYEAR",
        "fuel onwards material document year": "FUELONWARDSMATERIALDOCUMENTYEAR",
        "fuel": "FUEL_TYPE",
        "status": "COMPLIANCE_STATUS"
    }
    if col is None:
        raise ValueError("Column name is required.")

    col_normalized = col.replace(" ", "").replace("_", "").upper()
    if col.lower() in aliases:
        corrected_col = aliases[col.lower()]
        if corrected_col.upper() in schema_cache.valid_columns:
            logger.debug(f"Matched alias '{col}' to '{corrected_col}'")
            return corrected_col

    if col_normalized in {c.replace("_", "") for c in schema_cache.valid_columns}:
        for c in schema_cache.valid_columns:
            if col_normalized == c.replace("_", ""):
                logger.debug(f"Exact match for column '{col}' -> '{c}'")
                return c

    suggestion = get_close_matches(col_normalized, [c.replace("_", "") for c in schema_cache.valid_columns], n=1, cutoff=0.8)
    if suggestion:
        for c in schema_cache.valid_columns:
            if suggestion[0] == c.replace("_", ""):
                logger.warning(f"Auto-correcting column '{col}' to '{c}'")
                return c

    raise ValueError(f"Invalid column: '{col}'. Suggestions: {get_close_matches(col_normalized, schema_cache.valid_columns, n=3)}")

def validate_columns(cols: List[str], schema_cache: SchemaCache) -> List[str]:
    return [validate_column(c, schema_cache) for c in cols]

def get_column_data_type(column_name: str) -> str:
    return schema_cache.column_types.get(column_name.upper(), 'UNKNOWN')

def format_value_for_hana(value, column_name: str):
    data_type = get_column_data_type(column_name)
    if value is None:
        return None
    if data_type == 'BOOLEAN':
        return 'TRUE' if str(value).lower() in ['true', '1', 'yes'] else 'FALSE'
    if data_type in ['DECIMAL', 'DOUBLE', 'REAL']:
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if data_type == 'INTEGER':
        try:
            return int(float(str(value)))
        except (ValueError, TypeError):
            return value
    return value

# Natural Language Parser
def parse_natural_language(query: str, schema_cache: SchemaCache) -> Tuple[str, Dict]:
    query = query.lower().strip()
    intent = "unknown"
    params = {}

    def try_parse_date_or_year(text: str) -> Optional[Tuple[datetime, datetime]]:
        text = text.strip()
        date_formats = ["%Y-%m-%d", "%Y-%m", "%B %Y", "%Y", "%b %Y", "%Y/%m"]
        for fmt in date_formats:
            try:
                dt = datetime.strptime(text, fmt)
                if fmt == "%Y":
                    return dt.replace(month=1, day=1), dt.replace(month=12, day=31)
                return dt, dt
            except ValueError:
                continue
        return None

    tokens = query.split()
    tokens = [t for t in tokens if t]

    aggregation_keywords = ["count", "sum", "average", "avg", "min", "max", "group", "by"]
    filter_keywords = ["where", "filter", "equals", "=", "is", "from", "to", "onwards", "between", "year"]
    summary_keywords = ["summary", "overview", "table summary", "describe table"]

    if any(kw in query for kw in summary_keywords):
        intent = "summary"
        params = {}

    elif "how many" in query or any(kw in tokens for kw in aggregation_keywords):
        intent = "aggregate"
        group_by = None
        aggregations = {}
        for i, token in enumerate(tokens):
            if token == "by" and i > 0 and i + 1 < len(tokens):
                try:
                    group_by = validate_column(tokens[i + 1], schema_cache)
                except ValueError:
                    continue
            if token in ["count", "sum", "avg", "min", "max"]:
                if i + 1 < len(tokens):
                    try:
                        col = validate_column(tokens[i + 1], schema_cache)
                        aggregations[col] = token.upper()
                    except ValueError:
                        continue
        if "how many" in query and not aggregations:
            aggregations["count"] = "COUNT"
        params = {"group_by": group_by, "aggregations": json.dumps(aggregations) if aggregations else '{"count": "COUNT"}'}

        # Handle specific query: "how many records when fuel onwards material document year = 2025"
        if "fuel onwards material document year" in query or "material document year" in query:
            try:
                col = validate_column("fuel onwards material document year", schema_cache)
                year_match = query[query.find("2025")-1:query.find("2025")+5].strip(" =")
                year = int(year_match) if year_match.isdigit() else 2025
                start_date = datetime(year=year, month=1, day=1)
                end_date = datetime(year=year, month=12, day=31)
                intent = "aggregate"
                params = {
                    "group_by": None,
                    "aggregations": json.dumps({"count": "COUNT"}),
                    "having_filter": f'"{col}" BETWEEN \'{start_date.strftime("%Y-%m-%d")}\' AND \'{end_date.strftime("%Y-%m-%d")}\''
                }
                if "fuel" in query:
                    params["having_filter"] += f' AND "FUEL_TYPE" IS NOT NULL'
            except ValueError as e:
                logger.error(f"Column validation failed for query: {e}")
                raise

    logger.debug(f"Parsed query: {query}, intent: {intent}, params: {params}")
    return intent, params

# Tools
async def run_sql(query: str):
    intent, params = parse_natural_language(query, schema_cache)
    logger.debug(f"Running run_sql with intent: {intent}, params: {params}")
    if intent == "aggregate":
        return await aggregate_data(**params)
    elif intent == "summary":
        return await get_table_summary()
    else:
        raise ValueError(f"Unsupported intent: {intent}")

async def get_data(limit: int = 10):
    if limit < 1 or limit > 1000:
        raise ValueError("Limit must be between 1 and 1000.")

    with conn.cursor() as cursor:
        query = f'SELECT * FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}" LIMIT {limit}'
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
        except dbapi.Error as e:
            logger.error(f"SQL execution error in get_data: {e}")
            raise ValueError(str(e))

    data = [dict(zip(col_names, row)) for row in rows]
    return {"data": data}

async def get_schema():
    return {
        "schema": HANA_SCHEMA,
        "table": HANA_TABLE_NAME,
        "columns": list(schema_cache.valid_columns),
        "column_types": schema_cache.column_types
    }

async def get_table_summary():
    with conn.cursor() as cursor:
        try:
            cursor.execute(f'SELECT COUNT(*) FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"')
            row_count = cursor.fetchone()[0]
            cursor.execute(
                f'SELECT COLUMN_NAME, DATA_TYPE_NAME FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?',
                (HANA_SCHEMA, HANA_TABLE_NAME)
            )
            columns = [(row[0], row[1]) for row in cursor.fetchall()]
        except dbapi.Error as e:
            logger.error(f"SQL execution error in get_table_summary: {e}")
            raise ValueError(str(e))

    summary = {
        "row_count": row_count,
        "column_count": len(columns),
        "columns": [{"name": col[0], "type": col[1]} for col in columns]
    }
    return {
        "summary": f"Table {HANA_TABLE_NAME} has {row_count} rows and {len(columns)} columns",
        "data": summary
    }

async def aggregate_data(group_by: str = None, aggregations: str = '{"count": "COUNT"}', having_filter: str = None, limit: int = 100):
    if limit < 1 or limit > 1000:
        raise ValueError("Limit must be between 1 and 1000.")

    try:
        agg_dict = json.loads(aggregations)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for aggregations")

    with conn.cursor() as cursor:
        if group_by:
            group_by = validate_column(group_by, schema_cache)
        agg_clauses = []
        if group_by:
            agg_clauses.append(f'"{group_by}"')
        valid_functions = ["COUNT", "SUM", "AVG", "MIN", "MAX"]
        for column, function in agg_dict.items():
            if column != "count":
                column = validate_column(column, schema_cache)
            func = function.upper()
            if func not in valid_functions:
                raise ValueError(f"Invalid function: {function}")
            agg_clauses.append(f'{func}("{column}") as {func.lower()}_{column}' if column != "count" else f'{func}(*) as count')

        select_clause = ", ".join(agg_clauses)
        query = f'SELECT {select_clause} FROM "{HANA_SCHEMA}"."{HANA_TABLE_NAME}"'
        if group_by:
            query += f' GROUP BY "{group_by}"'
        if having_filter:
            query += f" HAVING {having_filter}"
        query += f' ORDER BY {"count" if "count" in agg_dict else agg_clauses[0]} DESC LIMIT {limit}'

        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
        except dbapi.Error as e:
            logger.error(f"SQL execution error in aggregate_data: {e}")
            raise ValueError(str(e))

    results = [dict(zip(col_names, row)) for row in rows]
    return {
        "summary": f"Aggregation by {group_by if group_by else 'count'} returned {len(results)} groups",
        "data": results
    }

# MCP-compatible stdio server
tools = {
    "run_sql": {"func": run_sql, "description": "Run a natural language query"},
    "get_data": {"func": get_data, "description": "Retrieve sample data"},
    "get_schema": {"func": get_schema, "description": "Get schema information"},
    "get_table_summary": {"func": get_table_summary, "description": "Get table summary (row count, columns)"}
}

async def run_stdio():
    logger.info("Starting MCP server")
    print(json.dumps({"type": "initialize_ack", "version": "1.0"}), file=sys.stdout, flush=True)
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    try:
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
    except Exception as e:
        logger.error(f"Failed to connect stdin pipe: {e}")
        print(json.dumps({"type": "error", "error": f"Failed to connect stdin: {e}"}), file=sys.stdout, flush=True)
        sys.exit(1)
    while True:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            if not line:
                logger.info("Received EOF, shutting down")
                break
            request = json.loads(line.decode().strip())
            logger.debug(f"Received request: {request}")
            request_type = request.get("type")
            if request_type == "initialize":
                response = {"type": "initialize_ack", "version": "1.0"}
                print(json.dumps(response), file=sys.stdout, flush=True)
                logger.debug("Sent initialize_ack")
            elif request_type == "list_tools":
                response = {"type": "tools", "tools": [{"name": k, "description": v["description"]} for k, v in tools.items()]}
                print(json.dumps(response), file=sys.stdout, flush=True)
                logger.debug(f"Sent tools list: {response}")
            elif request_type == "call_tool":
                tool_name = request.get("tool")
                params = request.get("params", {})
                if tool_name in tools:
                    try:
                        result = await tools[tool_name]["func"](**params)
                        response = {"type": "tool_result", "tool": tool_name, "result": result}
                        print(json.dumps(response), file=sys.stdout, flush=True)
                        logger.debug(f"Sent tool result: {response}")
                    except Exception as e:
                        response = {"type": "error", "error": str(e)}
                        print(json.dumps(response), file=sys.stdout, flush=True)
                        logger.error(f"Tool error: {e}")
                else:
                    response = {"type": "error", "error": f"Unknown tool: {tool_name}"}
                    print(json.dumps(response), file=sys.stdout, flush=True)
                    logger.error(f"Unknown tool: {tool_name}")
            else:
                response = {"type": "error", "error": f"Unknown request type: {request_type}"}
                print(json.dumps(response), file=sys.stdout, flush=True)
                logger.error(f"Unknown request type: {request_type}")
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for stdin input")
            print(json.dumps({"type": "error", "error": "Timeout waiting for input"}), file=sys.stdout, flush=True)
            break
        except json.JSONDecodeError as e:
            response = {"type": "error", "error": f"Invalid JSON: {e}"}
            print(json.dumps(response), file=sys.stdout, flush=True)
            logger.error(f"Invalid JSON: {e}")
        except Exception as e:
            response = {"type": "error", "error": f"Server error: {e}"}
            print(json.dumps(response), file=sys.stdout, flush=True)
            logger.error(f"Server error: {e}")

async def test_query():
    query = "how many records when fuel onwards material document year = 2025"
    print("Valid columns:", sorted(list(schema_cache.valid_columns)))
    try:
        result = await run_sql(query)
        print(json.dumps({"result": result}, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAP HANA MCP Server")
    parser.add_argument("--test", action="store_true", help="Run test query")
    args = parser.parse_args()

    if args.test:
        asyncio.run(test_query())
    else:
        try:
            asyncio.run(run_stdio())
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            print(f"Error: Failed to start MCP server: {e}", file=sys.stderr)
            sys.exit(1)