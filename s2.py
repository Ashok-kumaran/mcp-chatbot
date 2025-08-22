# # from typing import Dict, Any
# # from mcp.server.fastmcp import FastMCP
# # from fastapi import Query, HTTPException, Body
# # from hdbcli import dbapi
# # from dotenv import load_dotenv
# # from fastapi import File, UploadFile
# # import requests

# # import os
# # import random
# # import uuid
# # from datetime import datetime, timedelta
# # import logging

# # # Set up logging
# # logging.basicConfig(level=logging.DEBUG)
# # logger = logging.getLogger(__name__)

# # # Load environment variables
# # load_dotenv()

# # # Initialize FastMCP server
# # mcp = FastMCP("S4HANA_DB")

# # HANA_HOST = os.getenv("HANA_HOST")
# # HANA_PORT = int(os.getenv("HANA_PORT", "443"))
# # HANA_USER = os.getenv("HANA_USER")
# # HANA_PASS = os.getenv("HANA_PASS")
# # HANA_SCHEMA = os.getenv("HANA_SCHEMA")

# # # Connect to SAP HANA Cloud
# # conn = dbapi.connect(
# #     address=HANA_HOST,
# #     port=HANA_PORT,
# #     user=HANA_USER,
# #     password=HANA_PASS,
# #     encrypt=True,
# #     sslValidateCertificate=False
# # )

# # def get_table_schema(table_name):
# #     """Get the actual schema for a specific table with column lengths"""
# #     cursor = conn.cursor()
# #     try:
# #         cursor.execute(f"""
# #             SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE, LENGTH, DEFAULT_VALUE
# #             FROM SYS.TABLE_COLUMNS
# #             WHERE SCHEMA_NAME = '{HANA_SCHEMA}' AND TABLE_NAME = '{table_name}'
# #             ORDER BY POSITION
# #         """)
# #         results = cursor.fetchall()
# #         logger.debug(f"Schema for {table_name}: {results}")
# #         return [(col_name, data_type, is_nullable, length, default_val) for col_name, data_type, is_nullable, length, default_val in results]
# #     except Exception as e:
# #         logger.error(f"Error getting schema for {table_name}: {e}")
# #         return []
# #     finally:
# #         cursor.close()

# # def generate_random_value_for_column(column_name, data_type, is_nullable, max_length, default_value=None):
# #     """Enhanced random value generation with better error handling and logging"""
    
# #     try:
# #         logger.debug(f"Generating value for column: {column_name}, type: {data_type}, nullable: {is_nullable}, max_length: {max_length}")
        
# #         # Handle nullable columns (10% chance of NULL for nullable columns)
# #         if is_nullable == 'YES' and random.random() < 0.1:
# #             return None
        
# #         # Use default value if available for certain cases
# #         if default_value and random.random() < 0.3:  # 30% chance to use default
# #             return default_value
        
# #         column_lower = column_name.lower()
        
# #         # Determine safe max length for string values
# #         safe_length = min(max_length or 50, 100) if max_length else 50
        
# #         # String/Text types
# #         if data_type.upper() in ['NVARCHAR', 'VARCHAR', 'NCLOB', 'CLOB', 'TEXT', 'STRING']:
# #             return generate_string_value(column_lower, safe_length)
        
# #         # Numeric types
# #         elif data_type.upper() in ['INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT']:
# #             return generate_integer_value(column_lower)
        
# #         # Decimal/Float types
# #         elif data_type.upper() in ['DECIMAL', 'DOUBLE', 'REAL', 'FLOAT']:
# #             return generate_decimal_value(column_lower)
        
# #         # Date types
# #         elif data_type.upper() in ['DATE']:
# #             return generate_date_value()
        
# #         # Timestamp types
# #         elif data_type.upper() in ['TIMESTAMP', 'SECONDDATE']:
# #             return generate_timestamp_value()
        
# #         # Boolean types
# #         elif data_type.upper() in ['BOOLEAN']:
# #             return random.choice([True, False])
        
# #         # Default fallback
# #         else:
# #             logger.warning(f"Unknown data type {data_type} for column {column_name}, using default string")
# #             base_default = f"Default_{random.randint(1, 999)}"
# #             return base_default[:safe_length]
            
# #     except Exception as e:
# #         logger.error(f"Error generating value for column {column_name}: {e}")
# #         # Return a safe default value based on the data type
# #         if data_type.upper() in ['NVARCHAR', 'VARCHAR', 'NCLOB', 'CLOB', 'TEXT', 'STRING']:
# #             return f"DEFAULT_{random.randint(1, 999)}"[:10]
# #         elif data_type.upper() in ['INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT']:
# #             return random.randint(1, 1000)
# #         elif data_type.upper() in ['DECIMAL', 'DOUBLE', 'REAL', 'FLOAT']:
# #             return round(random.uniform(1.0, 100.0), 2)
# #         else:
# #             return None

# # def generate_string_value(column_lower, safe_length):
# #     """Generate string values based on column name patterns"""

# #     if 'id' in column_lower and column_lower not in ['objectid', 'object_id', 'organization_id']:
# #         base_value = f"ID{random.randint(100, 999)}"
# #         return base_value[:safe_length]
# #     elif 'guid' in column_lower or 'uuid' in column_lower:
# #         guid_value = str(uuid.uuid4())
# #         return guid_value
# #     elif 'code' in column_lower:
# #         return generate_code_value(column_lower, safe_length)
# #     elif 'desc' in column_lower or 'description' in column_lower:
# #         descriptions = [
# #             "Standard compliance transaction",
# #             "Environmental credit transfer",
# #             "Renewable fuel standard",
# #             "Carbon offset transaction",
# #             "Biofuel credit exchange"
# #         ]
# #         selected = random.choice(descriptions)
# #         return selected[:safe_length]
# #     elif 'name' in column_lower:
# #         names = [
# #             "EcoCredit_Transaction",
# #             "RFS2_Compliance",
# #             "LCFS_Transfer",
# #             "Carbon_Credit",
# #             "Biofuel_Exchange"
# #         ]
# #         selected = random.choice(names)
# #         return selected[:safe_length]
# #     elif 'status' in column_lower:
# #         statuses = ['ACTIVE', 'PENDING', 'COMPLETED', 'CANCELLED', 'EXPIRED']
# #         selected = random.choice(statuses)
# #         return selected[:safe_length]
# #     elif 'type' in column_lower:
# #         types = ['PURCHASE', 'SALE', 'TRANSFER', 'GENERATION', 'RETIREMENT']
# #         selected = random.choice(types)
# #         return selected[:safe_length]
# #     elif 'category' in column_lower:
# #         categories = ['RFS2', 'LCFS', 'RGGI', 'CAP_TRADE', 'VOLUNTARY']
# #         selected = random.choice(categories)
# #         return selected[:safe_length]
# #     elif 'currency' in column_lower:
# #         currencies = ['USD']
# #         selected = random.choice(currencies)
# #         return selected[:safe_length]
# #     elif 'user' in column_lower:
# #         base_user = f"user_{random.randint(1, 99)}"
# #         return base_user[:safe_length]
# #     elif 'comment' in column_lower or 'note' in column_lower:
# #         comments = [
# #             "Standard processing completed",
# #             "Additional documentation required",
# #             "QA review in progress",
# #             "Expedited transaction request",
# #             "Compliance verification pending"
# #         ]
# #         selected = random.choice(comments)
# #         return selected[:safe_length]
# #     elif 'reference' in column_lower or 'ref' in column_lower:
# #         base_ref = f"{random.randint(100000, 999999)}"
# #         return base_ref[:safe_length]
# #     elif 'batch' in column_lower:
# #         base_batch = f"{random.randint(10000, 99999)}"
# #         return base_batch[:safe_length]
# #     elif 'external' in column_lower:
# #         base_ext = f"{random.randint(1000000, 9999999)}"
# #         return base_ext[:safe_length]
# #     elif 'facility' in column_lower:
# #         base_fac = f"{random.randint(1, 999):03d}"
# #         return base_fac[:safe_length]
# #     elif 'organization' in column_lower or 'org' in column_lower:
# #         orgs = [ 'Generator', 'ObligatedParty', 'VoluntaryMarket']
# #         selected = random.choice(orgs)
# #         return selected[:safe_length]
# #     else:
# #         # Generic string value
# #         if safe_length <= 5:
# #             return f"V{random.randint(1, 999)}"[:safe_length]
# #         elif safe_length <= 10:
# #             return f"Val_{random.randint(1, 999)}"[:safe_length]
# #         else:
# #             return f"Value_{random.randint(1, 999)}"[:safe_length]

# # def generate_code_value(column_lower, safe_length):
# #     """Generate code values based on specific column patterns"""
    
# #     if 'fuel' in column_lower:
# #         fuel_codes = ['D3', 'D4', 'D5', 'D6', 'A1', 'E85']
# #         selected = random.choice(fuel_codes)
# #         return selected[:safe_length]
# #     elif 'status' in column_lower:
# #         status_codes = ['1', '2', '5', '6', '7', '8', '9', '10', '11', '12', '13']
# #         selected = random.choice(status_codes)
# #         return selected[:safe_length]
# #     elif 'credit' in column_lower:
# #         if safe_length <= 3:
# #             return str(random.randint(1, 99))[:safe_length]
# #         else:
# #             return f"CR{random.randint(10, 99)}"[:safe_length]
# #     elif 'compliance' in column_lower:
# #         compliance_codes = ['RFS2', 'LCFS', 'RGGI', 'CAT']
# #         selected = random.choice(compliance_codes)
# #         return selected[:safe_length]
# #     else:
# #         if safe_length <= 5:
# #             return f"C{random.randint(10, 99)}"[:safe_length]
# #         else:
# #             return f"CODE{random.randint(10, 99)}"[:safe_length]

# # def generate_integer_value(column_lower):
# #     """Generate integer values based on column patterns"""
    
# #     if 'id' in column_lower:
# #         if 'object' in column_lower:
# #             return random.randint(6000, 9999)
# #         else:
# #             return random.randint(1000, 9999)
# #     elif 'year' in column_lower:
# #         return random.randint(2020, 2025)
# #     elif 'month' in column_lower:
# #         return random.randint(1, 12)
# #     elif 'quarter' in column_lower:
# #         return random.randint(1, 4)
# #     elif 'code' in column_lower:
# #         return random.randint(1, 100)
# #     elif 'status' in column_lower:
# #         return random.randint(1, 13)
# #     elif 'quantity' in column_lower or 'amount' in column_lower:
# #         return random.randint(100, 50000)
# #     elif 'price' in column_lower:
# #         return random.randint(1, 1000)
# #     else:
# #         return random.randint(1, 10000)

# # def generate_decimal_value(column_lower):
# #     """Generate decimal values based on column patterns"""
    
# #     if 'price' in column_lower:
# #         return round(random.uniform(0.50, 15.00), 2)
# #     elif 'quantity' in column_lower or 'amount' in column_lower:
# #         return round(random.uniform(100.0, 50000.0), 2)
# #     elif 'rate' in column_lower:
# #         return round(random.uniform(0.01, 1.0), 4)
# #     elif 'percentage' in column_lower or 'percent' in column_lower:
# #         return round(random.uniform(0.0, 100.0), 2)
# #     else:
# #         return round(random.uniform(1.0, 1000.0), 2)

# # def generate_date_value():
# #     """Generate a random date"""
# #     base_date = datetime(2025, 1, 1)
# #     random_days = random.randint(0, 730)  # 2 years range
# #     return (base_date + timedelta(days=random_days)).strftime('%Y-%m-%d')

# # def generate_timestamp_value():
# #     """Generate a random timestamp"""
# #     base_date = datetime(2025, 1, 1)
# #     random_days = random.randint(0, 730)
# #     random_hours = random.randint(0, 23)
# #     random_minutes = random.randint(0, 59)
# #     random_seconds = random.randint(0, 59)
# #     return (base_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes, seconds=random_seconds)).strftime('%Y-%m-%d %H:%M:%S')

# # @mcp.tool(name="generate_and_insert_random_entry", description="Generate and insert random entries into a specified SAP HANA table")
# # async def generate_and_insert_random_entry(
# #     table: str = Query(..., description="SAP HANA table name"),
# #     count: int = Query(1, description="Number of random rows to insert (default 1)")
# # ):
# #     """
# #     Generate realistic random data and insert into a specified table using its schema.
# #     """
# #     if not conn:
# #         raise HTTPException(status_code=500, detail="❌ Database connection is not available.")
    
# #     # Dynamically get table schema to know column types
# #     schema = get_table_schema(table)
# #     if not schema:
# #         raise HTTPException(status_code=404, detail=f"❌ Could not retrieve schema for table '{table}'. Please check if the table exists and the user has permissions.")

# #     rows = []
# #     for _ in range(count):
# #         random_row = {}
# #         for col_name, data_type, is_nullable, length, default_val in schema:
# #             # Use the existing, more generic generation functions
# #             random_row[col_name] = generate_random_value_for_column(col_name, data_type, is_nullable, length, default_val)
# #         rows.append(random_row)

# #     if not rows:
# #         return {
# #             "object": "random_insert_result",
# #             "table": table,
# #             "message": "✅ No records to insert (schema empty or generation failed)."
# #         }

# #     columns = list(rows[0].keys())
# #     col_clause = ', '.join(f'"{col}"' for col in columns)
# #     val_clause = ', '.join(['?' for _ in columns])
# #     sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_clause})'

# #     max_params = 32767
# #     batch_size = max_params // len(columns)
# #     total_inserted = 0

# #     cursor = conn.cursor()
# #     try:
# #         for i in range(0, len(rows), batch_size):
# #             batch = rows[i:i+batch_size]
# #             values_list = [list(row.values()) for row in batch]
# #             cursor.executemany(sql, values_list)
# #             total_inserted += len(batch)
# #         conn.commit()
# #     except dbapi.Error as e:
# #         conn.rollback()
# #         raise HTTPException(status_code=400, detail=f"❌ Database insert error: {str(e)}")
# #     finally:
# #         cursor.close()

# #     return {
# #         "object": "random_insert_result",
# #         "table": table,
# #         "message": f"✅ Successfully inserted {total_inserted} random records into '{table}'",
# #         "columns_inserted": len(columns),
# #         "sample_data": rows[0]
# #     }

# # # Running the server
# # if __name__ == "__main__":
# #     # Initialize and run the server
# #     mcp.run(transport='stdio')



# from typing import Dict, Any
# from mcp.server.fastmcp import FastMCP
# from fastapi import Query, HTTPException, Body
# from hdbcli import dbapi
# from dotenv import load_dotenv
# from fastapi import File, UploadFile
# import requests

# import os
# import random
# import uuid
# from datetime import datetime, timedelta
# import logging

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Initialize FastMCP server
# mcp = FastMCP("S4HANA_DB")

# HANA_HOST = os.getenv("HANA_HOST")
# HANA_PORT = int(os.getenv("HANA_PORT", "443"))
# HANA_USER = os.getenv("HANA_USER")
# HANA_PASS = os.getenv("HANA_PASS")
# HANA_SCHEMA = os.getenv("HANA_SCHEMA")

# # Connect to SAP HANA Cloud
# conn = dbapi.connect(
#     address=HANA_HOST,
#     port=HANA_PORT,
#     user=HANA_USER,
#     password=HANA_PASS,
#     encrypt=True,
#     sslValidateCertificate=False
# )

# # Implementing tool execution
# @mcp.tool(name="get_schema", description="Retrieve the database schema for SAP HANA")
# async def get_schema():
#     cursor = conn.cursor()
#     try:
#         cursor.execute(f"""
#             SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE_NAME
#             FROM SYS.TABLE_COLUMNS
#             WHERE SCHEMA_NAME = '{HANA_SCHEMA}'
#         """)
#         results = cursor.fetchall()
#     finally:
#         cursor.close()

#     schema: Dict[str, Dict] = {}
#     for table_name, column_name, data_type in results:
#         if table_name not in schema:
#             schema[table_name] = {"type": "table", "fields": []}
#         schema[table_name]["fields"].append({
#             "name": column_name,
#             "type": data_type.lower()
#         })

#     return {
#         "version": "1.0",
#         "schema": schema
#     }

# @mcp.tool(name="get_data", description="Retrieve data from a specified SAP HANA table")
# async def get_data(table: str = Query(..., description="SAP HANA table name")):
#     cursor = conn.cursor()
#     try:
#         cursor.execute(f'SELECT * FROM "{HANA_SCHEMA}"."{table}" LIMIT 100')
#         columns = [desc[0] for desc in cursor.description]
#         rows = cursor.fetchall()
#     except dbapi.Error as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     finally:
#         cursor.close()

#     return {
#         "object": "list",
#         "table": table,
#         "rows": [dict(zip(columns, row)) for row in rows]
#     }

# @mcp.tool(name="insert_data", description="Insert data into a specified SAP HANA table")
# def insert_data(
#     table: str = Query(..., description="SAP HANA table name"),
#     data: Dict[str, Any] = Body(..., description="Column-value pairs to insert")
# ):
#     cursor = conn.cursor()
#     try:
#         col_clause = ', '.join(f'"{col}"' for col in data.keys())
#         val_clause = ', '.join(['?' for _ in data])
#         values = list(data.values())

#         sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_clause})'
#         cursor.execute(sql, values)
#         conn.commit()
#     except dbapi.Error as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     finally:
#         cursor.close()

#     return {
#         "object": "insert_result",
#         "message": f"✅ Successfully inserted row into '{table}'",
#         "data": data
#     }

# @mcp.tool(name="delete_data", description="Delete data from a specified SAP HANA table")
# async def delete_data(
#     table: str = Query(..., description="SAP HANA table name"),
#     where: dict = Body(..., description="WHERE clause column-value pairs")
# ):
#     cursor = conn.cursor()
#     try:
#         where_clause = ' AND '.join(f'"{col}" = ?' for col in where.keys())
#         values = list(where.values())
#         sql = f'DELETE FROM "{HANA_SCHEMA}"."{table}" WHERE {where_clause}'
#         cursor.execute(sql, values)
#         conn.commit()
#     except dbapi.Error as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     finally:
#         cursor.close()
#     return {
#         "object": "delete_result",
#         "message": f"✅ Successfully deleted row(s) from '{table}'",
#         "where": where
#     }

# @mcp.tool(name="update_data", description="Update data in a specified SAP HANA table")
# async def update_data(
#     table: str = Query(..., description="SAP HANA table name"),
#     data: dict = Body(..., description="Column-value pairs to update"),
#     where: dict = Body(..., description="WHERE clause column-value pairs")
# ):
#     cursor = conn.cursor()
#     try:
#         set_clause = ', '.join(f'"{col}" = ?' for col in data.keys())
#         where_clause = ' AND '.join(f'"{col}" = ?' for col in where.keys())
#         values = list(data.values()) + list(where.values())
#         sql = f'UPDATE "{HANA_SCHEMA}"."{table}" SET {set_clause} WHERE {where_clause}'
#         cursor.execute(sql, values)
#         conn.commit()
#     except dbapi.Error as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     finally:
#         cursor.close()
#     return {
#         "object": "update_result",
#         "message": f"✅ Successfully updated row(s) in '{table}'",
#         "data": data,
#         "where": where
#     }

# @mcp.tool(name="generate_report", description="Generate a report from a custom SQL query or table")
# async def generate_report(
#     table: str = Query(..., description="SAP HANA table name (optional, auto-quoted if provided)"),
#     query: str = Body(None, description="SQL SELECT query to generate the report (optional)")
# ):
#     """
#     Execute a custom SQL SELECT query for reporting purposes.
#     If 'query' is not provided but 'table' is, auto-generate a SELECT statement with proper quoting.
#     """
#     cursor = conn.cursor()
#     try:
#         if query:
#             # Use the provided query as-is
#             cursor.execute(query)
#         elif table:
#             # Auto-generate a SELECT statement with quoted schema and table
#             cursor.execute(f'SELECT * FROM "{HANA_SCHEMA}"."{table}" LIMIT 100')
#         else:
#             raise HTTPException(status_code=400, detail="Either 'query' or 'table' must be provided.")

#         columns = [desc[0] for desc in cursor.description]
#         rows = cursor.fetchall()
#     except dbapi.Error as e:
#         raise HTTPException(status_code=400, detail=f"Database error executing query: {e}")
#     finally:
#         cursor.close()

#     return {
#         "object": "report",
#         "table": table,
#         "query": query if query else f'SELECT * FROM "{HANA_SCHEMA}"."{table}" LIMIT 100',
#         "rows": [dict(zip(columns, row)) for row in rows]
#     }

# def get_table_schema(table_name):
#     """Get the actual schema for a specific table with column lengths"""
#     cursor = conn.cursor()
#     try:
#         cursor.execute(f"""
#             SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE, LENGTH, DEFAULT_VALUE
#             FROM SYS.TABLE_COLUMNS
#             WHERE SCHEMA_NAME = '{HANA_SCHEMA}' AND TABLE_NAME = '{table_name}'
#             ORDER BY POSITION
#         """)
#         results = cursor.fetchall()
#         logger.debug(f"Schema for {table_name}: {results}")
#         return [(col_name, data_type, is_nullable, length, default_val) for col_name, data_type, is_nullable, length, default_val in results]
#     except Exception as e:
#         logger.error(f"Error getting schema for {table_name}: {e}")
#         return []
#     finally:
#         cursor.close()

# def generate_random_value_for_column(column_name, data_type, is_nullable, max_length, default_value=None):
#     """Enhanced random value generation with better error handling and logging"""
    
#     try:
#         logger.debug(f"Generating value for column: {column_name}, type: {data_type}, nullable: {is_nullable}, max_length: {max_length}")
        
#         # Handle nullable columns (10% chance of NULL for nullable columns)
#         if is_nullable == 'YES' and random.random() < 0.1:
#             return None
        
#         # Use default value if available for certain cases
#         if default_value and random.random() < 0.3:  # 30% chance to use default
#             return default_value
        
#         column_lower = column_name.lower()
        
#         # Determine safe max length for string values
#         safe_length = min(max_length or 50, 100) if max_length else 50
        
#         # String/Text types
#         if data_type.upper() in ['NVARCHAR', 'VARCHAR', 'NCLOB', 'CLOB', 'TEXT', 'STRING']:
#             return generate_string_value(column_lower, safe_length)
        
#         # Numeric types
#         elif data_type.upper() in ['INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT']:
#             return generate_integer_value(column_lower)
        
#         # Decimal/Float types
#         elif data_type.upper() in ['DECIMAL', 'DOUBLE', 'REAL', 'FLOAT']:
#             return generate_decimal_value(column_lower)
        
#         # Date types
#         elif data_type.upper() in ['DATE']:
#             return generate_date_value()
        
#         # Timestamp types
#         elif data_type.upper() in ['TIMESTAMP', 'SECONDDATE']:
#             return generate_timestamp_value()
        
#         # Boolean types
#         elif data_type.upper() in ['BOOLEAN']:
#             return random.choice([True, False])
        
#         # Default fallback
#         else:
#             logger.warning(f"Unknown data type {data_type} for column {column_name}, using default string")
#             base_default = f"Default_{random.randint(1, 999)}"
#             return base_default[:safe_length]
            
#     except Exception as e:
#         logger.error(f"Error generating value for column {column_name}: {e}")
#         # Return a safe default value based on the data type
#         if data_type.upper() in ['NVARCHAR', 'VARCHAR', 'NCLOB', 'CLOB', 'TEXT', 'STRING']:
#             return f"DEFAULT_{random.randint(1, 999)}"[:10]
#         elif data_type.upper() in ['INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT']:
#             return random.randint(1, 1000)
#         elif data_type.upper() in ['DECIMAL', 'DOUBLE', 'REAL', 'FLOAT']:
#             return round(random.uniform(1.0, 100.0), 2)
#         else:
#             return None

# def generate_string_value(column_lower, safe_length):
#     """Generate string values based on column name patterns"""

#     if 'guid' in column_lower or column_lower in ['rootguid', 'matchgroupid', 'externalobjectguid']:
#         return str(uuid.uuid4())
    
#     elif 'id' in column_lower and column_lower not in ['objectid', 'object_id', 'organization_id']:
#         base_value = f"ID{random.randint(100, 999)}"
#         return base_value[:safe_length]
#     elif 'code' in column_lower:
#         return generate_code_value(column_lower, safe_length)
#     elif 'desc' in column_lower or 'description' in column_lower:
#         descriptions = [
#             "Standard compliance transaction",
#             "Environmental credit transfer",
#             "Renewable fuel standard",
#             "Carbon offset transaction",
#             "Biofuel credit exchange"
#         ]
#         selected = random.choice(descriptions)
#         return selected[:safe_length]
#     elif 'name' in column_lower:
#         names = [
#             "EcoCredit_Transaction",
#             "RFS2_Compliance",
#             "LCFS_Transfer",
#             "Carbon_Credit",
#             "Biofuel_Exchange"
#         ]
#         selected = random.choice(names)
#         return selected[:safe_length]
#     elif 'status' in column_lower:
#         statuses = ['ACTIVE', 'PENDING', 'COMPLETED', 'CANCELLED', 'EXPIRED']
#         selected = random.choice(statuses)
#         return selected[:safe_length]
#     elif 'type' in column_lower:
#         types = ['PURCHASE', 'SALE', 'TRANSFER', 'GENERATION', 'RETIREMENT']
#         selected = random.choice(types)
#         return selected[:safe_length]
#     elif 'category' in column_lower:
#         categories = ['RFS2', 'LCFS', 'RGGI', 'CAP_TRADE', 'VOLUNTARY']
#         selected = random.choice(categories)
#         return selected[:safe_length]
#     elif 'currency' in column_lower:
#         currencies = ['USD']
#         selected = random.choice(currencies)
#         return selected[:safe_length]
#     elif 'user' in column_lower:
#         base_user = f"user_{random.randint(1, 99)}"
#         return base_user[:safe_length]
#     elif 'comment' in column_lower or 'note' in column_lower:
#         comments = [
#             "Standard processing completed",
#             "Additional documentation required",
#             "QA review in progress",
#             "Expedited transaction request",
#             "Compliance verification pending"
#         ]
#         selected = random.choice(comments)
#         return selected[:safe_length]
#     elif 'reference' in column_lower or 'ref' in column_lower:
#         base_ref = f"{random.randint(100000, 999999)}"
#         return base_ref[:safe_length]
#     elif 'batch' in column_lower:
#         base_batch = f"{random.randint(10000, 99999)}"
#         return base_batch[:safe_length]
#     elif 'external' in column_lower:
#         base_ext = f"{random.randint(1000000, 9999999)}"
#         return base_ext[:safe_length]
#     elif 'facility' in column_lower:
#         base_fac = f"{random.randint(1, 999):03d}"
#         return base_fac[:safe_length]
#     elif 'organization' in column_lower or 'org' in column_lower:
#         orgs = [ 'Generator', 'ObligatedParty', 'VoluntaryMarket']
#         selected = random.choice(orgs)
#         return selected[:safe_length]
#     else:
#         # Generic string value
#         if safe_length <= 5:
#             return f"V{random.randint(1, 999)}"[:safe_length]
#         elif safe_length <= 10:
#             return f"Val_{random.randint(1, 999)}"[:safe_length]
#         else:
#             return f"Value_{random.randint(1, 999)}"[:safe_length]

# def generate_code_value(column_lower, safe_length):
#     """Generate code values based on specific column patterns"""
    
#     if 'fuel' in column_lower:
#         fuel_codes = ['D3', 'D4', 'D5', 'D6', 'A1', 'E85']
#         selected = random.choice(fuel_codes)
#         return selected[:safe_length]
#     elif 'status' in column_lower:
#         status_codes = ['1', '2', '5', '6', '7', '8', '9', '10', '11', '12', '13']
#         selected = random.choice(status_codes)
#         return selected[:safe_length]
#     elif 'credit' in column_lower:
#         if safe_length <= 3:
#             return str(random.randint(1, 99))[:safe_length]
#         else:
#             return f"CR{random.randint(10, 99)}"[:safe_length]
#     elif 'compliance' in column_lower:
#         compliance_codes = ['RFS2', 'LCFS', 'RGGI', 'CAT']
#         selected = random.choice(compliance_codes)
#         return selected[:safe_length]
#     else:
#         if safe_length <= 5:
#             return f"C{random.randint(10, 99)}"[:safe_length]
#         else:
#             return f"CODE{random.randint(10, 99)}"[:safe_length]

# def generate_integer_value(column_lower):
#     """Generate integer values based on column patterns"""
    
#     if 'id' in column_lower:
#         if 'object' in column_lower:
#             return random.randint(6000, 9999)
#         else:
#             return random.randint(1000, 9999)
#     elif 'year' in column_lower:
#         return random.randint(2020, 2025)
#     elif 'month' in column_lower:
#         return random.randint(1, 12)
#     elif 'quarter' in column_lower:
#         return random.randint(1, 4)
#     elif 'code' in column_lower:
#         return random.randint(1, 100)
#     elif 'status' in column_lower:
#         return random.randint(1, 13)
#     elif 'quantity' in column_lower or 'amount' in column_lower:
#         return random.randint(100, 50000)
#     elif 'price' in column_lower:
#         return random.randint(1, 1000)
#     else:
#         return random.randint(1, 10000)

# def generate_decimal_value(column_lower):
#     """Generate decimal values based on column patterns"""
    
#     if 'price' in column_lower:
#         return round(random.uniform(0.50, 15.00), 2)
#     elif 'quantity' in column_lower or 'amount' in column_lower:
#         return round(random.uniform(100.0, 50000.0), 2)
#     elif 'rate' in column_lower:
#         return round(random.uniform(0.01, 1.0), 4)
#     elif 'percentage' in column_lower or 'percent' in column_lower:
#         return round(random.uniform(0.0, 100.0), 2)
#     else:
#         return round(random.uniform(1.0, 1000.0), 2)

# def generate_date_value():
#     """Generate a random date"""
#     base_date = datetime(2025, 1, 1)
#     random_days = random.randint(0, 730)  # 2 years range
#     return (base_date + timedelta(days=random_days)).strftime('%Y-%m-%d')

# def generate_timestamp_value():
#     """Generate a random timestamp"""
#     base_date = datetime(2025, 1, 1)
#     random_days = random.randint(0, 730)
#     random_hours = random.randint(0, 23)
#     random_minutes = random.randint(0, 59)
#     random_seconds = random.randint(0, 59)
#     return (base_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes, seconds=random_seconds)).strftime('%Y-%m-%d %H:%M:%S')

# @mcp.tool(name="generate_and_insert_random_entry", description="Generate and insert random entries into a specified SAP HANA table")
# async def generate_and_insert_random_entry(
#     table: str = Query(..., description="SAP HANA table name"),
#     count: int = Query(1, description="Number of random rows to insert (default 1)")
# ):
#     """
#     Generate realistic random data and insert into a specified table using its schema.
#     """
#     if not conn:
#         raise HTTPException(status_code=500, detail="❌ Database connection is not available.")
    
#     # Dynamically get table schema to know column types
#     schema = get_table_schema(table)
#     if not schema:
#         raise HTTPException(status_code=404, detail=f"❌ Could not retrieve schema for table '{table}'. Please check if the table exists and the user has permissions.")

#     rows = []
#     for _ in range(count):
#         random_row = {}
#         for col_name, data_type, is_nullable, length, default_val in schema:
#             # Use the existing, more generic generation functions
#             random_row[col_name] = generate_random_value_for_column(col_name, data_type, is_nullable, length, default_val)
#         rows.append(random_row)

#     if not rows:
#         return {
#             "object": "random_insert_result",
#             "table": table,
#             "message": "✅ No records to insert (schema empty or generation failed)."
#         }

#     columns = list(rows[0].keys())
#     col_clause = ', '.join(f'"{col}"' for col in columns)
#     val_clause = ', '.join(['?' for _ in columns])
#     sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_clause})'

#     max_params = 32767
#     batch_size = max_params // len(columns)
#     total_inserted = 0

#     cursor = conn.cursor()
#     try:
#         for i in range(0, len(rows), batch_size):
#             batch = rows[i:i+batch_size]
#             values_list = [list(row.values()) for row in batch]
#             cursor.executemany(sql, values_list)
#             total_inserted += len(batch)
#         conn.commit()
#     except dbapi.Error as e:
#         conn.rollback()
#         raise HTTPException(status_code=400, detail=f"❌ Database insert error: {str(e)}")
#     finally:
#         cursor.close()

#     return {
#         "object": "random_insert_result",
#         "table": table,
#         "message": f"✅ Successfully inserted {total_inserted} random records into '{table}'",
#         "columns_inserted": len(columns),
#         "sample_data": rows[0]
#     }

# @mcp.tool(name="upload_json_records", description="Upload JSON records to a specified SAP HANA table")
# async def upload_json_records(
#     table: str = Query(..., description="Target SAP HANA table name"),
#     records: list = Body(..., description="List of JSON objects representing records")
# ):
#     """Insert a batch of records (from a JSON-loaded .txt file) into the specified table."""
#     if not isinstance(records, list):
#         raise HTTPException(status_code=400, detail="`records` must be a list of JSON objects.")

#     cursor = conn.cursor()
#     inserted_count = 0
#     try:
#         for record in records:
#             if not isinstance(record, dict):
#                 continue

#             columns = list(record.keys())
#             values = list(record.values())

#             col_clause = ', '.join(f'"{col}"' for col in columns)
#             val_clause = ', '.join(['?' for _ in values])

#             sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_clause})'

#             cursor.execute(sql, values)
#             inserted_count += 1

#         conn.commit()
#     except dbapi.Error as e:
#         conn.rollback()
#         raise HTTPException(status_code=400, detail=f"Insert error: {e}")
#     finally:
#         cursor.close()

#     return {
#         "object": "upload_result",
#         "table": table,
#         "inserted": inserted_count,
#         "message": f"✅ Successfully inserted {inserted_count} record(s) into '{table}'"
#     }


# # Running the server
# if __name__ == "__main__":
#     # Initialize and run the server
#     mcp.run(transport='stdio')

from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP
from fastapi import Query, HTTPException, Body
from hdbcli import dbapi
from dotenv import load_dotenv
import os
import random
import uuid
from datetime import datetime, timedelta
import logging

# Set up logging for debugging purposes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("S4HANA_DB")

# Get database connection details from environment variables
HANA_HOST = os.getenv("HANA_HOST")
HANA_PORT = int(os.getenv("HANA_PORT", "443"))
HANA_USER = os.getenv("HANA_USER")
HANA_PASS = os.getenv("HANA_PASS")
HANA_SCHEMA = os.getenv("HANA_SCHEMA")

# Global connection variable
conn = None

def get_db_connection():
    """Establishes and returns a database connection."""
    global conn
    if conn is None or not conn.isconnected():
        try:
            conn = dbapi.connect(
                address=HANA_HOST,
                port=HANA_PORT,
                user=HANA_USER,
                password=HANA_PASS,
                encrypt=True,
                sslValidateCertificate=False
            )
            logger.info("✅ Successfully connected to SAP HANA database.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise HTTPException(status_code=500, detail="❌ Failed to connect to SAP HANA database.")
    return conn

def get_table_schema(table_name: str) -> List[tuple]:
    """Retrieves the schema (columns, data types, etc.) for a specific table."""
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        query = f"""
            SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE, LENGTH, DEFAULT_VALUE
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?
            ORDER BY POSITION
        """
        cursor.execute(query, (HANA_SCHEMA, table_name))
        results = cursor.fetchall()
        logger.debug(f"Retrieved schema for {table_name}: {results}")
        return [(col[0], col[1], col[2], col[3], col[4]) for col in results]
    except Exception as e:
        logger.error(f"Error getting schema for {table_name}: {e}")
        return []
    finally:
        cursor.close()

def generate_random_value_for_column(column_name: str, data_type: str, is_nullable: str, max_length: int, default_value: Any = None) -> Any:
    """Generates a realistic random value based on column properties."""
    try:
        if is_nullable == 'YES' and random.random() < 0.1:
            return None
        
        if default_value is not None and random.random() < 0.3:
            return default_value
        
        column_lower = column_name.lower()
        safe_length = min(max_length or 50, 100) if max_length else 50
        
        if data_type.upper() in ['NVARCHAR', 'VARCHAR', 'NCLOB', 'CLOB', 'TEXT', 'STRING']:
            if 'guid' in column_lower:
                return str(uuid.uuid4())
            elif 'code' in column_lower:
                return random.choice(['CDE1', 'CDE2', 'CDE3', 'CDE4', 'CDE5'])[:safe_length]
            elif 'name' in column_lower:
                return f"TransactionName_{random.randint(1, 1000)}".upper()[:safe_length]
            elif 'description' in column_lower or 'desc' in column_lower:
                return f"Random transaction description for item {random.randint(1000, 9999)}"[:safe_length]
            elif 'status' in column_lower:
                return random.choice(['PENDING', 'APPROVED', 'REJECTED', 'COMPLETED'])[:safe_length]
            else:
                return f"Value_{random.randint(1, 1000)}"[:safe_length]
        
        elif data_type.upper() in ['INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT']:
            if 'id' in column_lower:
                return random.randint(10000, 99999)
            elif 'quantity' in column_lower or 'amount' in column_lower:
                return random.randint(100, 50000)
            else:
                return random.randint(1, 1000)
            
        elif data_type.upper() in ['BOOLEAN']:
            # Correctly generate a boolean value that the database can convert
            return random.choice([True, False])

        elif data_type.upper() in ['DECIMAL', 'DOUBLE', 'REAL', 'FLOAT']:
            if 'price' in column_lower:
                return round(random.uniform(0.50, 15.00), 2)
            elif 'quantity' in column_lower or 'amount' in column_lower:
                return round(random.uniform(100.0, 50000.0), 2)
            else:
                return round(random.uniform(1.0, 1000.0), 2)
        
        elif data_type.upper() in ['DATE']:
            start_date = datetime(2023, 1, 1)
            end_date = datetime.now()
            return (start_date + timedelta(days=random.randint(0, (end_date - start_date).days))).strftime('%Y-%m-%d')
        
        elif data_type.upper() in ['TIMESTAMP', 'SECONDDATE']:
            start_date = datetime(2023, 1, 1)
            end_date = datetime.now()
            random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
            return random_date.strftime('%Y-%m-%d %H:%M:%S')

        else:
            return f"DEFAULT_VAL_{random.randint(1, 999)}"[:safe_length]

    except Exception as e:
        logger.warning(f"Failed to generate value for column '{column_name}' of type '{data_type}'. Error: {e}")
        return None

# The core function to generate and insert multiple records
@mcp.tool(name="generate_and_insert_random_entry", description="Generate and insert a specified number of random records into a given SAP HANA table.")
async def generate_and_insert_random_entry(
    table: str = Query(..., description="The name of the SAP HANA table to insert into."),
    count: int = Query(50, description="The number of random records to generate and insert. Default is 50.")
):
    """
    Generates a specified number of random records based on the target table's schema and performs a single, efficient
    batch insert using `executemany` into the SAP HANA database.
    """
    schema = get_table_schema(table)
    if not schema:
        raise HTTPException(status_code=404, detail=f"❌ Could not retrieve schema for table '{table}'.")

    # Generate all rows of data as a list of lists
    rows_to_insert = []
    for _ in range(count):
        row_data = [
            generate_random_value_for_column(col_name, data_type, is_nullable, length, default_val)
            for col_name, data_type, is_nullable, length, default_val in schema
        ]
        rows_to_insert.append(row_data)

    if not rows_to_insert:
        return {
            "status": "success",
            "message": "✅ No records were generated due to an empty schema or generation failure."
        }
        
    columns = [col[0] for col in schema]
    col_clause = ', '.join(f'"{col}"' for col in columns)
    val_placeholders = ', '.join(['?' for _ in columns])
    sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_placeholders})'

    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        # Perform the batch insert
        cursor.executemany(sql, rows_to_insert)
        connection.commit()
        
        logger.info(f"✅ Successfully inserted {count} records into '{table}'.")
        return {
            "status": "success",
            "message": f"✅ Successfully inserted {count} random records into '{table}'.",
            "total_records_inserted": count,
            "sample_record": dict(zip(columns, rows_to_insert[0]))
        }
    except dbapi.Error as e:
        connection.rollback()
        logger.error(f"❌ Database insert error: {e}")
        raise HTTPException(status_code=400, detail=f"❌ Database insert error: {str(e)}")
    finally:
        cursor.close()

# Other CRUD tools remain the same
# ----------------------------------------------------------------------------------------------------------------------
@mcp.tool(name="get_data", description="Retrieve data from a specified SAP HANA table")
async def get_data(table: str = Query(..., description="SAP HANA table name")):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(f'SELECT * FROM "{HANA_SCHEMA}"."{table}" LIMIT 100')
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        return {
            "object": "list",
            "table": table,
            "rows": [dict(zip(columns, row)) for row in rows]
        }
    except dbapi.Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()

@mcp.tool(name="insert_data", description="Insert a single record into a specified SAP HANA table")
def insert_data(
    table: str = Query(..., description="SAP HANA table name"),
    data: Dict[str, Any] = Body(..., description="Column-value pairs to insert")
):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        columns = list(data.keys())
        values = list(data.values())
        col_clause = ', '.join(f'"{col}"' for col in columns)
        val_clause = ', '.join(['?' for _ in values])
        sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_clause})'
        
        cursor.execute(sql, values)
        connection.commit()
        
        return {
            "object": "insert_result",
            "message": f"✅ Successfully inserted row into '{table}'",
            "data": data
        }
    except dbapi.Error as e:
        connection.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()

@mcp.tool(name="delete_data", description="Delete data from a specified SAP HANA table")
async def delete_data(
    table: str = Query(..., description="SAP HANA table name"),
    where: dict = Body(..., description="WHERE clause column-value pairs")
):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        where_clause = ' AND '.join(f'"{col}" = ?' for col in where.keys())
        values = list(where.values())
        sql = f'DELETE FROM "{HANA_SCHEMA}"."{table}" WHERE {where_clause}'
        
        cursor.execute(sql, values)
        connection.commit()
        
        return {
            "object": "delete_result",
            "message": f"✅ Successfully deleted row(s) from '{table}'",
            "where": where
        }
    except dbapi.Error as e:
        connection.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()

@mcp.tool(name="update_data", description="Update data in a specified SAP HANA table")
async def update_data(
    table: str = Query(..., description="SAP HANA table name"),
    data: dict = Body(..., description="Column-value pairs to update"),
    where: dict = Body(..., description="WHERE clause column-value pairs")
):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        set_clause = ', '.join(f'"{col}" = ?' for col in data.keys())
        where_clause = ' AND '.join(f'"{col}" = ?' for col in where.keys())
        values = list(data.values()) + list(where.values())
        sql = f'UPDATE "{HANA_SCHEMA}"."{table}" SET {set_clause} WHERE {where_clause}'
        
        cursor.execute(sql, values)
        connection.commit()
        
        return {
            "object": "update_result",
            "message": f"✅ Successfully updated row(s) in '{table}'",
            "data": data,
            "where": where
        }
    except dbapi.Error as e:
        connection.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()

@mcp.tool(name="generate_report", description="Generate a report from a custom SQL query or table")
async def generate_report(
    table: str = Query(None, description="SAP HANA table name (optional, auto-quoted if provided)"),
    query: str = Body(None, description="SQL SELECT query to generate the report (optional)")
):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        if query:
            cursor.execute(query)
        elif table:
            cursor.execute(f'SELECT * FROM "{HANA_SCHEMA}"."{table}" LIMIT 100')
        else:
            raise HTTPException(status_code=400, detail="Either 'query' or 'table' must be provided.")
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        return {
            "object": "report",
            "table": table,
            "query": query if query else f'SELECT * FROM "{HANA_SCHEMA}"."{table}" LIMIT 100',
            "rows": [dict(zip(columns, row)) for row in rows]
        }
    except dbapi.Error as e:
        raise HTTPException(status_code=400, detail=f"Database error executing query: {e}")
    finally:
        cursor.close()

@mcp.tool(name="upload_json_records", description="Upload JSON records to a specified SAP HANA table")
async def upload_json_records(
    table: str = Query(..., description="Target SAP HANA table name"),
    records: List[Dict[str, Any]] = Body(..., description="List of JSON objects representing records")
):
    if not isinstance(records, list) or not records:
        raise HTTPException(status_code=400, detail="`records` must be a non-empty list of JSON objects.")
    
    first_record = records[0]
    columns = list(first_record.keys())
    col_clause = ', '.join(f'"{col}"' for col in columns)
    val_clause = ', '.join(['?' for _ in columns])
    sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_clause})'
    
    data_to_insert = [list(rec.values()) for rec in records]
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        cursor.executemany(sql, data_to_insert)
        connection.commit()
        
        return {
            "object": "upload_result",
            "table": table,
            "inserted": len(data_to_insert),
            "message": f"✅ Successfully inserted {len(data_to_insert)} records into '{table}'."
        }
    except dbapi.Error as e:
        connection.rollback()
        raise HTTPException(status_code=400, detail=f"Insert error: {e}")
    finally:
        cursor.close()
        
if __name__ == "__main__":
    mcp.run(transport='stdio')