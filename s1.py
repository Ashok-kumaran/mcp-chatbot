from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from fastapi import Query, HTTPException, Body
from hdbcli import dbapi
from dotenv import load_dotenv
from fastapi import File, UploadFile

 
import os
import random
import uuid
from datetime import datetime, timedelta
 
# Load environment variables
load_dotenv()
 
# Initialize FastMCP server
mcp = FastMCP("S4HANA_DB")
 
 
HANA_HOST = os.getenv("HANA_HOST")
HANA_PORT = int(os.getenv("HANA_PORT", "443"))
HANA_USER = os.getenv("HANA_USER")
HANA_PASS = os.getenv("HANA_PASS")
HANA_SCHEMA = os.getenv("HANA_SCHEMA")
 
# Connect to SAP HANA Cloud
conn = dbapi.connect(
    address=HANA_HOST,
    port=HANA_PORT,
    user=HANA_USER,
    password=HANA_PASS,
    encrypt=True,
    sslValidateCertificate=False
)
 
#Implementing tool execution
@mcp.tool(name="get_schema", description="Retrieve the database schema for SAP HANA")
async def get_schema():
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE_NAME
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = '{HANA_SCHEMA}'
        """)
        results = cursor.fetchall()
    finally:
        cursor.close()
 
    schema: Dict[str, Dict] = {}
    for table_name, column_name, data_type in results:
        if table_name not in schema:
            schema[table_name] = {"type": "table", "fields": []}
        schema[table_name]["fields"].append({
            "name": column_name,
            "type": data_type.lower()
        })
 
    return {
        "version": "1.0",
        "schema": schema
    }
 
@mcp.tool(name="get_data", description="Retrieve data from a specified SAP HANA table")
async def get_data(table: str = Query(..., description="SAP HANA table name")):
    cursor = conn.cursor()
    try:
        cursor.execute(f'SELECT * FROM "{HANA_SCHEMA}"."{table}" LIMIT 100')
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except dbapi.Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
 
    return {
        "object": "list",
        "table": table,
        "rows": [dict(zip(columns, row)) for row in rows]
    }
 
@mcp.tool(name="insert_data", description="Insert data into a specified SAP HANA table")
def insert_data(
    table: str = Query(..., description="SAP HANA table name"),
    data: Dict[str, Any] = Body(..., description="Column-value pairs to insert")
):
    cursor = conn.cursor()
    try:
        col_clause = ', '.join(f'"{col}"' for col in data.keys())
        val_clause = ', '.join(['?' for _ in data])
        values = list(data.values())
 
        sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_clause})'
        cursor.execute(sql, values)
        conn.commit()
    except dbapi.Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
 
    return {
        "object": "insert_result",
        "message": f"✅ Successfully inserted row into '{table}'",
        "data": data
    }
 
@mcp.tool(name="delete_data", description="Delete data from a specified SAP HANA table")
async def delete_data(
    table: str = Query(..., description="SAP HANA table name"),
    where: dict = Body(..., description="WHERE clause column-value pairs")
):
    cursor = conn.cursor()
    try:
        where_clause = ' AND '.join(f'"{col}" = ?' for col in where.keys())
        values = list(where.values())
        sql = f'DELETE FROM "{HANA_SCHEMA}"."{table}" WHERE {where_clause}'
        cursor.execute(sql, values)
        conn.commit()
    except dbapi.Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
    return {
        "object": "delete_result",
        "message": f"✅ Successfully deleted row(s) from '{table}'",
        "where": where
    }
 
@mcp.tool(name="update_data", description="Update data in a specified SAP HANA table")
async def update_data(
    table: str = Query(..., description="SAP HANA table name"),
    data: dict = Body(..., description="Column-value pairs to update"),
    where: dict = Body(..., description="WHERE clause column-value pairs")
):
    cursor = conn.cursor()
    try:
        set_clause = ', '.join(f'"{col}" = ?' for col in data.keys())
        where_clause = ' AND '.join(f'"{col}" = ?' for col in where.keys())
        values = list(data.values()) + list(where.values())
        sql = f'UPDATE "{HANA_SCHEMA}"."{table}" SET {set_clause} WHERE {where_clause}'
        cursor.execute(sql, values)
        conn.commit()
    except dbapi.Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
    return {
        "object": "update_result",
        "message": f"✅ Successfully updated row(s) in '{table}'",
        "data": data,
        "where": where
    }
 
@mcp.tool(name="generate_report", description="Generate a report from a custom SQL query or table")
async def generate_report(
    table: str = Query(..., description="SAP HANA table name (optional, auto-quoted if provided)"),
    query: str = Body(None, description="SQL SELECT query to generate the report (optional)")
):
    """
    Execute a custom SQL SELECT query for reporting purposes.
    If 'query' is not provided but 'table' is, auto-generate a SELECT statement with proper quoting.
    """
    cursor = conn.cursor()
    try:
        if query:
            # Use the provided query as-is
            cursor.execute(query)
        elif table:
            # Auto-generate a SELECT statement with quoted schema and table
            cursor.execute(f'SELECT * FROM "{HANA_SCHEMA}"."{table}" LIMIT 100')
        else:
            raise HTTPException(status_code=400, detail="Either 'query' or 'table' must be provided.")
 
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except dbapi.Error as e:
        raise HTTPException(status_code=400, detail=f"Database error executing query: {e}")
    finally:
        cursor.close()
 
    return {
        "object": "report",
        "table": table,
        "query": query if query else f'SELECT * FROM "{HANA_SCHEMA}"."{table}" LIMIT 100',
        "rows": [dict(zip(columns, row)) for row in rows]
    }
 

 
@mcp.tool(name="upload_json_records", description="Upload JSON records to a specified SAP HANA table")
async def upload_json_records(
    table: str = Query(..., description="Target SAP HANA table name"),
    records: list = Body(..., description="List of JSON objects representing records")
):
    """Insert a batch of records (from a JSON-loaded .txt file) into the specified table."""
    if not isinstance(records, list):
        raise HTTPException(status_code=400, detail="`records` must be a list of JSON objects.")
 
    cursor = conn.cursor()
    inserted_count = 0
    try:
        for record in records:
            if not isinstance(record, dict):
                continue
 
            columns = list(record.keys())
            values = list(record.values())
 
            col_clause = ', '.join(f'"{col}"' for col in columns)
            val_clause = ', '.join(['?' for _ in values])
 
            sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_clause})'
 
            cursor.execute(sql, values)
            inserted_count += 1
 
        conn.commit()
    except dbapi.Error as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"Insert error: {e}")
    finally:
        cursor.close()
 
    return {
        "object": "upload_result",
        "table": table,
        "inserted": inserted_count,
        "message": f"✅ Successfully inserted {inserted_count} record(s) into '{table}'"
    }

@mcp.tool(name="generate_and_insert_random_entry", description="Generate and insert a random entry into a specified SAP HANA table")
async def generate_and_insert_random_entry(
    table: str = Query(..., description="SAP HANA table name")
):  
    """
    Generate realistic random data for EMTS table based on the provided schema.
    The function will create a random entry with realistic values for each column.
    """
    # Define realistic sample data pools for EMTS-specific fields
    fuel_codes = [3, 4, 5, 6, 7]
    fuel_descriptions = ['3-Cellulosic Biofuel', '4-Biomassed based Diesel', '5-Advanced Biofuel', '6-Renewable fuel', '7-Cellulosic biofuel']
 
    assignment_codes = [1, 2]
    assignment_descriptions = ['1-Assigned to fuel', '2-Separated per 40 CFR 80.1429']
 
    transaction_status_codes = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    transaction_status_descriptions = ['01-Proposed', '02-Checking', '05-Pending', '06-Processing', '07-Completed', '08-Failed', '09-Expired', '10-Cancelled', '11-Denied', '12-Removing', '13-Submission Failed']
 
    emts_Trad_Part_ptd = [1840, 5093, 474, 4491, 3683, 4828, 4029, 4808, 4725, 2012, 5685]
   
    transaction_type_codes = [1, 2, 3, 4, 5, 6, 7, 8]
    transaction_type_descriptions = ['01-Generate', '02-Separate', '03-Retire', '04-Buy', '05-Sell', '06-Unretire', '07-Lock', '08-Unlock']
 
    buy_sell_reason_codes = ['10-Standard Trade', '11-Spot Trade', '110-Tolling Agreement', '12-Term Contract Trade', '120-Intra company transfer', '13-Consignment trade', '30-Remedial Action – Incorrect Trading Partner', '50-Deny Trade', '60-Cancel Trade', '80-Remedial Action-Duplicate Trade']
    buy_sell_reason_descriptions = ['Purchase', 'Sale', 'Transfer']
 
    impacts = ['I', 'D']
    impact_descriptions = ['Increase', 'Decrease']
 
    object_categories = ['RIN', 'EMTS', 'FUEL']
    object_category_descriptions = ['Renewable Identification Number', 'EMTS Transaction', 'Fuel Transaction']
 
    organization_ids = ['9093']
    facility_ids = ['FAC001', 'FAC002', 'FAC003', 'FAC004', 'FAC005']
    currencies = ['USD']
    units_of_measurement = ['UG6']
    processing_statuses = ['Created', 'Pending', 'Expired', 'Completed']
    processing_status_descriptions = ['Created', 'Pending', 'Expired', 'Completed']
 
    now = datetime.now()
    random_data = {}
 
    try:
        random_data['attachedRINIndicator'] = random.randint(0, 1)
        random_data['objectId'] = random.randint(1000000, 9999999)
        random_data['transactionPartnerOrgId'] = random.randint(1000, 9999)
        random_data['transferComplianceYear'] = random.randint(2020, 2025)
        random_data['transferMonth'] = random.randint(1, 12)
        random_data['transferQuarter'] = random.randint(1, 4)
 
        random_data['billOfLading'] = f"BOL{random.randint(100000, 999999)}"
        random_data['businessPartnerDesc'] = f"Business Partner {random.randint(1, 100)}"
        random_data['cancelledUser'] = f"user{random.randint(1, 1000)}"
        random_data['createdBy'] = f"system_user_{random.randint(1, 50)}"
        random_data['modifiedBy'] = f"system_user_{random.randint(1, 50)}"
 
        random_data['emtsActionOutbound'] = random.choice(['SUBMIT', 'UPDATE', 'CANCEL'])
 
        idx = random.randint(0, len(assignment_codes) - 1)
        random_data['emtsAssignmentCode'] = assignment_codes[idx]
        random_data['emtsAssignmentCodeDesc'] = assignment_descriptions[idx]
 
        random_data['emtsBatchNumberText'] = f"BATCH{random.randint(10000, 99999)}"
        random_data['emtsBuyOrSellReasonCode'] = random.choice(buy_sell_reason_codes)
        random_data['emtsBuyOrSellReasonCodeDesc'] = random.choice(buy_sell_reason_descriptions)
        random_data['emtsTradingPartnerBillOfLading'] = f"TPBOL{random.randint(100000, 999999)}"
        random_data['emtsTradingPartnerInvoice'] = f"TPINV{random.randint(100000, 999999)}"
        random_data['emtsTradingPartnerPtd'] = random.choice(emts_Trad_Part_ptd)
 
        idx = random.randint(0, len(transaction_status_codes) - 1)
        random_data['emtsTransactionStatusCode'] = transaction_status_codes[idx]
        random_data['emtsTransactionStatusCodeDesc'] = transaction_status_descriptions[idx]
 
        idx = random.randint(0, len(transaction_type_codes) - 1)
        random_data['emtsTransactionTypeCode'] = transaction_type_codes[idx]
        random_data['emtsTransactionTypeCodeDesc'] = transaction_type_descriptions[idx]
 
       
        random_data['emtsTransmissionError'] = random.choice([None, 'TIMEOUT', 'INVALID_DATA', 'CONNECTION_ERROR'])
        random_data['emtsTransmissionStatus'] = random.choice([1, 2, 3])
 
        random_data['extTransactionNumber'] = f"EXT{random.randint(1000000, 9999999)}"
        random_data['matchedExtTransactionNumber'] = f"MEXT{random.randint(1000000, 9999999)}"
        random_data['ptdNumber'] = f"PTD{random.randint(100000, 999999)}"
        random_data['invoice'] = f"INV{random.randint(100000, 999999)}"
 
        comments = [
            "Standard transaction processing",
            "Additional documentation required",
            "Expedited processing requested",
            "Quality assurance review completed",
            "Compliance verification pending"
        ]
        random_data['externalComments'] = random.choice(comments)
        random_data['internalComments'] = random.choice(comments)
        random_data['regulationComments'] = random.choice(comments)
 
        random_data['externalObjectGUID'] = str(uuid.uuid4())
        random_data['rootGUID'] = str(uuid.uuid4())
        random_data['matchGroupId'] = str(uuid.uuid4())
 
        fuel_idx = random.randint(0, len(fuel_codes) - 1)
        random_data['fuelCode'] = fuel_codes[fuel_idx]
        random_data['fuelCodeDesc'] = fuel_descriptions[fuel_idx]
        random_data['fuelQuantity'] = round(random.uniform(1000.0, 50000.0), 2)
        random_data['fuelUnitOfMeasurement'] = random.choice(units_of_measurement)
 
        random_data['gallonPriceAmount'] = round(random.uniform(2.50, 6.00), 2)
        random_data['gallonPriceCurrency'] = random.choice(currencies)
        random_data['rinPriceAmount'] = round(random.uniform(0.50, 2.00), 2)
        random_data['rinPriceCurrency'] = random.choice(currencies)
 
        random_data['generateFacilityId'] = random.choice(facility_ids)
        random_data['generateOrganizationId'] = random.choice(organization_ids)
        random_data['organizationId'] = random.choice(organization_ids)
 
        impact_idx = random.randint(0, 1)
        random_data['impact'] = impacts[impact_idx]
        random_data['impactDesc'] = impact_descriptions[impact_idx]
 
        obj_cat_idx = random.randint(0, len(object_categories) - 1)
        random_data['objectCategory'] = object_categories[obj_cat_idx]
        random_data['objectCategoryDesc'] = object_category_descriptions[obj_cat_idx]
        random_data['objectType'] = random.choice(['TRANSACTION', 'BATCH', 'TRANSFER'])
        random_data['objectTypeDesc'] = random.choice(['Transaction Record', 'Batch Record', 'Transfer Record'])
        random_data['objectStatusDesc'] = random.choice(['Active', 'Pending', 'Complete', 'Cancelled'])
 
        idx = random.randint(0, len(processing_statuses) - 1)
        random_data['processingStatus'] = processing_statuses[idx]
        random_data['processingStatusDesc'] = processing_status_descriptions[idx]
 
        random_data['qapCode'] = random.choice([10, 30])
        random_data['qapCodeDesc'] = random.choice(['10-Q-RIN', '30-Unverified'])
        random_data['reasonCodeDesc'] = random.choice(['Normal Processing', 'Expedited Request', 'Correction', 'Resubmission'])
 
        random_data['regulationCategory'] = random.choice(['RFS2', 'LCFS', 'OTHER'])
        random_data['regulationCategoryDesc'] = random.choice(['Renewable Fuel Standard 2', 'Low Carbon Fuel Standard', 'Other Regulation'])
        random_data['regulationQuantity'] = round(random.uniform(500.0, 25000.0), 2)
        random_data['regulationType'] = random.choice(['FEDERAL', 'STATE', 'LOCAL'])
        random_data['regulationTypeDesc'] = random.choice(['Federal Regulation', 'State Regulation', 'Local Regulation'])
        random_data['regulationUnitOfMeasurement'] = random.choice(units_of_measurement)
 
        random_data['rinYear'] = str(random.randint(2020, 2025))
 
        scenario = random.choice(['RFS2_RF_CCO', 'RFS2_OTC_CCO', 'RFS2_MADJ_CCO'])
        random_data['subObjectScenario'] = scenario
        random_data['subObjectScenarioDesc'] = scenario
 
        random_data['submissionComplianceYear'] = str(random.randint(2020, 2025))
        random_data['submissionIdentifier'] = f"SUB{random.randint(1000000, 9999999)}"
        random_data['submissionMonth'] = f"{random.randint(1, 12):02d}"
        random_data['submissionQuarter'] = f"{random.randint(1, 4):02d}"
 
        trade = random.choice(['Initiated', 'Received'])
        random_data['tradeSource'] = trade
 
        trans_cat = random.choice(['SAL', 'PUR'])
        random_data['transactionCategory'] = trans_cat
        random_data['transactionCategoryDesc'] = trans_cat
 
        trans_type = random.choice(['EXP', 'IMP', 'MSL', 'PRD', 'PUR', 'REG', 'SAL', 'TRN'])
        random_data['transactionType'] = trans_type
        random_data['transactionTypeDesc'] = trans_type
 
        base_date = datetime(2023, 1, 1)
        random_days = random.randint(0, 700)
 
        random_data['createdAt'] = (base_date + timedelta(days=random_days)).strftime('%Y-%m-%d %H:%M:%S')
        random_data['modifiedAt'] = (base_date + timedelta(days=random_days + random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S')
        random_data['emtsExpirationDate'] = (base_date + timedelta(days=random_days + random.randint(365, 730))).strftime('%Y-%m-%d')
        random_data['submissionDate'] = (base_date + timedelta(days=random_days + random.randint(-30, 30))).strftime('%Y-%m-%d')
        random_data['transferDate'] = (base_date + timedelta(days=random_days + random.randint(-10, 10))).strftime('%Y-%m-%d')
 
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"❌ Error generating random data: {str(e)}")
 
    # Insert the data into the database
    cursor = conn.cursor()
    try:
        col_clause = ', '.join(f'"{col}"' for col in random_data.keys())
        val_clause = ', '.join(['?' for _ in random_data])
        values = list(random_data.values())
 
        sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_clause})'
 
        print(f"Executing SQL: {sql}")
        print(f"Sample values: {dict(list(random_data.items())[:5])}...")
 
        cursor.execute(sql, values)
        conn.commit()
 
        row_count = cursor.rowcount
 
    except dbapi.Error as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"❌ Database insert error: {str(e)}")
    finally:
        cursor.close()
 
    return {
        "object": "random_insert_result",
        "table": table,
        "message": f"✅ Successfully inserted {row_count} realistic record into '{table}'",
        "columns_inserted": len(random_data),
        "sample_data": {
            "billOfLading": random_data.get('billOfLading'),
            "invoice": random_data.get('invoice'),
            "gallonPriceAmount": random_data.get('gallonPriceAmount'),
            "fuelCode": random_data.get('fuelCode'),
            "fuelQuantity": random_data.get('fuelQuantity'),
            "internalComments": random_data.get('internalComments'),
            "rinPriceAmount": random_data.get('rinPriceAmount'),
            "rinYear": random_data.get('rinYear'),
            "transferDate": random_data.get('transferDate')          
           
        }
    }

# run the server
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
 