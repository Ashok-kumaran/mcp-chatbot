from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from fastapi import Query, HTTPException, Body
from hdbcli import dbapi
from dotenv import load_dotenv
from fastapi import File, UploadFile
import requests

import os
import random
import uuid
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

def get_table_schema(table_name):
    """Get the actual schema for a specific table with column lengths"""
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE, LENGTH, DEFAULT_VALUE
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = '{HANA_SCHEMA}' AND TABLE_NAME = '{table_name}'
            ORDER BY POSITION
        """)
        results = cursor.fetchall()
        logger.debug(f"Schema for {table_name}: {results}")
        return [(col_name, data_type, is_nullable, length, default_val) for col_name, data_type, is_nullable, length, default_val in results]
    except Exception as e:
        logger.error(f"Error getting schema for {table_name}: {e}")
        return []
    finally:
        cursor.close()

def generate_random_value_for_column(column_name, data_type, is_nullable, max_length, default_value=None):
    """Enhanced random value generation with better error handling and logging"""
    
    try:
        logger.debug(f"Generating value for column: {column_name}, type: {data_type}, nullable: {is_nullable}, max_length: {max_length}")
        
        # Handle nullable columns (10% chance of NULL for nullable columns)
        if is_nullable == 'YES' and random.random() < 0.1:
            return None
        
        # Use default value if available for certain cases
        if default_value and random.random() < 0.3:  # 30% chance to use default
            return default_value
        
        column_lower = column_name.lower()
        
        # Determine safe max length for string values
        safe_length = min(max_length or 50, 100) if max_length else 50
        
        # String/Text types
        if data_type.upper() in ['NVARCHAR', 'VARCHAR', 'NCLOB', 'CLOB', 'TEXT', 'STRING']:
            return generate_string_value(column_lower, safe_length)
        
        # Numeric types
        elif data_type.upper() in ['INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT']:
            return generate_integer_value(column_lower)
        
        # Decimal/Float types
        elif data_type.upper() in ['DECIMAL', 'DOUBLE', 'REAL', 'FLOAT']:
            return generate_decimal_value(column_lower)
        
        # Date types
        elif data_type.upper() in ['DATE']:
            return generate_date_value()
        
        # Timestamp types
        elif data_type.upper() in ['TIMESTAMP']:
            return generate_timestamp_value()
        
        # Boolean types
        elif data_type.upper() in ['BOOLEAN']:
            return random.choice([True, False])
        
        # Default fallback
        else:
            logger.warning(f"Unknown data type {data_type} for column {column_name}, using default string")
            base_default = f"Default_{random.randint(1, 999)}"
            return base_default[:safe_length]
            
    except Exception as e:
        logger.error(f"Error generating value for column {column_name}: {e}")
        # Return a safe default value based on the data type
        if data_type.upper() in ['NVARCHAR', 'VARCHAR', 'NCLOB', 'CLOB', 'TEXT', 'STRING']:
            return f"DEFAULT_{random.randint(1, 999)}"[:10]
        elif data_type.upper() in ['INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT']:
            return random.randint(1, 1000)
        elif data_type.upper() in ['DECIMAL', 'DOUBLE', 'REAL', 'FLOAT']:
            return round(random.uniform(1.0, 100.0), 2)
        else:
            return None

def generate_string_value(column_lower, safe_length):
    """Generate string values based on column name patterns"""
    
    if 'id' in column_lower and column_lower != 'objectid':
        base_value = f"ID{random.randint(100, 999)}"
        return base_value[:safe_length]
    elif 'guid' in column_lower or 'uuid' in column_lower:
        guid_value = str(uuid.uuid4())
        return guid_value[:safe_length]
    elif 'code' in column_lower:
        return generate_code_value(column_lower, safe_length)
    elif 'desc' in column_lower or 'description' in column_lower:
        descriptions = [
            "Standard compliance transaction",
            "Environmental credit transfer",
            "Renewable fuel standard",
            "Carbon offset transaction",
            "Biofuel credit exchange"
        ]
        selected = random.choice(descriptions)
        return selected[:safe_length]
    elif 'name' in column_lower:
        names = [
            "EcoCredit_Transaction",
            "RFS2_Compliance",
            "LCFS_Transfer",
            "Carbon_Credit",
            "Biofuel_Exchange"
        ]
        selected = random.choice(names)
        return selected[:safe_length]
    elif 'status' in column_lower:
        statuses = ['ACTIVE', 'PENDING', 'COMPLETED', 'CANCELLED', 'EXPIRED']
        selected = random.choice(statuses)
        return selected[:safe_length]
    elif 'type' in column_lower:
        types = ['PURCHASE', 'SALE', 'TRANSFER', 'GENERATION', 'RETIREMENT']
        selected = random.choice(types)
        return selected[:safe_length]
    elif 'category' in column_lower:
        categories = ['RFS2', 'LCFS', 'RGGI', 'CAP_TRADE', 'VOLUNTARY']
        selected = random.choice(categories)
        return selected[:safe_length]
    elif 'currency' in column_lower:
        currencies = ['USD']
        selected = random.choice(currencies)
        return selected[:safe_length]
    elif 'user' in column_lower:
        base_user = f"user_{random.randint(1, 99)}"
        return base_user[:safe_length]
    elif 'comment' in column_lower or 'note' in column_lower:
        comments = [
            "Standard processing completed",
            "Additional documentation required",
            "QA review in progress",
            "Expedited transaction request",
            "Compliance verification pending"
        ]
        selected = random.choice(comments)
        return selected[:safe_length]
    elif 'reference' in column_lower or 'ref' in column_lower:
        base_ref = f"REF{random.randint(100000, 999999)}"
        return base_ref[:safe_length]
    elif 'batch' in column_lower:
        base_batch = f"BATCH{random.randint(10000, 99999)}"
        return base_batch[:safe_length]
    elif 'external' in column_lower:
        base_ext = f"EXT{random.randint(1000000, 9999999)}"
        return base_ext[:safe_length]
    elif 'facility' in column_lower:
        base_fac = f"FAC{random.randint(1, 999):03d}"
        return base_fac[:safe_length]
    elif 'organization' in column_lower or 'org' in column_lower:
        orgs = [ 'Generator', 'ObligatedParty', 'VoluntaryMarket']
        selected = random.choice(orgs)
        return selected[:safe_length]
    else:
        # Generic string value
        if safe_length <= 5:
            return f"V{random.randint(1, 999)}"[:safe_length]
        elif safe_length <= 10:
            return f"Val_{random.randint(1, 999)}"[:safe_length]
        else:
            return f"Value_{random.randint(1, 999)}"[:safe_length]

def generate_code_value(column_lower, safe_length):
    """Generate code values based on specific column patterns"""
    
    if 'fuel' in column_lower:
        fuel_codes = ['D3', 'D4', 'D5', 'D6', 'A1', 'E85']
        selected = random.choice(fuel_codes)
        return selected[:safe_length]
    elif 'status' in column_lower:
        status_codes = ['1', '2', '5', '6', '7', '8', '9', '10', '11', '12', '13']
        selected = random.choice(status_codes)
        return selected[:safe_length]
    elif 'credit' in column_lower:
        if safe_length <= 3:
            return str(random.randint(1, 99))[:safe_length]
        else:
            return f"CR{random.randint(10, 99)}"[:safe_length]
    elif 'compliance' in column_lower:
        compliance_codes = ['RFS2', 'LCFS', 'RGGI', 'CAT']
        selected = random.choice(compliance_codes)
        return selected[:safe_length]
    else:
        if safe_length <= 5:
            return f"C{random.randint(10, 99)}"[:safe_length]
        else:
            return f"CODE{random.randint(10, 99)}"[:safe_length]

def generate_integer_value(column_lower):
    """Generate integer values based on column patterns"""
    
    if 'id' in column_lower:
        if 'object' in column_lower:
            return random.randint(6000, 9999)
        else:
            return random.randint(1000, 9999)
    elif 'year' in column_lower:
        return random.randint(2020, 2025)
    elif 'month' in column_lower:
        return random.randint(1, 12)
    elif 'quarter' in column_lower:
        return random.randint(1, 4)
    elif 'code' in column_lower:
        return random.randint(1, 100)
    elif 'status' in column_lower:
        return random.randint(1, 13)
    elif 'quantity' in column_lower or 'amount' in column_lower:
        return random.randint(100, 50000)
    elif 'price' in column_lower:
        return random.randint(1, 1000)
    else:
        return random.randint(1, 10000)

def generate_decimal_value(column_lower):
    """Generate decimal values based on column patterns"""
    
    if 'price' in column_lower:
        return round(random.uniform(0.50, 15.00), 2)
    elif 'quantity' in column_lower or 'amount' in column_lower:
        return round(random.uniform(100.0, 50000.0), 2)
    elif 'rate' in column_lower:
        return round(random.uniform(0.01, 1.0), 4)
    elif 'percentage' in column_lower or 'percent' in column_lower:
        return round(random.uniform(0.0, 100.0), 2)
    else:
        return round(random.uniform(1.0, 1000.0), 2)

def generate_date_value():
    """Generate a random date"""
    base_date = datetime(2025, 1, 1)
    random_days = random.randint(0, 730)  # 2 years range
    return (base_date + timedelta(days=random_days)).strftime('%Y-%m-%d')

def generate_timestamp_value():
    """Generate a random timestamp"""
    base_date = datetime(2025, 1, 1)
    random_days = random.randint(0, 730)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    random_seconds = random.randint(0, 59)
    return (base_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes, seconds=random_seconds)).strftime('%Y-%m-%d %H:%M:%S')

@mcp.tool(name="generate_and_insert_random_entry", description="Generate and insert random entries into a specified SAP HANA table")
async def generate_and_insert_random_entry(
    table: str = Query(..., description="SAP HANA table name"),
    count: int = Query(1, description="Number of random rows to insert (default 1)")
):  
    """
    Generate realistic random data for EMTS table and insert multiple rows.
    Only the generated columns are included in the INSERT; other columns remain NULL/default.
    """

    # ---- constants (keep names consistent with usage below) ----
    FUELCODE = [3, 4, 5, 6, 7]
    FUELDESCRIPTION = [
        '3-Cellulosic Biofuel',
        '4-Biomassed based Diesel',
        '5-Advanced Biofuel',
        '6-Renewable fuel',
        '7-Cellulosic biofuel'
    ]

    ASSIGNMENTCODES = [1, 2]
    ASSIGNMENTDESCRIPTIONS = [
        '1-Assigned to fuel',
        '2-Separated per 40 CFR 80.1429'
    ]

    TRANSACTIONSTATUSCODES = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    TRANSACTIONSTATUSDESCRIPTIONS = [
        '01-Proposed', '02-Checking', '05-Pending', '06-Processing', '07-Completed',
        '08-Failed', '09-Expired', '10-Cancelled', '11-Denied', '12-Removing', '13-Submission Failed'
    ]

    EMTSTRADEPARTPTD = [1840, 5093, 474, 4491, 3683, 4828, 4029, 4808, 4725, 2012, 5685]

    TRANSACTIONTYPECODES = [1, 2, 3, 4, 5, 6, 7, 8]
    TRANSACTIONTYPEDESCRIPTIONS = [
        '01-Generate', '02-Separate', '03-Retire', '04-Buy',
        '05-Sell', '06-Unretire', '07-Lock', '08-Unlock'
    ]

    BUYSELLREASONCODES = [
        '10-Standard Trade', '11-Spot Trade', '110-Tolling Agreement',
        '12-Term Contract Trade', '120-Intra company transfer', '13-Consignment trade',
        '30-Remedial Action – Incorrect Trading Partner', '50-Deny Trade',
        '60-Cancel Trade', '80-Remedial Action-Duplicate Trade'
    ]
    BUYSELLREASONDESCRIPTIONS = ['Purchase', 'Sale', 'Transfer']

    IMPACTS = ['I', 'D']
    IMPACT_DESCRIPTIONS = ['Increase', 'Decrease']

    OBJECT_CATEGORIES = ['RIN', 'EMTS', 'FUEL']
    OBJECT_CATEGORY_DESCRIPTIONS = [
        'Renewable Identification Number', 'EMTS Transaction', 'Fuel Transaction'
    ]

    ORGANIZATION_IDS = ['9093']
    FACILITY_IDS = ['FAC001', 'FAC002', 'FAC003', 'FAC004', 'FAC005']
    CURRENCIES = ['USD']
    UNITS_OF_MEASUREMENT = ['UG6']
    PROCESSING_STATUSES = ['Created', 'Pending', 'Expired', 'Completed']
    PROCESSING_STATUS_DESCRIPTIONS = ['Created', 'Pending', 'Expired', 'Completed']

    def make_random_row():
        random_data = {}
        try:
            random_data['ATTACHEDRININDICATOR'] = random.randint(0, 1)
            random_data['OBJECTID'] = random.randint(1000000, 9999999)
            random_data['TRANSACTIONPARTNERORGID'] = random.randint(1000, 9999)
            random_data['TRANSFERCOMPLIANCEYEAR'] = random.randint(2020, 2025)
            random_data['TRANSFERMONTH'] = random.randint(1, 12)
            random_data['TRANSFERQUARTER'] = random.randint(1, 4)

            random_data['BILLOFLADING'] = f"BOL{random.randint(100000, 999999)}"
            random_data['BUSINESSPARTNERDESC'] = f"Business Partner {random.randint(1, 100)}"
            random_data['CANCELLEDUSER'] = f"user{random.randint(1, 1000)}"
            random_data['CREATEDBY'] = f"system_user_{random.randint(1, 50)}"
            random_data['MODIFIEDBY'] = f"system_user_{random.randint(1, 50)}"

            random_data['EMTSACTIONOUTBOUND'] = random.choice(['SUBMIT', 'UPDATE', 'CANCEL'])

            idx = random.randint(0, len(ASSIGNMENTCODES) - 1)
            random_data['EMTSASSIGNMENTCODE'] = ASSIGNMENTCODES[idx]
            random_data['EMTSASSIGNMENTCODEDESC'] = ASSIGNMENTDESCRIPTIONS[idx]

            random_data['EMTSBATCHNUMBERTEXT'] = f"BATCH{random.randint(10000, 99999)}"
            random_data['EMTSBUYORSELLREASONCODE'] = random.choice(BUYSELLREASONCODES)
            random_data['EMTSBUYORSELLREASONCODEDESC'] = random.choice(BUYSELLREASONDESCRIPTIONS)
            random_data['EMTSTRADINGPARTNERBILLOFLADING'] = f"TPBOL{random.randint(100000, 999999)}"
            random_data['EMTSTRADINGPARTNERINVOICE'] = f"TPINV{random.randint(100000, 999999)}"
            random_data['EMTSTRADINGPARTNERPTD'] = random.choice(EMTSTRADEPARTPTD)

            idx = random.randint(0, len(TRANSACTIONSTATUSCODES) - 1)
            random_data['EMTSTRANSACTIONSTATUSCODE'] = TRANSACTIONSTATUSCODES[idx]
            random_data['EMTSTRANSACTIONSTATUSCODEDESC'] = TRANSACTIONSTATUSDESCRIPTIONS[idx]

            idx = random.randint(0, len(TRANSACTIONTYPECODES) - 1)
            random_data['EMTSTRANSACTIONTYPECODE'] = TRANSACTIONTYPECODES[idx]
            random_data['EMTSTRANSACTIONTYPECODEDESC'] = TRANSACTIONTYPEDESCRIPTIONS[idx]

            random_data['EMTSTRANSMISSIONERROR'] = random.choice([None, 'TIMEOUT', 'INVALID_DATA', 'CONNECTION_ERROR'])
            random_data['EMTSTRANSMISSIONSTATUS'] = random.choice([1, 2, 3])

            random_data['EXTTRANSACTIONNUMBER'] = f"EXT{random.randint(1000000, 9999999)}"
            random_data['PTDNUMBER'] = f"PTD{random.randint(100000, 999999)}"
            random_data['INVOICE'] = f"INV{random.randint(100000, 999999)}"

            comments = [
                "Standard transaction processing",
                "Additional documentation required",
                "Expedited processing requested", 
                "Quality assurance review completed",
                "Compliance verification pending"
            ]
            # Use consistent casing for keys (assuming uppercase column names)
            random_data['EXTERNALCOMMENTS'] = random.choice(comments)
            random_data['INTERNALCOMMENTS'] = random.choice(comments)
            random_data['REGULATIONCOMMENTS'] = random.choice(comments)

            random_data['EXTERNALOBJECTGUID'] = str(uuid.uuid4())
            random_data['ROOTGUID'] = str(uuid.uuid4())
            random_data['MATCHGROUPID'] = str(uuid.uuid4())

            fuel_idx = random.randint(0, len(FUELCODE) - 1)
            random_data['FUELCODE'] = FUELCODE[fuel_idx]
            random_data['FUELCODEDESC'] = FUELDESCRIPTION[fuel_idx]
            random_data['FUELQUANTITY'] = round(random.uniform(1000.0, 50000.0), 2)
            random_data['FUELUNITOFMEASUREMENT'] = random.choice(UNITS_OF_MEASUREMENT)

            random_data['GALLONPRICEAMOUNT'] = round(random.uniform(2.50, 6.00), 2)
            random_data['GALLONPRICECURRENCY'] = random.choice(CURRENCIES)
            random_data['RINPRICEAMOUNT'] = round(random.uniform(0.50, 2.00), 2)
            random_data['RINPRICECURRENCY'] = random.choice(CURRENCIES)

            random_data['GENERATEFACILITYID'] = random.choice(FACILITY_IDS)
            random_data['GENERATEORGANIZATIONID'] = random.choice(ORGANIZATION_IDS)
            random_data['ORGANIZATIONID'] = random.choice(ORGANIZATION_IDS)

            impact_idx = random.randint(0, 1)
            random_data['IMPACT'] = IMPACTS[impact_idx]
            random_data['IMPACTDESC'] = IMPACT_DESCRIPTIONS[impact_idx]

            obj_cat_idx = random.randint(0, len(OBJECT_CATEGORIES) - 1)
            random_data['OBJECTCATEGORY'] = OBJECT_CATEGORIES[obj_cat_idx]
            random_data['OBJECTCATEGORYDESC'] = OBJECT_CATEGORY_DESCRIPTIONS[obj_cat_idx]
            random_data['OBJECTTYPE'] = random.choice(['TRANSACTION', 'BATCH', 'TRANSFER'])
            random_data['OBJECTTYPEDESC'] = random.choice(['Transaction Record', 'Batch Record', 'Transfer Record'])
            random_data['OBJECTSTATUSDESC'] = random.choice(['Active', 'Pending', 'Complete', 'Cancelled'])

            idx = random.randint(0, len(PROCESSING_STATUSES) - 1)
            random_data['PROCESSINGSTATUS'] = PROCESSING_STATUSES[idx]
            random_data['PROCESSINGSTATUSDESC'] = PROCESSING_STATUS_DESCRIPTIONS[idx]

            random_data['QAPCODE'] = random.choice([10, 30])
            random_data['QAPCODEDESC'] = random.choice(['10-Q-RIN', '30-Unverified'])
            random_data['REASONCODEDESC'] = random.choice(['Normal Processing', 'Expedited Request', 'Correction', 'Resubmission'])

            random_data['REGULATIONCATEGORY'] = random.choice(['RFS2', 'LCFS', 'OTHER'])
            random_data['REGULATIONCATEGORYDESC'] = random.choice(['Renewable Fuel Standard 2', 'Low Carbon Fuel Standard', 'Other Regulation'])
            random_data['REGULATIONQUANTITY'] = round(random.uniform(500.0, 25000.0), 2)
            random_data['REGULATIONTYPE'] = random.choice(['FEDERAL', 'STATE', 'LOCAL'])
            random_data['REGULATIONTYPEDESC'] = random.choice(['Federal Regulation', 'State Regulation', 'Local Regulation'])
            random_data['REGULATIONUNITOFMEASUREMENT'] = random.choice(UNITS_OF_MEASUREMENT)

            random_data['RINYEAR'] = str(random.randint(2020, 2025))

            scenario = random.choice(['RFS2_RF_CCO', 'RFS2_OTC_CCO', 'RFS2_MADJ_CCO'])
            random_data['SUBOBJECTSCENARIO'] = scenario
            random_data['SUBOBJECTSCENARIODESC'] = scenario

            random_data['SUBMISSIONCOMPLIANCEYEAR'] = str(random.randint(2020, 2025))
            random_data['SUBMISSIONIDENTIFIER'] = f"SUB{random.randint(1000000, 9999999)}"
            random_data['SUBMISSIONMONTH'] = f"{random.randint(1, 12):02d}"
            random_data['SUBMISSIONQUARTER'] = f"{random.randint(1, 4):02d}"

            trade = random.choice(['Initiated', 'Received'])
            random_data['TRADESOURCE'] = trade

            trans_cat = random.choice(['SAL', 'PUR'])
            random_data['TRANSACTIONCATEGORY'] = trans_cat
            random_data['TRANSACTIONCATEGORYDESC'] = trans_cat

            trans_type = random.choice(['EXP', 'IMP', 'MSL', 'PRD', 'PUR', 'REG', 'SAL', 'TRN'])
            random_data['TRANSACTIONTYPE'] = trans_type
            random_data['TRANSACTIONTYPEDESC'] = trans_type

            base_date = datetime(2023, 1, 1)
            random_days = random.randint(0, 700)

            random_data['CREATEDAT'] = (base_date + timedelta(days=random_days)).strftime('%Y-%m-%d %H:%M:%S')
            random_data['MODIFIEDAT'] = (base_date + timedelta(days=random_days + random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S')
            random_data['EMTSEXPIRATIONDATE'] = (base_date + timedelta(days=random_days + random.randint(365, 730))).strftime('%Y-%m-%d')
            random_data['SUBMISSIONDATE'] = (base_date + timedelta(days=random_days + random.randint(-30, 30))).strftime('%Y-%m-%d')
            random_data['TRANSFERDATE'] = (base_date + timedelta(days=random_days + random.randint(-10, 10))).strftime('%Y-%m-%d')

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"❌ Error generating random data: {str(e)}")

        return random_data

    # ---- generate rows and insert ----
    n = max(1, int(count))
    rows = [make_random_row() for _ in range(n)]
    columns = list(rows[0].keys())

    col_clause = ', '.join(f'"{col}"' for col in columns)
    val_clause = ', '.join(['?' for _ in columns])
    sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_clause})'

    # batch insert
    max_params = 32767
    batch_size = max(1, max_params // max(1, len(columns)))
    total_inserted = 0

    cursor = conn.cursor()
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            values_list = [[row[c] for c in columns] for row in batch]
            cursor.executemany(sql, values_list)
            total_inserted += len(batch)
        conn.commit()
    except dbapi.Error as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"❌ Database insert error: {str(e)}")
    finally:
        cursor.close()

    return {
        "object": "random_insert_result",
        "table": table,
        "message": f"✅ Successfully inserted {total_inserted} random records into '{table}'",
        "columns_inserted": len(columns),
        "sample_data": rows[0]
    }

#Running the server
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')