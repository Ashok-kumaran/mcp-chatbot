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

# EMTS-specific data pools
class EMTSDataPools:
    """Centralized EMTS-specific data for realistic generation"""
    
    fuel_codes = [3, 4, 5, 6, 7]
    fuel_descriptions = ['3-Cellulosic Biofuel', '4-Biomassed based Diesel', '5-Advanced Biofuel', '6-Renewable fuel', '7-Cellulosic biofuel']

    assignment_codes = [1, 2]
    assignment_descriptions = ['1-Assigned to fuel', '2-Separated per 40 CFR 80.1429']

    transaction_status_codes = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    transaction_status_descriptions = ['01-Proposed', '02-Checking', '05-Pending', '06-Processing', '07-Completed', '08-Failed', '09-Expired', '10-Cancelled', '11-Denied', '12-Removing', '13-Submission Failed']

    emts_trading_partners = [1840, 5093, 474, 4491, 3683, 4828, 4029, 4808, 4725, 2012, 5685]
    
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

    emts_actions = ['SUBMIT', 'UPDATE', 'CANCEL']
    transmission_errors = [None, 'TIMEOUT', 'INVALID_DATA', 'CONNECTION_ERROR']
    transmission_statuses = [1, 2, 3]
    
    comments = [
        "Standard transaction processing",
        "Additional documentation required", 
        "Expedited processing requested", 
        "Quality assurance review completed",
        "Compliance verification pending"
    ]

def is_emts_table(table_name):
    """Check if this is an EMTS-related table based on name patterns"""
    emts_patterns = ['emts', 'rin', 'fuel', 'credit', 'compliance', 'transaction']
    table_lower = table_name.lower()
    return any(pattern in table_lower for pattern in emts_patterns)

def generate_emts_specific_value(column_name, data_type, is_nullable, max_length):
    """Generate EMTS-specific values for known EMTS columns"""
    
    column_lower = column_name.lower()
    
    # Handle nullable columns (10% chance of NULL for nullable columns)
    if is_nullable == 'YES' and random.random() < 0.1:
        return None
    
    safe_length = min(max_length or 50, 100) if max_length else 50
    
    # EMTS-specific field mappings
    if 'attachedrin' in column_lower and 'indicator' in column_lower:
        return random.randint(0, 1)
    
    elif 'transactionpartnerorgid' in column_lower:
        return random.randint(1000, 9999)
    
    elif 'transfercomplianceyear' in column_lower:
        return random.randint(2020, 2025)
    
    elif 'transfermonth' in column_lower:
        return random.randint(1, 12)
    
    elif 'transferquarter' in column_lower:
        return random.randint(1, 4)
    
    elif 'billoflading' in column_lower:
        if 'tradingpartner' in column_lower:
            return f"TPBOL{random.randint(100000, 999999)}"[:safe_length]
        else:
            return f"BOL{random.randint(100000, 999999)}"[:safe_length]
    
    elif 'businesspartnerdesc' in column_lower:
        return f"Business Partner {random.randint(1, 100)}"[:safe_length]
    
    elif 'emtsaction' in column_lower and 'outbound' in column_lower:
        return random.choice(EMTSDataPools.emts_actions)
    
    elif 'emtsassignmentcode' in column_lower:
        if 'desc' in column_lower:
            idx = random.randint(0, len(EMTSDataPools.assignment_codes) - 1)
            return EMTSDataPools.assignment_descriptions[idx][:safe_length]
        else:
            return random.choice(EMTSDataPools.assignment_codes)
    
    elif 'emtsbatchnumber' in column_lower:
        return f"BATCH{random.randint(10000, 99999)}"[:safe_length]
    
    elif 'emtsbuyorsellreasoncode' in column_lower:
        if 'desc' in column_lower:
            return random.choice(EMTSDataPools.buy_sell_reason_descriptions)[:safe_length]
        else:
            return random.choice(EMTSDataPools.buy_sell_reason_codes)[:safe_length]
    
    elif 'emtstradingpartner' in column_lower:
        if 'invoice' in column_lower:
            return f"TPINV{random.randint(100000, 999999)}"[:safe_length]
        elif 'ptd' in column_lower:
            return random.choice(EMTSDataPools.emts_trading_partners)
    
    elif 'emtstransactionstatus' in column_lower:
        if 'desc' in column_lower:
            idx = random.randint(0, len(EMTSDataPools.transaction_status_codes) - 1)
            return EMTSDataPools.transaction_status_descriptions[idx][:safe_length]
        else:
            return random.choice(EMTSDataPools.transaction_status_codes)
    
    elif 'emtstransactiontype' in column_lower:
        if 'desc' in column_lower:
            idx = random.randint(0, len(EMTSDataPools.transaction_type_codes) - 1)
            return EMTSDataPools.transaction_type_descriptions[idx][:safe_length]
        else:
            return random.choice(EMTSDataPools.transaction_type_codes)
    
    elif 'emtstransmission' in column_lower:
        if 'error' in column_lower:
            return random.choice(EMTSDataPools.transmission_errors)
        elif 'status' in column_lower:
            return random.choice(EMTSDataPools.transmission_statuses)
    
    elif 'exttransactionnumber' in column_lower:
        if 'matched' in column_lower:
            return f"MEXT{random.randint(1000000, 9999999)}"[:safe_length]
        else:
            return f"EXT{random.randint(1000000, 9999999)}"[:safe_length]
    
    elif 'ptdnumber' in column_lower:
        return f"PTD{random.randint(100000, 999999)}"[:safe_length]
    
    elif 'comments' in column_lower:
        return random.choice(EMTSDataPools.comments)[:safe_length]
    
    elif 'guid' in column_lower:
        return str(uuid.uuid4())[:safe_length]
    
    elif 'fuelcode' in column_lower:
        if 'desc' in column_lower:
            idx = random.randint(0, len(EMTSDataPools.fuel_codes) - 1)
            return EMTSDataPools.fuel_descriptions[idx][:safe_length]
        else:
            return random.choice(EMTSDataPools.fuel_codes)
    
    elif 'fuelquantity' in column_lower:
        return round(random.uniform(1000.0, 50000.0), 2)
    
    elif 'fuelunitofmeasurement' in column_lower:
        return random.choice(EMTSDataPools.units_of_measurement)[:safe_length]
    
    elif 'price' in column_lower:
        if 'gallon' in column_lower:
            if 'amount' in column_lower:
                return round(random.uniform(2.50, 6.00), 2)
            elif 'currency' in column_lower:
                return random.choice(EMTSDataPools.currencies)[:safe_length]
        elif 'rin' in column_lower:
            if 'amount' in column_lower:
                return round(random.uniform(0.50, 2.00), 2)
            elif 'currency' in column_lower:
                return random.choice(EMTSDataPools.currencies)[:safe_length]
    
    elif 'facilityid' in column_lower:
        return random.choice(EMTSDataPools.facility_ids)[:safe_length]
    
    elif 'organizationid' in column_lower:
        return random.choice(EMTSDataPools.organization_ids)[:safe_length]
    
    elif column_lower == 'impact':
        return random.choice(EMTSDataPools.impacts)
    elif 'impactdesc' in column_lower:
        return random.choice(EMTSDataPools.impact_descriptions)[:safe_length]
    
    elif 'objectcategory' in column_lower:
        if 'desc' in column_lower:
            idx = random.randint(0, len(EMTSDataPools.object_categories) - 1)
            return EMTSDataPools.object_category_descriptions[idx][:safe_length]
        else:
            return random.choice(EMTSDataPools.object_categories)[:safe_length]
    
    elif 'objecttype' in column_lower:
        if 'desc' in column_lower:
            return random.choice(['Transaction Record', 'Batch Record', 'Transfer Record'])[:safe_length]
        else:
            return random.choice(['TRANSACTION', 'BATCH', 'TRANSFER'])[:safe_length]
    
    elif 'objectstatusdesc' in column_lower:
        return random.choice(['Active', 'Pending', 'Complete', 'Cancelled'])[:safe_length]
    
    elif 'processingstatus' in column_lower:
        if 'desc' in column_lower:
            return random.choice(EMTSDataPools.processing_status_descriptions)[:safe_length]
        else:
            return random.choice(EMTSDataPools.processing_statuses)[:safe_length]
    
    elif 'qapcode' in column_lower:
        if 'desc' in column_lower:
            return random.choice(['10-Q-RIN', '30-Unverified'])[:safe_length]
        else:
            return random.choice([10, 30])
    
    elif 'reasoncodedesc' in column_lower:
        return random.choice(['Normal Processing', 'Expedited Request', 'Correction', 'Resubmission'])[:safe_length]
    
    elif 'regulation' in column_lower:
        if 'category' in column_lower:
            if 'desc' in column_lower:
                return random.choice(['Renewable Fuel Standard 2', 'Low Carbon Fuel Standard', 'Other Regulation'])[:safe_length]
            else:
                return random.choice(['RFS2', 'LCFS', 'OTHER'])[:safe_length]
        elif 'quantity' in column_lower:
            return round(random.uniform(500.0, 25000.0), 2)
        elif 'type' in column_lower:
            if 'desc' in column_lower:
                return random.choice(['Federal Regulation', 'State Regulation', 'Local Regulation'])[:safe_length]
            else:
                return random.choice(['FEDERAL', 'STATE', 'LOCAL'])[:safe_length]
        elif 'unitofmeasurement' in column_lower:
            return random.choice(EMTSDataPools.units_of_measurement)[:safe_length]
    
    elif 'rinyear' in column_lower:
        return str(random.randint(2020, 2025))[:safe_length]
    
    elif 'subobjectscenario' in column_lower:
        scenario = random.choice(['RFS2_RF_CCO', 'RFS2_OTC_CCO', 'RFS2_MADJ_CCO'])
        return scenario[:safe_length]
    
    elif 'submission' in column_lower:
        if 'complianceyear' in column_lower:
            return str(random.randint(2020, 2025))[:safe_length]
        elif 'identifier' in column_lower:
            return f"SUB{random.randint(1000000, 9999999)}"[:safe_length]
        elif 'month' in column_lower:
            return f"{random.randint(1, 12):02d}"[:safe_length]
        elif 'quarter' in column_lower:
            return f"{random.randint(1, 4):02d}"[:safe_length]
        elif 'date' in column_lower:
            base_date = datetime(2023, 1, 1)
            random_days = random.randint(0, 700)
            return (base_date + timedelta(days=random_days + random.randint(-30, 30))).strftime('%Y-%m-%d')
    
    elif 'tradesource' in column_lower:
        return random.choice(['Initiated', 'Received'])[:safe_length]
    
    elif 'transactioncategory' in column_lower:
        trans_cat = random.choice(['SAL', 'PUR'])
        return trans_cat[:safe_length]
    
    elif 'transactiontype' in column_lower and 'emts' not in column_lower:
        trans_type = random.choice(['EXP', 'IMP', 'MSL', 'PRD', 'PUR', 'REG', 'SAL', 'TRN'])
        return trans_type[:safe_length]
    
    # Date fields
    elif any(date_field in column_lower for date_field in ['emtsexpirationdate', 'transferdate']):
        base_date = datetime(2023, 1, 1)
        random_days = random.randint(0, 700)
        if 'expiration' in column_lower:
            return (base_date + timedelta(days=random_days + random.randint(365, 730))).strftime('%Y-%m-%d')
        else:
            return (base_date + timedelta(days=random_days + random.randint(-10, 10))).strftime('%Y-%m-%d')
    
    # Fall back to None if no specific EMTS pattern matches
    return None

def generate_random_value_for_column(column_name, data_type, is_nullable, max_length, default_value=None, table_name=None):
    """Enhanced random value generation with EMTS-specific support"""
   
    try:
        logger.debug(f"Generating value for column: {column_name}, type: {data_type}, nullable: {is_nullable}, max_length: {max_length}")
        
        # First, try EMTS-specific generation if it's an EMTS table
        if table_name and is_emts_table(table_name):
            emts_value = generate_emts_specific_value(column_name, data_type, is_nullable, max_length)
            if emts_value is not None:
                logger.debug(f"Generated EMTS-specific value for {column_name}: {emts_value}")
                return emts_value
        
        # Handle nullable columns (10% chance of NULL for nullable columns)
        if is_nullable == 'YES' and random.random() < 0.1:
            return None
       
        # Use default value if available for certain cases
        if default_value and random.random() < 0.3:  # 30% chance to use default
            return default_value
       
        column_lower = column_name.lower()
       
        # Determine safe max length for string values
        safe_length = min(max_length or 50, 100) if max_length else 50
       
        # String/Text types - Use your existing logic
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

# Keep all your existing helper functions (generate_string_value, generate_code_value, etc.)
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

@mcp.tool(name="generate_and_insert_random_entry", description="Generate and insert random entries into a specified SAP HANA table with EMTS support")
async def generate_and_insert_random_entry(
    table: str = Query(..., description="SAP HANA table name"),
    count: int = Query(1, description="Number of random rows to insert (default 1)"),
    force_emts: bool = Query(False, description="Force EMTS-specific data generation even for non-EMTS tables")
):  
    """
    Enhanced version with EMTS-specific data generation and better error handling.
    """
   
    logger.info(f"Starting random data generation for table: {table}, count: {count}, force_emts: {force_emts}")
   
    # Get the actual schema for this table
    table_schema = get_table_schema(table)
   
    if not table_schema:
        error_msg = f"Could not retrieve schema for table '{table}'. Please check if the table exists in schema '{HANA_SCHEMA}'."
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
   
    logger.info(f"Found {len(table_schema)} columns in table {table}")
    
    # Determine if we should use EMTS-specific generation
    use_emts_generation = force_emts or is_emts_table(table)
    logger.info(f"Using EMTS-specific generation: {use_emts_generation}")
   
    def make_random_row():
        random_data = {}
       
        try:
            for column_info in table_schema:
                column_name, data_type, is_nullable, max_length = column_info[:4]
                default_value = column_info[4] if len(column_info) > 4 else None
               
                value = generate_random_value_for_column(
                    column_name, 
                    data_type, 
                    is_nullable, 
                    max_length, 
                    default_value,
                    table if use_emts_generation else None
                )
                random_data[column_name] = value
                logger.debug(f"Generated value for {column_name}: {value}")
               
        except Exception as e:
            error_msg = f"Error generating random data for column: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
 
        return random_data
 
    # Generate the specified number of rows
    try:
        rows = []
        for i in range(count):
            row = make_random_row()
            rows.append(row)
            logger.debug(f"Generated row {i+1}: {row}")
           
    except Exception as e:
        error_msg = f"Error generating random rows: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
   
    if not rows:
        error_msg = "No data generated"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
   
    # Validate that we have data before inserting
    sample_row = rows[0]
    if not sample_row:
        error_msg = "Generated row is empty"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
   
    logger.info(f"Sample generated row: {sample_row}")
   
    columns = list(sample_row.keys())
    col_clause = ', '.join(f'"{col}"' for col in columns)
    val_clause = ', '.join(['?' for _ in columns])
    sql = f'INSERT INTO "{HANA_SCHEMA}"."{table}" ({col_clause}) VALUES ({val_clause})'
   
    logger.debug(f"Insert SQL: {sql}")
 
    # Handle batch insertion for large datasets
    max_params = 32767
    batch_size = max_params // len(columns) if len(columns) > 0 else 1000
    total_inserted = 0
 
    cursor = conn.cursor()
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            values_list = [list(row.values()) for row in batch]
           
            logger.debug(f"Inserting batch {i//batch_size + 1}, size: {len(batch)}")
           
            cursor.executemany(sql, values_list)
            total_inserted += len(batch)
           
        conn.commit()
        logger.info(f"Successfully inserted {total_inserted} rows into {table}")
       
    except dbapi.Error as e:
        conn.rollback()
        error_msg = f"Database insert error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    finally:
        cursor.close()
 
    return {
        "object": "random_insert_result",
        "table": table,
        "message": f"✅ Successfully inserted {total_inserted} random records into '{table}'" + (f" (EMTS mode)" if use_emts_generation else ""),
        "columns_inserted": len(columns),
        "total_rows": total_inserted,
        "sample_data": sample_row,
        "schema_columns": [col[0] for col in table_schema],
        "emts_mode": use_emts_generation
    }

@mcp.tool(name="debug_table_schema", description="Debug tool to inspect table schema and generate sample data without inserting")
async def debug_table_schema(
    table: str = Query(..., description="SAP HANA table name"),
    force_emts: bool = Query(False, description="Force EMTS-specific data generation for testing")
):
    """
    Debug tool to inspect the table schema and test data generation without actually inserting.
    """
   
    logger.info(f"Debugging schema for table: {table}")
   
    # Get the actual schema for this table
    table_schema = get_table_schema(table)
   
    if not table_schema:
        return {
            "error": f"Could not retrieve schema for table '{table}'",
            "schema": None,
            "sample_data": None
        }
   
    # Determine if we should use EMTS-specific generation
    use_emts_generation = force_emts or is_emts_table(table)
   
    # Generate one sample row
    sample_data = {}
    column_details = []
   
    for column_info in table_schema:
        column_name, data_type, is_nullable, max_length = column_info[:4]
        default_value = column_info[4] if len(column_info) > 4 else None
       
        try:
            value = generate_random_value_for_column(
                column_name, 
                data_type, 
                is_nullable, 
                max_length, 
                default_value,
                table if use_emts_generation else None
            )
            sample_data[column_name] = value
        except Exception as e:
            sample_data[column_name] = f"ERROR: {str(e)}"
       
        column_details.append({
            "name": column_name,
            "type": data_type,
            "nullable": is_nullable,
            "max_length": max_length,
            "default_value": default_value,
            "generated_value": sample_data[column_name]
        })
   
    return {
        "object": "debug_result",
        "table": table,
        "schema_found": len(table_schema) > 0,
        "column_count": len(table_schema),
        "columns": column_details,
        "sample_data": sample_data,
        "emts_mode": use_emts_generation,
        "emts_table_detected": is_emts_table(table)
    }

@mcp.tool(name="generate_emts_specific_data", description="Generate EMTS-specific test data for any table (legacy EMTS mode)")
async def generate_emts_specific_data(
    table: str = Query(..., description="SAP HANA table name"),
    count: int = Query(1, description="Number of random rows to insert (default 1)")
):
    """
    Legacy EMTS-specific data generation - forces EMTS mode regardless of table name.
    This is equivalent to your original EMTS-specific function.
    """
    return await generate_and_insert_random_entry(table=table, count=count, force_emts=True)

# Running the server
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')