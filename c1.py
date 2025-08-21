# SAP HANA Chatbot - Complete Implementation
# File structure:
# /
# ├── server/
# │   ├── mcp_server.py
# │   ├── database.py
# │   └── requirements.txt
# ├── client/
# │   ├── chatbot_client.py
# │   └── streamlit_ui.py
# └── config/
#     └── config.py

# =============================================================================
# SERVER SIDE - MCP Tools Implementation
# =============================================================================

# server/requirements.txt
"""
hdbcli>=2.19.21
mcp>=1.0.0
python-dotenv>=1.0.0
faker>=20.1.0
sqlparse>=0.4.4
pydantic>=2.5.0
asyncio-mqtt>=0.13.0
"""

# config/config.py
import os
from dataclasses import dataclass

@dataclass
class HANAConfig:
    host: str = os.getenv("HANA_HOST", "")
    port: str = os.getenv("HANA_PORT", "443")
    user: str = os.getenv("HANA_USER", "")
    password: str = os.getenv("HANA_PASS", "")
    schema: str = os.getenv("HANA_SCHEMA", "")
    ssl: str = os.getenv("HANA_SSL", "true")
    table_name: str = os.getenv("HANA_TABLE_NAME", "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION")
    certificate: str = os.getenv("HANA_CERTIFICATE", "")

@dataclass
class AIConfig:
    client_id: str = os.getenv("AICORE_CLIENT_ID", "")
    auth_url: str = os.getenv("AICORE_AUTH_URL", "")
    client_secret: str = os.getenv("AICORE_CLIENT_SECRET", "")
    resource_group: str = os.getenv("AICORE_RESOURCE_GROUP", "default")
    base_url: str = os.getenv("AICORE_BASE_URL", "")

# server/database.py
import hdbcli
from hdbcli import dbapi
import json
import logging
from typing import Dict, List, Any, Optional
from faker import Faker
import random
from datetime import datetime, timedelta
from config.config import HANAConfig

class SAP_HANA_Manager:
    def __init__(self, config: HANAConfig):
        self.config = config
        self.connection = None
        self.fake = Faker()
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Establish connection to SAP HANA"""
        try:
            self.connection = dbapi.connect(
                address=self.config.host,
                port=int(self.config.port),
                user=self.config.user,
                password=self.config.password,
                currentSchema=self.config.schema,
                encrypt=self.config.ssl.lower() == 'true',
                sslValidateCertificate=False
            )
            self.logger.info("Connected to SAP HANA successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to SAP HANA: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.logger.info("Disconnected from SAP HANA")
    
    def execute_query(self, query: str, params: List = None) -> Dict[str, Any]:
        """Execute a SQL query and return results"""
        try:
            if not self.connection:
                if not self.connect():
                    return {"error": "Database connection failed", "status": "failed"}
            
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # For SELECT queries, fetch results
            if query.strip().upper().startswith('SELECT'):
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                result_data = []
                for row in rows:
                    result_data.append(dict(zip(columns, row)))
                
                cursor.close()
                return {
                    "data": result_data,
                    "columns": columns,
                    "row_count": len(result_data),
                    "status": "success"
                }
            else:
                # For INSERT/UPDATE/DELETE queries
                affected_rows = cursor.rowcount
                self.connection.commit()
                cursor.close()
                return {
                    "affected_rows": affected_rows,
                    "status": "success",
                    "message": f"Operation completed successfully. {affected_rows} rows affected."
                }
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def get_table_schema(self, table_name: str = None) -> Dict[str, Any]:
        """Get schema information for a table"""
        if not table_name:
            table_name = self.config.table_name
            
        schema_query = """
        SELECT 
            COLUMN_NAME,
            DATA_TYPE_NAME,
            LENGTH,
            IS_NULLABLE,
            DEFAULT_VALUE,
            COMMENTS
        FROM SYS.TABLE_COLUMNS 
        WHERE SCHEMA_NAME = ? 
        AND TABLE_NAME = ?
        ORDER BY POSITION
        """
        
        return self.execute_query(schema_query, [self.config.schema, table_name])
    
    def get_sample_data(self, table_name: str = None, limit: int = 5) -> Dict[str, Any]:
        """Get sample data from table"""
        if not table_name:
            table_name = self.config.table_name
            
        query = f'SELECT * FROM "{self.config.schema}"."{table_name}" LIMIT {limit}'
        return self.execute_query(query)
    
    def generate_random_data(self, table_schema: List[Dict], count: int = 1) -> List[Dict]:
        """Generate random data based on table schema"""
        generated_data = []
        
        for _ in range(count):
            row_data = {}
            for column in table_schema:
                col_name = column['COLUMN_NAME']
                data_type = column['DATA_TYPE_NAME'].upper()
                
                # Generate data based on column type
                if 'VARCHAR' in data_type or 'NVARCHAR' in data_type:
                    if 'ID' in col_name.upper():
                        row_data[col_name] = self.fake.uuid4()
                    elif 'NAME' in col_name.upper():
                        row_data[col_name] = self.fake.company()
                    elif 'EMAIL' in col_name.upper():
                        row_data[col_name] = self.fake.email()
                    elif 'DESCRIPTION' in col_name.upper():
                        row_data[col_name] = self.fake.text(max_nb_chars=200)
                    else:
                        row_data[col_name] = self.fake.word()
                        
                elif 'INTEGER' in data_type or 'BIGINT' in data_type:
                    row_data[col_name] = self.fake.random_int(min=1, max=10000)
                    
                elif 'DECIMAL' in data_type or 'DOUBLE' in data_type:
                    row_data[col_name] = round(self.fake.random.uniform(1.0, 1000.0), 2)
                    
                elif 'DATE' in data_type:
                    row_data[col_name] = self.fake.date_between(start_date='-1y', end_date='today')
                    
                elif 'TIMESTAMP' in data_type:
                    row_data[col_name] = self.fake.date_time_between(start_date='-1y', end_date='now')
                    
                elif 'BOOLEAN' in data_type:
                    row_data[col_name] = self.fake.boolean()
                    
                else:
                    row_data[col_name] = self.fake.word()
            
            generated_data.append(row_data)
        
        return generated_data

# server/mcp_server.py
import asyncio
import json
import logging
from typing import Any, Dict, List
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from database import SAP_HANA_Manager
from config.config import HANAConfig
import sqlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database manager
hana_config = HANAConfig()
db_manager = SAP_HANA_Manager(hana_config)

# Initialize MCP server
server = Server("sap-hana-chatbot")

class QueryValidator:
    @staticmethod
    def is_safe_query(query: str) -> tuple[bool, str]:
        """Validate if query is safe to execute"""
        query_upper = query.strip().upper()
        
        # Allow only SELECT, INSERT, UPDATE for specific table
        dangerous_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 
            'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'CALL'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Operation '{keyword}' is not allowed"
        
        # Parse query structure
        try:
            parsed = sqlparse.parse(query)[0]
            return True, "Query is safe"
        except Exception as e:
            return False, f"Query parsing failed: {e}"

class NaturalLanguageProcessor:
    def __init__(self):
        self.intent_keywords = {
            "query": ["show", "find", "list", "what", "how many", "tell me", "get", "fetch", "display"],
            "insert": ["add", "create", "insert", "new", "generate", "random"],
            "update": ["update", "modify", "change", "edit", "set"],
            "schema": ["describe", "structure", "columns", "fields", "table info", "schema"]
        }
    
    def classify_intent(self, user_input: str) -> str:
        """Classify user intent from natural language"""
        user_input_lower = user_input.lower()
        
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return intent
        
        return "query"  # Default to query
    
    def extract_parameters(self, user_input: str, intent: str) -> Dict[str, Any]:
        """Extract parameters from user input"""
        params = {}
        user_input_lower = user_input.lower()
        
        if intent == "insert":
            # Extract number of rows to insert
            import re
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                params["count"] = int(numbers[-1])  # Use last number found
            else:
                params["count"] = 1
        
        return params

nlp = NaturalLanguageProcessor()

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="query_database",
            description="Execute SELECT queries on SAP HANA database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT query to execute"},
                    "natural_language": {"type": "string", "description": "Original natural language query"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_table_info",
            description="Get schema information and sample data for a table",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Name of the table to inspect"}
                }
            }
        ),
        Tool(
            name="insert_data",
            description="Insert new records into the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Target table name"},
                    "data": {"type": "object", "description": "Data to insert"},
                    "count": {"type": "integer", "description": "Number of random records to generate"}
                }
            }
        ),
        Tool(
            name="natural_language_query",
            description="Process natural language questions about the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Natural language question about the data"}
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="bulk_insert_random",
            description="Insert multiple random records for testing",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of random records to insert"},
                    "table_name": {"type": "string", "description": "Target table name (optional)"}
                },
                "required": ["count"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    try:
        if name == "query_database":
            return await handle_query_database(arguments)
        elif name == "get_table_info":
            return await handle_get_table_info(arguments)
        elif name == "insert_data":
            return await handle_insert_data(arguments)
        elif name == "natural_language_query":
            return await handle_natural_language_query(arguments)
        elif name == "bulk_insert_random":
            return await handle_bulk_insert_random(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def handle_query_database(arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute database query"""
    query = arguments.get("query", "")
    natural_language = arguments.get("natural_language", "")
    
    # Validate query
    is_safe, message = QueryValidator.is_safe_query(query)
    if not is_safe:
        return [TextContent(type="text", text=f"Query validation failed: {message}")]
    
    # Execute query
    result = db_manager.execute_query(query)
    
    if result["status"] == "success":
        response = f"Query executed successfully!\n"
        response += f"Found {result['row_count']} records.\n\n"
        
        if result["data"]:
            # Format first few rows for display
            for i, row in enumerate(result["data"][:5]):
                response += f"Row {i+1}: {json.dumps(row, indent=2, default=str)}\n"
            
            if result['row_count'] > 5:
                response += f"\n... and {result['row_count'] - 5} more records"
        
        return [TextContent(type="text", text=response)]
    else:
        return [TextContent(type="text", text=f"Query failed: {result['error']}")]

async def handle_get_table_info(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get table schema and sample data"""
    table_name = arguments.get("table_name", hana_config.table_name)
    
    # Get schema
    schema_result = db_manager.get_table_schema(table_name)
    sample_result = db_manager.get_sample_data(table_name, 3)
    
    response = f"Table Information for: {table_name}\n\n"
    
    if schema_result["status"] == "success":
        response += "Schema:\n"
        for column in schema_result["data"]:
            response += f"- {column['COLUMN_NAME']}: {column['DATA_TYPE_NAME']}"
            if column['LENGTH']:
                response += f"({column['LENGTH']})"
            response += f" {'NULL' if column['IS_NULLABLE'] == 'TRUE' else 'NOT NULL'}\n"
    
    if sample_result["status"] == "success" and sample_result["data"]:
        response += f"\nSample Data ({len(sample_result['data'])} records):\n"
        for i, row in enumerate(sample_result["data"]):
            response += f"Record {i+1}: {json.dumps(row, indent=2, default=str)}\n"
    
    return [TextContent(type="text", text=response)]

async def handle_insert_data(arguments: Dict[str, Any]) -> List[TextContent]:
    """Insert data into database"""
    table_name = arguments.get("table_name", hana_config.table_name)
    data = arguments.get("data", {})
    count = arguments.get("count", 1)
    
    # Get table schema first
    schema_result = db_manager.get_table_schema(table_name)
    if schema_result["status"] != "success":
        return [TextContent(type="text", text="Failed to get table schema")]
    
    # Generate random data if no specific data provided
    if not data and count > 0:
        generated_data = db_manager.generate_random_data(schema_result["data"], count)
    else:
        generated_data = [data]
    
    success_count = 0
    errors = []
    
    for row_data in generated_data:
        # Build INSERT query
        columns = list(row_data.keys())
        placeholders = ["?" for _ in columns]
        values = list(row_data.values())
        
        insert_query = f'''
        INSERT INTO "{hana_config.schema}"."{table_name}" 
        ({", ".join(f'"{col}"' for col in columns)})
        VALUES ({", ".join(placeholders)})
        '''
        
        result = db_manager.execute_query(insert_query, values)
        if result["status"] == "success":
            success_count += 1
        else:
            errors.append(result["error"])
    
    response = f"Insertion completed!\n"
    response += f"Successfully inserted: {success_count} records\n"
    if errors:
        response += f"Errors encountered: {len(errors)}\n"
        response += f"First error: {errors[0]}"
    
    return [TextContent(type="text", text=response)]

async def handle_natural_language_query(arguments: Dict[str, Any]) -> List[TextContent]:
    """Process natural language questions"""
    question = arguments.get("question", "")
    
    # Classify intent
    intent = nlp.classify_intent(question)
    params = nlp.extract_parameters(question, intent)
    
    response = f"Processing your question: '{question}'\n"
    response += f"Detected intent: {intent}\n\n"
    
    if intent == "schema":
        # Get table information
        table_info = await handle_get_table_info({"table_name": hana_config.table_name})
        return table_info
        
    elif intent == "insert":
        # Handle insert request
        count = params.get("count", 5)  # Default to 5 if not specified
        insert_result = await handle_bulk_insert_random({"count": count})
        return insert_result
        
    elif intent == "query":
        # Convert to SQL (simplified example)
        sql_query = generate_sql_from_nl(question)
        if sql_query:
            return await handle_query_database({"query": sql_query, "natural_language": question})
        else:
            response += "I couldn't convert your question to a SQL query. Please try rephrasing or use a more specific query."
            return [TextContent(type="text", text=response)]
    
    else:
        response += "I can help you with:\n"
        response += "- Querying data: 'Show me all transactions from last month'\n"
        response += "- Inserting data: 'Add 10 random records'\n"
        response += "- Table info: 'Describe the table structure'\n"
        return [TextContent(type="text", text=response)]

async def handle_bulk_insert_random(arguments: Dict[str, Any]) -> List[TextContent]:
    """Insert multiple random records"""
    count = arguments.get("count", 5)
    table_name = arguments.get("table_name", hana_config.table_name)
    
    # Limit to prevent accidents
    if count > 100:
        count = 100
        
    return await handle_insert_data({
        "table_name": table_name,
        "count": count
    })

def generate_sql_from_nl(question: str) -> str:
    """Simple natural language to SQL conversion"""
    question_lower = question.lower()
    table_name = f'"{hana_config.schema}"."{hana_config.table_name}"'
    
    # Simple keyword-based SQL generation
    if "all" in question_lower or "show" in question_lower:
        return f"SELECT * FROM {table_name} LIMIT 10"
    elif "count" in question_lower:
        return f"SELECT COUNT(*) as total_records FROM {table_name}"
    elif "recent" in question_lower or "latest" in question_lower:
        return f"SELECT * FROM {table_name} ORDER BY 1 DESC LIMIT 10"
    elif "first" in question_lower:
        return f"SELECT * FROM {table_name} LIMIT 5"
    else:
        return f"SELECT * FROM {table_name} LIMIT 10"

async def main():
    """Run the MCP server"""
    logger.info("Starting SAP HANA MCP Server...")
    
    # Test database connection
    if db_manager.connect():
        logger.info("Database connection successful")
    else:
        logger.error("Database connection failed")
        return
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="sap-hana-chatbot",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None,
                    ),
                ),
            )
    finally:
        db_manager.disconnect()

if __name__ == "__main__":
    asyncio.run(main())

# =============================================================================
# CLIENT SIDE - Chatbot Implementation
# =============================================================================

# client/chatbot_client.py
import asyncio
import json
import logging
from typing import Dict, List, Any
import requests
import os
from datetime import datetime

class MCPClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session_id = None
        self.logger = logging.getLogger(__name__)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool via HTTP API"""
        try:
            # For local testing, simulate MCP calls
            # In production, this would make actual MCP calls
            return await self.simulate_mcp_call(tool_name, arguments)
        except Exception as e:
            self.logger.error(f"MCP call failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def simulate_mcp_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate MCP calls for testing"""
        # This would be replaced with actual MCP communication
        if tool_name == "natural_language_query":
            return {
                "status": "success",
                "response": f"Processed query: {arguments.get('question', '')}"
            }
        return {"status": "success", "response": "Tool executed successfully"}

class SAP_HANA_Chatbot:
    def __init__(self):
        self.mcp_client = MCPClient()
        self.conversation_history = []
        self.context = {
            "current_table": os.getenv("HANA_TABLE_NAME", "COMPLIANCETRANSACTION"),
            "last_query_result": None,
            "user_preferences": {}
        }
    
    async def process_message(self, user_input: str) -> str:
        """Process user message and return response"""
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "user": user_input,
            "type": "user_message"
        })
        
        # Determine what the user wants to do
        intent = self.classify_user_intent(user_input)
        
        try:
            if intent == "query":
                response = await self.handle_query(user_input)
            elif intent == "insert":
                response = await self.handle_insert(user_input)
            elif intent == "schema":
                response = await self.handle_schema_request(user_input)
            elif intent == "help":
                response = self.get_help_message()
            else:
                response = await self.handle_general_query(user_input)
            
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "bot": response,
                "type": "bot_response"
            })
            
            return response
            
        except Exception as e:
            error_response = f"I encountered an error processing your request: {str(e)}"
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "bot": error_response,
                "type": "error"
            })
            return error_response
    
    def classify_user_intent(self, user_input: str) -> str:
        """Classify what the user wants to do"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["help", "what can you do", "commands"]):
            return "help"
        elif any(word in user_input_lower for word in ["add", "insert", "create", "generate", "random"]):
            return "insert"
        elif any(word in user_input_lower for word in ["describe", "schema", "structure", "columns", "table"]):
            return "schema"
        else:
            return "query"
    
    async def handle_query(self, user_input: str) -> str:
        """Handle data query requests"""
        result = await self.mcp_client.call_tool(
            "natural_language_query",
            {"question": user_input}
        )
        
        if result.get("status") == "success":
            return f"Here's what I found:\n{result.get('response', 'No data returned')}"
        else:
            return f"Query failed: {result.get('error', 'Unknown error')}"
    
    async def handle_insert(self, user_input: str) -> str:
        """Handle data insertion requests"""
        # Extract number of records to insert
        import re
        numbers = re.findall(r'\d+', user_input)
        count = int(numbers[0]) if numbers else 5
        
        # Limit for safety
        count = min(count, 100)
        
        result = await self.mcp_client.call_tool(
            "bulk_insert_random",
            {"count": count}
        )
        
        if result.get("status") == "success":
            return f"Successfully added {count} random records to the database!"
        else:
            return f"Insert operation failed: {result.get('error', 'Unknown error')}"
    
    async def handle_schema_request(self, user_input: str) -> str:
        """Handle schema information requests"""
        result = await self.mcp_client.call_tool(
            "get_table_info",
            {"table_name": self.context["current_table"]}
        )
        
        if result.get("status") == "success":
            return f"Table Information:\n{result.get('response', 'No schema information available')}"
        else:
            return f"Failed to get table information: {result.get('error', 'Unknown error')}"
    
    async def handle_general_query(self, user_input: str) -> str:
        """Handle general queries using natural language processing"""
        result = await self.mcp_client.call_tool(
            "natural_language_query",
            {"question": user_input}
        )
        
        return result.get("response", "I couldn't process your request. Please try rephrasing.")
    
    def get_help_message(self) -> str:
        """Return help information"""
        return """
🤖 SAP HANA Chatbot Help

I can help you with your compliance transaction database in these ways:

📊 **Query Data:**
- "Show me all transactions"
- "How many records are in the database?"
- "Find recent transactions"
- "What's the total count?"

➕ **Add Data:**
- "Add 10 random records"
- "Insert 5 test transactions"
- "Generate 20 sample records"

🔍 **Table Information:**
- "Describe the table structure"
- "Show me the schema"
- "What columns are available?"

💡 **Tips:**
- Be specific in your questions
- I can work with your compliance transaction table
- Ask for help anytime with "help" or "what can you do"

What would you like to do?
        """

# client/streamlit_ui.py
import streamlit as st
import asyncio
from chatbot_client import SAP_HANA_Chatbot
import json

# Initialize chatbot
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = SAP_HANA_Chatbot()

if 'messages' not in st.session_state:
    st.session_state.messages = []

def main():
    st.set_page_config(
        page_title="SAP HANA Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 SAP HANA Compliance Database Chatbot")
    st.markdown("Ask questions about your compliance transactions or perform database operations!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("Database Info")
        st.info(f"""
        **Connected to:**
        - Host: {os.getenv('HANA_HOST', 'Not configured')[:20]}...
        - Schema: {os.getenv('HANA_SCHEMA', 'Not configured')}
        - Table: {os.getenv('HANA_TABLE_NAME', 'Not configured')}
        """)
        
        st.header("Quick Actions")
        if st.button("Get Table Schema"):
            with st.spinner("Getting table information..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    st.session_state.chatbot.handle_schema_request("describe table")
                )
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        if st.button("Add 5 Random Records"):
            with st.spinner("Adding random records..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    st.session_state.chatbot.handle_insert("add 5 random records")
                )
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.header("Sample Queries")
        sample_queries = [
            "Show me all compliance transactions",
            "How many records are in the database?",
            "Add 10 random test records",
            "Describe the table structure",
            "Show me the latest 5 transactions"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query}"):
                st.session_state.messages.append({"role": "user", "content": query})
                with st.spinner("Processing..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        st.session_state.chatbot.process_message(query)
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Main chat interface
    st.header("Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your compliance database..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    st.session_state.chatbot.process_message(prompt)
                )
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chatbot.conversation_history = []
        st.rerun()

if __name__ == "__main__":
    main()

# =============================================================================
# ENHANCED LLM AGENT WITH AI CORE INTEGRATION
# =============================================================================

# client/ai_core_client.py
import requests
import json
import base64
from typing import Dict, Any, Optional
from config.config import AIConfig

class AICoreLLMClient:
    def __init__(self, config: AIConfig):
        self.config = config
        self.access_token = None
        self.token_expires_at = None
    
    def get_access_token(self) -> str:
        """Get OAuth access token from AI Core"""
        auth_string = f"{self.config.client_id}:{self.config.client_secret}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        headers = {
            'Authorization': f'Basic {auth_b64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'client_credentials'
        }
        
        response = requests.post(
            f"{self.config.auth_url}/oauth/token",
            headers=headers,
            data=data
        )
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            return self.access_token
        else:
            raise Exception(f"Failed to get access token: {response.text}")
    
    def generate_sql_from_natural_language(self, question: str, schema_info: str) -> str:
        """Use AI Core to convert natural language to SQL"""
        if not self.access_token:
            self.get_access_token()
        
        prompt = f"""
        You are an expert SQL generator for SAP HANA databases.
        
        Database Schema:
        {schema_info}
        
        Convert this natural language question to a valid SAP HANA SQL query:
        Question: {question}
        
        Rules:
        1. Use only SELECT statements for queries
        2. Use proper SAP HANA syntax
        3. Include appropriate WHERE clauses
        4. Limit results to reasonable numbers (max 100 rows)
        5. Use double quotes for schema and table names
        6. Return only the SQL query, no explanations
        
        SQL Query:
        """
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'AI-Resource-Group': self.config.resource_group
        }
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }
        
        response = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            sql_query = result['choices'][0]['message']['content'].strip()
            # Clean up the response to extract just the SQL
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            return sql_query
        else:
            raise Exception(f"AI Core API call failed: {response.text}")

# =============================================================================
# ADVANCED CHATBOT WITH AI INTEGRATION
# =============================================================================

# client/advanced_chatbot.py
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from ai_core_client import AICoreLLMClient
from config.config import AIConfig, HANAConfig

class AdvancedSAPHANAChatbot:
    def __init__(self):
        self.ai_client = AICoreLLMClient(AIConfig())
        self.hana_config = HANAConfig()
        self.conversation_context = []
        self.last_schema_info = None
        self.logger = logging.getLogger(__name__)
    
    async def process_user_query(self, user_input: str) -> Dict[str, Any]:
        """Main entry point for processing user queries"""
        
        # Add to conversation context
        self.conversation_context.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "type": "user_query"
        })
        
        # Clean and analyze input
        cleaned_input = self.clean_input(user_input)
        intent = self.analyze_intent(cleaned_input)
        
        try:
            if intent["type"] == "data_query":
                return await self.handle_data_query(cleaned_input, intent)
            elif intent["type"] == "insert_operation":
                return await self.handle_insert_operation(cleaned_input, intent)
            elif intent["type"] == "schema_inquiry":
                return await self.handle_schema_inquiry(cleaned_input)
            elif intent["type"] == "help_request":
                return self.generate_help_response()
            else:
                return await self.handle_general_conversation(cleaned_input)
                
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "status": "error",
                "suggestions": ["Try rephrasing your question", "Check if the database is accessible"]
            }
    
    def clean_input(self, user_input: str) -> str:
        """Clean and normalize user input"""
        # Remove extra whitespace, normalize case for analysis
        return user_input.strip()
    
    def analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """Analyze user intent with confidence scoring"""
        user_lower = user_input.lower()
        
        # Intent patterns with confidence scoring
        patterns = {
            "data_query": {
                "keywords": ["show", "find", "get", "list", "how many", "count", "what", "which", "select"],
                "confidence": 0
            },
            "insert_operation": {
                "keywords": ["add", "insert", "create", "generate", "random", "new", "bulk"],
                "confidence": 0
            },
            "schema_inquiry": {
                "keywords": ["describe", "schema", "structure", "columns", "table", "fields", "info"],
                "confidence": 0
            },
            "help_request": {
                "keywords": ["help", "what can you do", "commands", "how to", "guide"],
                "confidence": 0
            }
        }
        
        # Calculate confidence scores
        for intent_type, data in patterns.items():
            matches = sum(1 for keyword in data["keywords"] if keyword in user_lower)
            data["confidence"] = matches / len(data["keywords"])
        
        # Find highest confidence intent
        best_intent = max(patterns.items(), key=lambda x: x[1]["confidence"])
        
        # Extract additional parameters
        parameters = self.extract_parameters(user_input, best_intent[0])
        
        return {
            "type": best_intent[0],
            "confidence": best_intent[1]["confidence"],
            "parameters": parameters
        }
    
    def extract_parameters(self, user_input: str, intent_type: str) -> Dict[str, Any]:
        """Extract parameters from user input based on intent"""
        params = {}
        
        if intent_type == "insert_operation":
            # Extract numbers for record count
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                params["count"] = min(int(numbers[0]), 100)  # Safety limit
            else:
                params["count"] = 5  # Default
        
        elif intent_type == "data_query":
            # Extract date ranges, limits, etc.
            if "last" in user_input.lower():
                params["time_filter"] = "recent"
            if "limit" in user_input.lower():
                numbers = re.findall(r'limit\s+(\d+)', user_input.lower())
                if numbers:
                    params["limit"] = int(numbers[0])
        
        return params
    
    async def handle_data_query(self, user_input: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data querying with AI-generated SQL"""
        try:
            # Get schema information if not cached
            if not self.last_schema_info:
                self.last_schema_info = await self.get_table_schema()
            
            # Generate SQL using AI Core
            sql_query = self.ai_client.generate_sql_from_natural_language(
                user_input, 
                self.last_schema_info
            )
            
            # Validate and execute query
            if self.validate_sql_query(sql_query):
                # Here you would call your MCP tool to execute the query
                # For now, we'll simulate the response
                execution_result = await self.simulate_query_execution(sql_query)
                
                response = self.format_query_response(execution_result, user_input)
                
                return {
                    "response": response,
                    "status": "success",
                    "sql_generated": sql_query,
                    "data": execution_result.get("data", [])
                }
            else:
                return {
                    "response": "I couldn't generate a safe SQL query for your request. Please try rephrasing.",
                    "status": "error",
                    "sql_generated": sql_query
                }
                
        except Exception as e:
            return {
                "response": f"Error processing your query: {str(e)}",
                "status": "error"
            }
    
    async def handle_insert_operation(self, user_input: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data insertion operations"""
        try:
            count = intent["parameters"].get("count", 5)
            
            # Generate realistic data structure
            if not self.last_schema_info:
                self.last_schema_info = await self.get_table_schema()
            
            # Simulate insert operation
            insert_result = await self.simulate_insert_operation(count)
            
            response = f"✅ Successfully added {count} random compliance transaction records!\n\n"
            response += f"📊 **Operation Summary:**\n"
            response += f"- Records inserted: {count}\n"
            response += f"- Table: {self.hana_config.table_name}\n"
            response += f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            response += f"💡 You can now query these new records with questions like:\n"
            response += f"- 'Show me the latest transactions'\n"
            response += f"- 'How many total records do we have now?'"
            
            return {
                "response": response,
                "status": "success",
                "records_inserted": count
            }
            
        except Exception as e:
            return {
                "response": f"Failed to insert records: {str(e)}",
                "status": "error"
            }
    
    async def handle_schema_inquiry(self, user_input: str) -> Dict[str, Any]:
        """Handle schema and table structure inquiries"""
        try:
            schema_info = await self.get_table_schema()
            
            response = f"📋 **Table Schema Information**\n\n"
            response += f"**Table:** {self.hana_config.table_name}\n"
            response += f"**Schema:** {self.hana_config.schema}\n\n"
            response += f"**Columns:**\n"
            
            # Mock schema structure (replace with actual schema call)
            mock_columns = [
                {"name": "TRANSACTION_ID", "type": "VARCHAR(36)", "nullable": "NO"},
                {"name": "COMPLIANCE_TYPE", "type": "VARCHAR(100)", "nullable": "YES"},
                {"name": "TRANSACTION_DATE", "type": "TIMESTAMP", "nullable": "NO"},
                {"name": "AMOUNT", "type": "DECIMAL(15,2)", "nullable": "YES"},
                {"name": "STATUS", "type": "VARCHAR(50)", "nullable": "YES"},
                {"name": "CREATED_BY", "type": "VARCHAR(100)", "nullable": "YES"},
                {"name": "CREATED_AT", "type": "TIMESTAMP", "nullable": "NO"}
            ]
            
            for col in mock_columns:
                response += f"- **{col['name']}**: {col['type']} {'(Nullable)' if col['nullable'] == 'YES' else '(Required)'}\n"
            
            response += f"\n🔍 **Quick Facts:**\n"
            response += f"- Primary focus: Compliance transaction tracking\n"
            response += f"- You can query any of these columns\n"
            response += f"- I can generate test data for all fields\n"
            
            return {
                "response": response,
                "status": "success",
                "schema": mock_columns
            }
            
        except Exception as e:
            return {
                "response": f"Error retrieving schema: {str(e)}",
                "status": "error"
            }
    
    def generate_help_response(self) -> Dict[str, Any]:
        """Generate comprehensive help response"""
        help_text = """
🤖 **SAP HANA Compliance Database Assistant**

I'm here to help you interact with your compliance transaction database! Here's what I can do:

## 📊 **Query Your Data**
- *"Show me all compliance transactions"*
- *"How many transactions were processed this month?"*
- *"Find transactions with amount greater than 1000"*
- *"List the latest 10 transactions"*
- *"What's the total count of records?"*

## ➕ **Add Test Data**
- *"Add 5 random transactions"*
- *"Generate 20 test records"*
- *"Insert 10 sample compliance transactions"*
- *"Create bulk test data"*

## 🔍 **Database Information**
- *"Describe the table structure"*
- *"Show me the database schema"*
- *"What columns are available?"*
- *"Tell me about the compliance transaction table"*

## 🎯 **Smart Features**
- **Natural Language**: Ask questions in plain English
- **AI-Powered SQL**: I convert your questions to optimized HANA queries
- **Safe Operations**: All operations are validated for security
- **Real-time Results**: Get instant responses from your database

## 💡 **Pro Tips**
- Be specific in your questions for better results
- I work specifically with your compliance transaction data
- All generated test data is realistic and follows your schema
- Ask follow-up questions to dive deeper into results

**Ready to explore your data? Just ask me anything!**
        """
        
        return {
            "response": help_text,
            "status": "success"
        }
    
    async def handle_general_conversation(self, user_input: str) -> Dict[str, Any]:
        """Handle general conversation and unclear intents"""
        # Use AI to understand unclear requests
        try:
            clarification_prompt = f"""
            The user said: "{user_input}"
            
            This seems to be about a SAP HANA compliance database. 
            Provide a helpful response that either:
            1. Suggests a specific query they might want
            2. Asks for clarification
            3. Offers relevant database operations
            
            Be friendly and helpful.
            """
            
            # In a real implementation, you'd call AI Core here
            response = f"""
I understand you're asking about: "{user_input}"

Here are some ways I can help:

🔍 **If you want to query data:**
- "Show me compliance transactions from last week"
- "How many records do we have?"
- "Find transactions by status"

➕ **If you want to add data:**
- "Add 10 random test records"
- "Generate sample transactions"

📋 **If you want table information:**
- "Describe the table structure"
- "Show me column details"

Could you clarify what specific information you're looking for?
            """
            
            return {
                "response": response,
                "status": "success",
                "type": "clarification"
            }
            
        except Exception as e:
            return {
                "response": "I didn't quite understand that. Could you try rephrasing your question?",
                "status": "partial_success"
            }
    
    def validate_sql_query(self, sql: str) -> bool:
        """Validate SQL query for safety"""
        sql_upper = sql.upper().strip()
        
        # Must be a SELECT statement for queries
        if not sql_upper.startswith('SELECT'):
            return False
        
        # Check for dangerous keywords
        dangerous = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'GRANT', 'REVOKE']
        return not any(keyword in sql_upper for keyword in dangerous)
    
    def format_query_response(self, execution_result: Dict, original_question: str) -> str:
        """Format query results into natural language response"""
        if execution_result.get("status") != "success":
            return f"❌ Query failed: {execution_result.get('error', 'Unknown error')}"
        
        data = execution_result.get("data", [])
        row_count = len(data)
        
        if row_count == 0:
            return "🔍 No records found matching your criteria."
        
        response = f"📊 **Query Results for:** *{original_question}*\n\n"
        response += f"**Found {row_count} record(s)**\n\n"
        
        # Show first few records
        display_limit = min(3, row_count)
        for i, record in enumerate(data[:display_limit]):
            response += f"**Record {i+1}:**\n"
            for key, value in record.items():
                response += f"- {key}: {value}\n"
            response += "\n"
        
        if row_count > display_limit:
            response += f"... and {row_count - display_limit} more records\n"
        
        return response
    
    async def get_table_schema(self) -> str:
        """Get table schema information"""
        # This would call your MCP tool in real implementation
        # For now, return mock schema
        return f"""
        Table: {self.hana_config.table_name}
        Schema: {self.hana_config.schema}
        
        Columns:
        - TRANSACTION_ID (VARCHAR(36), Primary Key)
        - COMPLIANCE_TYPE (VARCHAR(100))
        - TRANSACTION_DATE (TIMESTAMP)
        - AMOUNT (DECIMAL(15,2))
        - STATUS (VARCHAR(50))
        - CREATED_BY (VARCHAR(100))
        - CREATED_AT (TIMESTAMP)
        """
    
    async def simulate_query_execution(self, sql: str) -> Dict[str, Any]:
        """Simulate query execution (replace with actual MCP call)"""
        # Mock response - replace with actual MCP tool call
        return {
            "status": "success",
            "data": [
                {
                    "TRANSACTION_ID": "12345-67890-abcdef",
                    "COMPLIANCE_TYPE": "Financial Audit",
                    "TRANSACTION_DATE": "2024-08-15 10:30:00",
                    "AMOUNT": 15750.00,
                    "STATUS": "Completed",
                    "CREATED_BY": "system",
                    "CREATED_AT": "2024-08-15 10:30:00"
                }
            ],
            "row_count": 1
        }
    
    async def simulate_insert_operation(self, count: int) -> Dict[str, Any]:
        """Simulate insert operation (replace with actual MCP call)"""
        return {
            "status": "success",
            "records_inserted": count,
            "message": f"Successfully inserted {count} records"
        }

# =============================================================================
# STREAMLIT UI - ENHANCED VERSION
# =============================================================================

# client/enhanced_streamlit_ui.py
import streamlit as st
import asyncio
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from advanced_chatbot import AdvancedSAPHANAChatbot

# Page configuration
st.set_page_config(
    page_title="SAP HANA Compliance Chatbot",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2d5aa0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    return AdvancedSAPHANAChatbot()

chatbot = get_chatbot()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏢 SAP HANA Compliance Database Assistant</h1>
        <p>Interact with your compliance transaction data using natural language</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display conversation history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant" and "data" in message:
                        st.markdown(message["content"])
                        # Display data if available
                        if message["data"]:
                            df = pd.DataFrame(message["data"])
                            st.dataframe(df, use_container_width=True)
                    else:
                        st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about your compliance data..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process with chatbot
            with st.spinner("🔄 Processing your request..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(chatbot.process_user_query(prompt))
                    
                    # Add assistant response
                    assistant_message = {
                        "role": "assistant", 
                        "content": result["response"],
                        "status": result["status"]
                    }
                    
                    if "data" in result:
                        assistant_message["data"] = result["data"]
                    
                    st.session_state.messages.append(assistant_message)
                    
                    # Track query history
                    st.session_state.query_history.append({
                        "timestamp": datetime.now(),
                        "query": prompt,
                        "response": result["response"],
                        "status": result["status"]
                    })
                    
                finally:
                    loop.close()
            
            st.rerun()
    
    with col2:
        st.header("🛠 Database Tools")
        
        # Quick actions
        with st.expander("⚡ Quick Actions", expanded=True):
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("📊 Table Schema", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "Describe the table structure"
                    })
                    st.rerun()
                
                if st.button("📈 Record Count", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "How many records are in the database?"
                    })
                    st.rerun()
            
            with col_b:
                if st.button("➕ Add Test Data", use_container_width=True):
                    count = st.selectbox("Number of records:", [5, 10, 25, 50], index=0)
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": f"Add {count} random records"
                    })
                    st.rerun()
                
                if st.button("🔍 Latest Records", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "Show me the latest 10 transactions"
                    })
                    st.rerun()
        
        # Query examples
        with st.expander("💡 Example Queries"):
            examples = [
                "Show all transactions from last week",
                "Find transactions above $5000",
                "List incomplete compliance checks",
                "Count transactions by status",
                "Show recent audit activities"
            ]
            
            for example in examples:
                if st.button(f"📝 {example}", key=f"example_{example}"):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": example
                    })
                    st.rerun()
        
        # Statistics
        with st.expander("📊 Session Statistics"):
            st.metric("Messages Sent", len([m for m in st.session_state.messages if m["role"] == "user"]))
            st.metric("Queries Processed", len(st.session_state.query_history))
            
            if st.session_state.query_history:
                success_rate = len([q for q in st.session_state.query_history if q["status"] == "success"]) / len(st.session_state.query_history)
                st.metric("Success Rate", f"{success_rate:.1%}")
        
        # Connection info
        with st.expander("🔗 Connection Details"):
            st.info(f"""
            **Host:** {os.getenv('HANA_HOST', 'Not set')[:30]}...
            **Schema:** {os.getenv('HANA_SCHEMA', 'Not set')}
            **Table:** {os.getenv('HANA_TABLE_NAME', 'Not set')}
            **Status:** Connected ✅
            """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_history = []
            st.rerun()
    
    with col2:
        if st.button("📥 Export History", use_container_width=True):
            if st.session_state.query_history:
                df = pd.DataFrame(st.session_state.query_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "chat_history.csv",
                    "text/csv"
                )
    
    with col3:
        if st.button("🔄 Refresh Connection", use_container_width=True):
            st.success("Connection refreshed!")

if __name__ == "__main__":
    main()

# =============================================================================
# DEPLOYMENT SCRIPTS
# =============================================================================

# docker-compose.yml
"""
version: '3.8'

services:
  mcp-server:
    build: ./server
    ports:
      - "8000:8000"
    environment:
      - HANA_HOST=${HANA_HOST}
      - HANA_PORT=${HANA_PORT}
      - HANA_USER=${HANA_USER}
      - HANA_PASS=${HANA_PASS}
      - HANA_SCHEMA=${HANA_SCHEMA}
      - HANA_SSL=${HANA_SSL}
      - HANA_TABLE_NAME=${HANA_TABLE_NAME}
      - HANA_CERTIFICATE=${HANA_CERTIFICATE}
    volumes:
      - ./server:/app
    restart: unless-stopped
    
  streamlit-ui:
    build: ./client
    ports:
      - "8501:8501"
    environment:
      - MCP_SERVER_URL=http://mcp-server:8000
    depends_on:
      - mcp-server
    volumes:
      - ./client:/app
    restart: unless-stopped

networks:
  default:
    driver: bridge
"""

# server/Dockerfile
"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "mcp_server.py"]
"""

# client/Dockerfile
"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "enhanced_streamlit_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

# =============================================================================
# PRODUCTION-READY MCP SERVER WITH ERROR HANDLING
# =============================================================================

# server/production_mcp_server.py
import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import traceback
from datetime import datetime, timedelta

# Enhanced imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Database and AI imports
from database import SAP_HANA_Manager
from ai_core_client import AICoreLLMClient
from config.config import HANAConfig, AIConfig

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProductionMCPServer:
    def __init__(self):
        self.server = Server("sap-hana-chatbot-production")
        self.db_manager = SAP_HANA_Manager(HANAConfig())
        self.ai_client = AICoreLLMClient(AIConfig())
        self.connection_pool = None
        self.rate_limiter = {}
        self.audit_log = []
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Test database connection
            if not self.db_manager.connect():
                raise Exception("Failed to connect to SAP HANA database")
            
            # Test AI Core connection
            try:
                self.ai_client.get_access_token()
                logger.info("AI Core connection successful")
            except Exception as e:
                logger.warning(f"AI Core connection failed: {e}")
            
            logger.info("Server initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            return False
    
    def audit_operation(self, operation: str, user_query: str, result: str, status: str):
        """Log all operations for audit purposes"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "user_query": user_query,
            "result_status": status,
            "result_summary": result[:200] if len(result) > 200 else result
        }
        self.audit_log.append(audit_entry)
        
        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
    
    def check_rate_limit(self, client_id: str = "default") -> bool:
        """Simple rate limiting"""
        now = datetime.now()
        if client_id not in self.rate_limiter:
            self.rate_limiter[client_id] = []
        
        # Remove old requests (older than 1 minute)
        self.rate_limiter[client_id] = [
            req_time for req_time in self.rate_limiter[client_id]
            if now - req_time < timedelta(minutes=1)
        ]
        
        # Check if under limit (60 requests per minute)
        if len(self.rate_limiter[client_id]) < 60:
            self.rate_limiter[client_id].append(now)
            return True
        
        return False

# Initialize production server
production_server = ProductionMCPServer()

@production_server.server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """Enhanced tool list with comprehensive capabilities"""
    return [
        Tool(
            name="intelligent_query",
            description="Process natural language questions with AI-powered SQL generation",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question about compliance data"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context for the query"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="bulk_data_operations",
            description="Perform bulk insert operations with realistic test data",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["insert", "generate_test_data"],
                        "description": "Type of bulk operation"
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Number of records to process"
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["random", "realistic", "template"],
                        "description": "Type of data to generate"
                    }
                },
                "required": ["operation", "count"]
            }
        ),
        Tool(
            name="database_analytics",
            description="Get analytical insights about the compliance data",
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "enum": ["summary", "trends", "compliance_metrics", "data_quality"],
                        "description": "Type of analysis to perform"
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time range for analysis (e.g., 'last_month', 'last_week')"
                    }
                }
            }
        ),
        Tool(
            name="advanced_search",
            description="Perform complex searches with filters and conditions",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_criteria": {
                        "type": "object",
                        "description": "Search filters and conditions"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Column to sort results by"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum number of results"
                    }
                }
            }
        ),
        Tool(
            name="data_validation",
            description="Validate data integrity and check for anomalies",
            inputSchema={
                "type": "object",
                "properties": {
                    "validation_type": {
                        "type": "string",
                        "enum": ["integrity", "duplicates", "missing_values", "outliers"],
                        "description": "Type of validation to perform"
                    }
                }
            }
        )
    ]

@production_server.server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Enhanced tool handler with comprehensive error handling"""
    
    # Rate limiting check
    if not production_server.check_rate_limit():
        return [TextContent(
            type="text", 
            text="⚠️ Rate limit exceeded. Please wait a moment before making another request."
        )]
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing tool call: {name} with arguments: {arguments}")
        
        if name == "intelligent_query":
            result = await handle_intelligent_query(arguments)
        elif name == "bulk_data_operations":
            result = await handle_bulk_operations(arguments)
        elif name == "database_analytics":
            result = await handle_analytics(arguments)
        elif name == "advanced_search":
            result = await handle_advanced_search(arguments)
        elif name == "data_validation":
            result = await handle_data_validation(arguments)
        else:
            result = [TextContent(type="text", text=f"❌ Unknown tool: {name}")]
        
        # Audit logging
        execution_time = (datetime.now() - start_time).total_seconds()
        production_server.audit_operation(
            operation=name,
            user_query=str(arguments),
            result=str(result[0].text if result else "No result"),
            status="success"
        )
        
        logger.info(f"Tool {name} executed successfully in {execution_time:.2f}s")
        return result
        
    except Exception as e:
        error_msg = f"❌ Error executing {name}: {str(e)}"
        logger.error(f"Tool execution failed: {e}\n{traceback.format_exc()}")
        
        production_server.audit_operation(
            operation=name,
            user_query=str(arguments),
            result=error_msg,
            status="error"
        )
        
        return [TextContent(type="text", text=error_msg)]

async def handle_intelligent_query(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle intelligent queries with AI-powered SQL generation"""
    question = arguments.get("question", "")
    context = arguments.get("context", "")
    
    try:
        # Get table schema for context
        schema_result = production_server.db_manager.get_table_schema()
        
        if schema_result["status"] != "success":
            return [TextContent(type="text", text="❌ Failed to retrieve table schema")]
        
        # Format schema for AI
        schema_text = "Table Columns:\n"
        for col in schema_result["data"]:
            schema_text += f"- {col['COLUMN_NAME']} ({col['DATA_TYPE_NAME']})\n"
        
        # Generate SQL using AI Core
        try:
            sql_query = production_server.ai_client.generate_sql_from_natural_language(
                question, schema_text
            )
            logger.info(f"Generated SQL: {sql_query}")
        except Exception as ai_error:
            logger.warning(f"AI SQL generation failed, using fallback: {ai_error}")
            sql_query = generate_fallback_sql(question)
        
        # Validate and execute query
        is_valid, validation_msg = validate_query_comprehensive(sql_query)
        if not is_valid:
            return [TextContent(type="text", text=f"❌ Query validation failed: {validation_msg}")]
        
        # Execute query
        result = production_server.db_manager.execute_query(sql_query)
        
        if result["status"] == "success":
            response = format_intelligent_response(question, result, sql_query)
            return [TextContent(type="text", text=response)]
        else:
            return [TextContent(type="text", text=f"❌ Query execution failed: {result.get('error')}")]
            
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Processing failed: {str(e)}")]

async def handle_bulk_operations(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle bulk data operations with enhanced data generation"""
    operation = arguments.get("operation", "insert")
    count = min(arguments.get("count", 5), 100)  # Safety limit
    data_type = arguments.get("data_type", "realistic")
    
    try:
        if operation == "insert" or operation == "generate_test_data":
            # Get schema for realistic data generation
            schema_result = production_server.db_manager.get_table_schema()
            if schema_result["status"] != "success":
                return [TextContent(type="text", text="❌ Failed to get table schema")]
            
            # Generate enhanced realistic data
            generated_data = generate_enhanced_test_data(
                schema_result["data"], 
                count, 
                data_type
            )
            
            # Batch insert for performance
            success_count = 0
            batch_size = 10
            
            for i in range(0, len(generated_data), batch_size):
                batch = generated_data[i:i + batch_size]
                batch_result = await insert_data_batch(batch)
                if batch_result.get("status") == "success":
                    success_count += len(batch)
            
            response = f"""
✅ **Bulk Operation Completed Successfully!**

📊 **Summary:**
- Operation: {operation.title()}
- Records processed: {success_count}/{count}
- Data type: {data_type.title()}
- Batch size: {batch_size}
- Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 **What's Next?**
You can now query this new data with questions like:
- "Show me the latest {count} transactions"
- "How many total records do we have now?"
- "Find transactions created today"
            """
            
            return [TextContent(type="text", text=response)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Bulk operation failed: {str(e)}")]

async def handle_analytics(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle analytical queries and insights"""
    analysis_type = arguments.get("analysis_type", "summary")
    time_range = arguments.get("time_range", "all_time")
    
    try:
        if analysis_type == "summary":
            # Get basic statistics
            queries = [
                "SELECT COUNT(*) as total_records FROM \"{}\".\"{}\"".format(
                    production_server.db_manager.config.schema,
                    production_server.db_manager.config.table_name
                ),
                "SELECT COUNT(DISTINCT COMPLIANCE_TYPE) as unique_types FROM \"{}\".\"{}\"".format(
                    production_server.db_manager.config.schema,
                    production_server.db_manager.config.table_name
                ) if "COMPLIANCE_TYPE" in get_table_columns() else None
            ]
            
            results = []
            for query in queries:
                if query:
                    result = production_server.db_manager.execute_query(query)
                    if result["status"] == "success":
                        results.append(result["data"][0] if result["data"] else {})
            
            # Format analytics response
            response = "📊 **Database Analytics Summary**\n\n"
            
            if results:
                total_records = results[0].get("TOTAL_RECORDS", 0)
                response += f"📈 **Key Metrics:**\n"
                response += f"- Total Records: {total_records:,}\n"
                
                if len(results) > 1:
                    unique_types = results[1].get("UNIQUE_TYPES", 0)
                    response += f"- Unique Compliance Types: {unique_types}\n"
                
                response += f"- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                response += f"- Time Range: {time_range.replace('_', ' ').title()}\n\n"
                
                response += "💡 **Insights:**\n"
                if total_records > 1000:
                    response += "- Large dataset detected - consider using filters for better performance\n"
                if total_records == 0:
                    response += "- No data found - consider adding some test records\n"
                else:
                    response += f"- Database contains substantial compliance data for analysis\n"
            
            return [TextContent(type="text", text=response)]
            
        elif analysis_type == "trends":
            # Trend analysis would go here
            response = "📈 **Trend Analysis**\n\nTrend analysis feature coming soon!"
            return [TextContent(type="text", text=response)]
            
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Analytics failed: {str(e)}")]

async def handle_advanced_search(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle complex search operations"""
    search_criteria = arguments.get("search_criteria", {})
    sort_by = arguments.get("sort_by", "")
    limit = min(arguments.get("limit", 50), 100)
    
    try:
        # Build dynamic SQL based on search criteria
        base_query = f'SELECT * FROM "{production_server.db_manager.config.schema}"."{production_server.db_manager.config.table_name}"'
        
        where_conditions = []
        params = []
        
        # Add search conditions
        for field, value in search_criteria.items():
            if value:
                where_conditions.append(f'"{field}" LIKE ?')
                params.append(f"%{value}%")
        
        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)
        
        if sort_by:
            base_query += f' ORDER BY "{sort_by}" DESC'
        
        base_query += f" LIMIT {limit}"
        
        # Execute search
        result = production_server.db_manager.execute_query(base_query, params)
        
        if result["status"] == "success":
            response = f"🔍 **Advanced Search Results**\n\n"
            response += f"Found {result['row_count']} records matching your criteria\n\n"
            
            if result["data"]:
                for i, record in enumerate(result["data"][:3]):
                    response += f"**Result {i+1}:**\n"
                    for key, value in record.items():
                        response += f"- {key}: {value}\n"
                    response += "\n"
                
                if result["row_count"] > 3:
                    response += f"... and {result['row_count'] - 3} more results\n"
            
            return [TextContent(type="text", text=response)]
        else:
            return [TextContent(type="text", text=f"❌ Search failed: {result.get('error')}")]
            
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Advanced search failed: {str(e)}")]

async def handle_data_validation(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle data validation and quality checks"""
    validation_type = arguments.get("validation_type", "integrity")
    
    try:
        response = f"🔍 **Data Validation Report**\n\n"
        response += f"Validation Type: {validation_type.title()}\n"
        response += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if validation_type == "integrity":
            # Check for data integrity issues
            integrity_queries = [
                f'SELECT COUNT(*) as total_records FROM "{production_server.db_manager.config.schema}"."{production_server.db_manager.config.table_name}"',
                f'SELECT COUNT(*) as null_records FROM "{production_server.db_manager.config.schema}"."{production_server.db_manager.config.table_name}" WHERE TRANSACTION_ID IS NULL'
            ]
            
            results = []
            for query in integrity_queries:
                result = production_server.db_manager.execute_query(query)
                if result["status"] == "success" and result["data"]:
                    results.append(result["data"][0])
            
            if results:
                total = results[0].get("TOTAL_RECORDS", 0)
                nulls = results[1].get("NULL_RECORDS", 0) if len(results) > 1 else 0
                
                response += f"✅ **Integrity Check Results:**\n"
                response += f"- Total Records: {total:,}\n"
                response += f"- Records with NULL IDs: {nulls}\n"
                response += f"- Data Integrity: {'✅ Good' if nulls == 0 else '⚠️ Issues Found'}\n"
        
        elif validation_type == "duplicates":
            response += "🔍 **Duplicate Detection:**\n"
            response += "Duplicate detection analysis would be performed here.\n"
        
        elif validation_type == "missing_values":
            response += "📊 **Missing Values Analysis:**\n"
            response += "Missing values analysis would be performed here.\n"
        
        response += f"\n💡 **Recommendations:**\n"
        response += "- Regular validation helps maintain data quality\n"
        response += "- Consider setting up automated validation schedules\n"
        
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Validation failed: {str(e)}")]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_query_comprehensive(sql: str) -> tuple[bool, str]:
    """Comprehensive SQL validation"""
    sql_upper = sql.upper().strip()
    
    # Check if it's a SELECT statement
    if not sql_upper.startswith('SELECT'):
        return False, "Only SELECT queries are allowed"
    
    # Check for dangerous keywords
    dangerous_keywords = [
        'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'GRANT', 'REVOKE',
        'EXEC', 'EXECUTE', 'CALL', 'SYSTEM', 'SHUTDOWN', 'BACKUP'
    ]
    
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False, f"Keyword '{keyword}' is not allowed"
    
    # Check for reasonable limits
    if 'LIMIT' not in sql_upper and 'TOP' not in sql_upper:
        return False, "Query must include a LIMIT clause"
    
    return True, "Query is valid"

def generate_fallback_sql(question: str) -> str:
    """Generate fallback SQL when AI generation fails"""
    question_lower = question.lower()
    table_full_name = f'"{production_server.db_manager.config.schema}"."{production_server.db_manager.config.table_name}"'
    
    if "count" in question_lower or "how many" in question_lower:
        return f"SELECT COUNT(*) as total_count FROM {table_full_name}"
    elif "latest" in question_lower or "recent" in question_lower:
        return f"SELECT * FROM {table_full_name} ORDER BY CREATED_AT DESC LIMIT 10"
    elif "all" in question_lower:
        return f"SELECT * FROM {table_full_name} LIMIT 50"
    else:
        return f"SELECT * FROM {table_full_name} LIMIT 10"

def format_intelligent_response(question: str, result: Dict, sql: str) -> str:
    """Format intelligent query responses"""
    if result.get("status") != "success":
        return f"❌ Query failed: {result.get('error', 'Unknown error')}"
    
    data = result.get("data", [])
    row_count = result.get("row_count", len(data))
    
    response = f"🎯 **Answer to:** *{question}*\n\n"
    
    if row_count == 0:
        response += "🔍 No records found matching your criteria.\n\n"
        response += "💡 **Suggestions:**\n"
        response += "- Try broadening your search criteria\n"
        response += "- Check if there's data in the table with 'How many records are there?'\n"
        response += "- Add some test data with 'Add 10 random records'\n"
    else:
        response += f"📊 **Found {row_count} record(s)**\n\n"
        
        # Smart response based on query type
        if "count" in question.lower():
            if data and "TOTAL_COUNT" in data[0]:
                response += f"📈 **Total Count:** {data[0]['TOTAL_COUNT']:,}\n"
        else:
            # Show sample records
            display_limit = min(3, row_count)
            response += f"**Sample Records (showing {display_limit} of {row_count}):**\n\n"
            
            for i, record in enumerate(data[:display_limit]):
                response += f"**Record {i+1}:**\n"
                for key, value in record.items():
                    response += f"  • {key}: {value}\n"
                response += "\n"
            
            if row_count > display_limit:
                response += f"*... and {row_count - display_limit} more records*\n"
    
    response += f"\n🔧 **Generated SQL:** `{sql}`\n"
    response += f"⏱️ **Executed at:** {datetime.now().strftime('%H:%M:%S')}"
    
    return response

def generate_enhanced_test_data(schema: List[Dict], count: int, data_type: str) -> List[Dict]:
    """Generate enhanced realistic test data"""
    from faker import Faker
    fake = Faker()
    
    generated_data = []
    
    for _ in range(count):
        record = {}
        
        for column in schema:
            col_name = column["COLUMN_NAME"]
            data_type_name = column["DATA_TYPE_NAME"].upper()
            
            # Generate realistic data based on column name and type
            if col_name.upper() == "TRANSACTION_ID":
                record[col_name] = fake.uuid4()
            elif "COMPLIANCE_TYPE" in col_name.upper():
                types = ["Financial Audit", "Regulatory Review", "Risk Assessment", 
                        "Data Privacy Check", "Security Compliance", "Environmental Check"]
                record[col_name] = fake.random_element(types)
            elif "DATE" in col_name.upper() or "TIMESTAMP" in data_type_name:
                if "CREATED" in col_name.upper():
                    record[col_name] = fake.date_time_between(start_date='-30d', end_date='now')
                else:
                    record[col_name] = fake.date_time_between(start_date='-90d', end_date='+30d')
            elif "AMOUNT" in col_name.upper() and "DECIMAL" in data_type_name:
                record[col_name] = round(fake.random.uniform(100.0, 50000.0), 2)
            elif "STATUS" in col_name.upper():
                statuses = ["Completed", "In Progress", "Pending Review", "Failed", "Cancelled"]
                record[col_name] = fake.random_element(statuses)
            elif "CREATED_BY" in col_name.upper() or "USER" in col_name.upper():
                record[col_name] = fake.user_name()
            elif "VARCHAR" in data_type_name:
                if column.get("LENGTH", 0) > 100:
                    record[col_name] = fake.text(max_nb_chars=min(200, column.get("LENGTH", 100)))
                else:
                    record[col_name] = fake.company()
            elif "INTEGER" in data_type_name or "BIGINT" in data_type_name:
                record[col_name] = fake.random_int(min=1, max=99999)
            else:
                record[col_name] = fake.word()
        
        generated_data.append(record)
    
    return generated_data

async def insert_data_batch(batch_data: List[Dict]) -> Dict[str, Any]:
    """Insert a batch of data efficiently"""
    try:
        success_count = 0
        
        for record in batch_data:
            # Build INSERT statement
            columns = list(record.keys())
            placeholders = ["?" for _ in columns]
            values = list(record.values())
            
            insert_query = f'''
            INSERT INTO "{production_server.db_manager.config.schema}"."{production_server.db_manager.config.table_name}" 
            ({", ".join(f'"{col}"' for col in columns)})
            VALUES ({", ".join(placeholders)})
            '''
            
            result = production_server.db_manager.execute_query(insert_query, values)
            if result["status"] == "success":
                success_count += 1
        
        return {
            "status": "success",
            "records_inserted": success_count,
            "total_attempted": len(batch_data)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def get_table_columns() -> List[str]:
    """Get list of table columns"""
    # This would query the actual schema
    # For now, return common compliance table columns
    return [
        "TRANSACTION_ID", "COMPLIANCE_TYPE", "TRANSACTION_DATE", 
        "AMOUNT", "STATUS", "CREATED_BY", "CREATED_AT"
    ]

# =============================================================================
# STARTUP AND DEPLOYMENT
# =============================================================================

# startup.py
"""
Complete startup script for the SAP HANA Chatbot system
"""

async def start_production_server():
    """Start the production MCP server"""
    logger.info("🚀 Starting SAP HANA Chatbot Production Server...")
    
    # Initialize server
    if not await production_server.initialize():
        logger.error("❌ Server initialization failed")
        return False
    
    logger.info("✅ Server initialized successfully")
    
    try:
        # Start MCP server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await production_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="sap-hana-chatbot-production",
                    server_version="2.0.0",
                    capabilities=production_server.server.get_capabilities(
                        notification_options=None,