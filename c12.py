#Enhanced chatbot with schema-aware data insertion, random data generation, etc
from typing import Optional
from contextlib import AsyncExitStack
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from tool_discovery import generate_tool_help


import os
import sys
import asyncio
import json
import re
import aiofiles

# === Load environment variables ===
load_dotenv()

# === SAP AI Core Configuration ===
AICORE_CLIENT_ID = os.getenv("AICORE_CLIENT_ID")
AICORE_AUTH_URL = os.getenv("AICORE_AUTH_URL")
AICORE_CLIENT_SECRET = os.getenv("AICORE_CLIENT_SECRET")
AICORE_RESOURCE_GROUP = os.getenv("AICORE_RESOURCE_GROUP")
AICORE_BASE_URL = os.getenv("AICORE_BASE_URL")
LLM_DEPLOYMENT_ID = "d38dd2015862a15d"

def parse_tool_response(response_text):
    # First try the standard TOOL:/PARAMS: format
    tool_match = re.search(r"TOOL:\s*(\w+)", response_text)
    params_match = re.search(r"PARAMS:\s*(\{.*\})", response_text, re.DOTALL)
    if tool_match and params_match:
        tool_name = tool_match.group(1)
        params = json.loads(params_match.group(1))
        return tool_name, params
    
    # Try to parse JSON format response
    try:
        # Remove markdown code blocks if present
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Parse JSON
        json_response = json.loads(cleaned_text)
        if 'TOOL' in json_response and 'PARAMS' in json_response:
            return json_response['TOOL'], json_response['PARAMS']
    except (json.JSONDecodeError, KeyError):
        pass
    
    return None, None

def estimate_tokens(text):
    """Rough estimation of tokens (1 token ≈ 4 characters for English text)"""
    return len(text) // 4

def truncate_text(text, max_tokens=1000):
    """Truncate text to approximately max_tokens"""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools = []
        self.memory = []
        self.cached_schema = None
        self.max_memory_pairs = 3  # Keep only last 3 conversation pairs
        self.max_tokens_per_message = 2000  # Limit each message to ~2000 tokens

    # === Connect to Server ===
    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        print("\n✅ Connected to server with tools:", [tool.name for tool in self.tools])

    # === Memory Management ===
    def manage_memory(self, new_human_msg, new_ai_msg):
        """Add new messages and manage memory size"""
        # Truncate messages if they're too long
        if hasattr(new_human_msg, 'content'):
            new_human_msg.content = truncate_text(new_human_msg.content, self.max_tokens_per_message)
        if hasattr(new_ai_msg, 'content'):
            new_ai_msg.content = truncate_text(new_ai_msg.content, self.max_tokens_per_message)
        
        # Add new messages
        self.memory.append(new_human_msg)
        self.memory.append(new_ai_msg)
        
        # Keep only the last N pairs (2N messages)
        max_messages = self.max_memory_pairs * 2
        if len(self.memory) > max_messages:
            self.memory = self.memory[-max_messages:]

    def get_total_context_tokens(self, system_prompt, current_query):
        """Estimate total tokens in the current context"""
        total = estimate_tokens(system_prompt) + estimate_tokens(current_query)
        for msg in self.memory:
            if hasattr(msg, 'content'):
                total += estimate_tokens(msg.content)
        return total

    # === Get Schema with Caching ===
    async def get_schema(self, force_refresh=False):
        """Retrieve database schema, with caching for performance"""
        if self.cached_schema and not force_refresh:
            return self.cached_schema
        
        try:
            tool_result = await self.session.call_tool("get_schema", {})
            
            # Extract schema data from tool result
            schema_text = ""
            if hasattr(tool_result, 'content') and tool_result.content:
                for content in tool_result.content:
                    if hasattr(content, 'text'):
                        schema_text = content.text
                        break
            
            if schema_text:
                try:
                    self.cached_schema = json.loads(schema_text)
                    return self.cached_schema
                except json.JSONDecodeError:
                    self.cached_schema = {"raw_schema": schema_text}
                    return self.cached_schema
        except Exception as e:
            print(f"⚠️ Warning: Could not retrieve schema - {str(e)}")
            return None
        
        return None

    # === Schema-Aware Data Insertion ===
    async def handle_data_insertion(self, user_input: str, schema_data: dict):
        """Handle data insertion using schema-aware LLM processing"""
        
        # Truncate schema if it's too large
        schema_str = json.dumps(schema_data, indent=2)
        if estimate_tokens(schema_str) > 3000:
            schema_str = truncate_text(schema_str, 3000)
        
        schema_prompt = f"""
        You are a database insertion assistant. The user wants to insert data into a database.
        
        DATABASE SCHEMA (truncated if large):
        {schema_str}
        
        USER INPUT: "{user_input}"
        
        TASK: Analyze the user input and create appropriate insert_data parameters based on the schema.
        
        INSTRUCTIONS:
        1. Identify which table the user wants to insert data into (default: "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION" if unclear).
        2. Extract the data values from the user input.
        3. Match the extracted data with the appropriate schema columns.
        4. Handle data type conversions (strings, numbers, dates, etc.).
        5. Set reasonable defaults for missing required fields if possible.
        6. IMPORTANT: If the user wants to generate or create a **random** record (e.g., "generate a sample", "create dummy data"), 
        use this tool: generate_and_insert_random_entry and table default: "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION" if unclear
        7.Only use `insert_data` when the user provides **explicit values** for insertion (e.g., "insert a COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION with name John and ID 101").
        8. Return ONLY in this exact format:

        TOOL: <tool_name>
        PARAMS: { ... }

        
        TOOL: insert_data
        PARAMS: {{"table": "<table_name>", "data": {{"column1": "value1", "column2": "value2"}}}}
        
        IMPORTANT:
        - Use exact column names from the schema
        - Convert values to appropriate data types
        - For missing required fields, use reasonable defaults or ask for clarification
        - If the table name is ambiguous, use table "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION" as default
        - Ensure all JSON is properly formatted
        """
        
        llm = ChatOpenAI(deployment_id=LLM_DEPLOYMENT_ID)
        lc_messages = [HumanMessage(content=schema_prompt)]
        
        llm_response = llm.invoke(lc_messages)
        response_text = llm_response.content
        
        # Parse the LLM response for tool call
        tool_name, params = parse_tool_response(response_text)
        
        if tool_name in ("insert_data", "generate_and_insert_random_entry") and params:
            # Execute the insertion
            try:
                tool_result = await self.session.call_tool(tool_name, params)
                
                # Process and return user-friendly response
                return await self._process_insertion_result(user_input, tool_result, params)
                
            except Exception as e:
                return f"❌ Error inserting data: {str(e)}"
        else:
            return f"❌ Could not parse the insertion request. LLM Response: {response_text}"

    async def _process_insertion_result(self, original_input: str, tool_result, params: dict) -> str:
        """Process insertion result and provide user-friendly feedback"""
        
        # Extract result from tool response
        result_text = ""
        if hasattr(tool_result, 'content') and tool_result.content:
            for content in tool_result.content:
                if hasattr(content, 'text'):
                    result_text = content.text
                    break
        
        # Create a simple confirmation without using LLM
        table_name = params.get('table', 'table')
        inserted_data = params.get('data', {})
        
        # Parse the JSON response to check for success
        try:
            result_json = json.loads(result_text)
            # Check for success indicators
            if (result_json.get('message', '').lower().find('successfully') != -1 or 
                result_json.get('object') == 'insert_result'):
                record_count = len(inserted_data)
                return f"✅ Successfully inserted record into {table_name} table."
            else:
                return f"❌ Error inserting data into {table_name}: {result_text}"
        except json.JSONDecodeError:
            # Fallback to text-based checking
            if ("error" in result_text.lower() or "failed" in result_text.lower()):
                return f"❌ Error inserting data into {table_name}: {result_text}"
            else:
                record_count = len(inserted_data)
                return f"✅ Successfully inserted record into {table_name} table."

    # === Main Query Processing ===
    async def process_query(self, query: str) -> str:
        # Build system prompt with tool descriptions
        def format_tool_params(tool):
            if hasattr(tool, 'input_schema') and tool.input_schema and 'properties' in tool.input_schema:
                params = [f'{name}: {prop.get("type", "any")}' for name, prop in tool.input_schema['properties'].items()]
                return ', '.join(params)
            elif hasattr(tool, 'parameters') and tool.parameters:
                if isinstance(tool.parameters, list):
                    params = [f'{param.name}: {param.type}' for param in tool.parameters if hasattr(param, 'name') and hasattr(param, 'type')]
                    return ', '.join(params)
            return ''
        
        tool_descriptions = "\n".join([
            f"- {tool.name}({format_tool_params(tool)}): {tool.description}"
            for tool in self.tools
        ])

        # Check if this is a data insertion, deletion, or update request
        insertion_keywords = ['insert', 'add', 'create', 'new record', 'new row', 'save', 'store']
        deletion_keywords = ['delete', 'remove', 'drop']
        update_keywords = ['update', 'modify', 'change', 'set']
        help_keywords = ['what can you do', 'help', 'list tools', 'available tools', 'capabilities']

        lower_query = query.lower()
        if any(keyword in lower_query for keyword in insertion_keywords):
            schema_data = await self.get_schema()
            if schema_data:
                return await self.handle_data_insertion(query, schema_data)
            else:
                print("⚠️ Schema not available, using fallback method")
        elif any(keyword in lower_query for keyword in deletion_keywords):
            schema_data = await self.get_schema()
            if schema_data:
                return await self.handle_data_deletion(query, schema_data)
            else:
                print("⚠️ Schema not available, using fallback method")
        elif any(keyword in lower_query for keyword in update_keywords):
            schema_data = await self.get_schema()
            if schema_data:
                return await self.handle_data_update(query, schema_data)
            else:
                print("⚠️ Schema not available, using fallback method")
        elif any(keyword in query.lower() for keyword in help_keywords):   #Interactive Tool Discovery" feature
            return generate_tool_help(self.tools)
        
        # Create a more concise system prompt
        system_prompt = (
            "You are a helpful database assistant with these tools:\n"
            f"{tool_descriptions}\n\n"
            "INSTRUCTIONS:\n"
            "- For data queries/counts: use generate_and_insert_random_entry\n"
            "- For schema info: use get_schema\n"
            "- Default Schema: HANA_SCHEMA, Default Table: COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION\n"
            "- Tool format: TOOL: <name>\\nPARAMS: <JSON>\n"
            "- Otherwise, respond naturally in plain text\n"
            "- For delete_data: {\"table\": \"name\", \"where\": {\"col\": \"val\"}}\n"
            "- For update_data: {\"table\": \"name\", \"data\": {\"col\": \"new_val\"}, \"where\": {\"col\": \"match_val\"}}\n"
            "- for generate_and_insert_random_entry use null for values which are not being set"
        )

        # Check context size before proceeding
        estimated_tokens = self.get_total_context_tokens(system_prompt, query)
        if estimated_tokens > 100000:  # Leave some buffer
            print(f"⚠️ Context getting large ({estimated_tokens} tokens), clearing older memory...")
            self.memory = self.memory[-4:]  # Keep only last 2 pairs

        lc_messages = [SystemMessage(content=system_prompt)]
        lc_messages.extend(self.memory)
        lc_messages.append(HumanMessage(content=query))

        llm = ChatOpenAI(deployment_id=LLM_DEPLOYMENT_ID)
        llm_response = llm.invoke(lc_messages)
        response_text = llm_response.content

        # Store the latest exchange in memory with proper management
        self.manage_memory(HumanMessage(content=query), llm_response)

        # Clean up response if it's in JSON format
        if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
            try:
                json_response = json.loads(response_text)
                # Extract the actual response from common JSON structures
                for key in ['response', 'answer', 'content', 'message']:
                    if key in json_response:
                        response_text = json_response[key]
                        break
            except json.JSONDecodeError:
                pass  # If it's not valid JSON, keep original response

        tool_name, params = parse_tool_response(response_text)
        if tool_name:
            # Inject default table and schema if not provided
            if isinstance(params, dict):
                if 'table' not in params:
                    params['table'] = "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION"
                if 'schema' not in params:
                    params['schema'] = "HANA_SCHEMA"

            try:
                tool_result = await self.session.call_tool(tool_name, params)
                # Post-process the tool result
                processed_result = await self._process_tool_result(query, tool_name, tool_result)
                return processed_result
            except Exception as e:
                return f"❌ Error executing tool {tool_name}: {str(e)}"
        else:
            # If no tool was called but the query seems to need one, try to force a tool call
            if any(keyword in lower_query for keyword in ['find', 'get', 'show', 'retrieve', 'value', 'data', 'row', 'record', 'table']):
                # Try to extract table name from query
                table_name = "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION"  # default
                # Look for table name patterns
                import re
                table_match = re.search(r'\b(\w+_\w+_\w+)\b', query)  # Pattern like COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION
                if table_match:
                    table_name = table_match.group(1)
                elif 'COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION' in lower_query:
                    table_name = "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION"

                # Force a generate_and_insert_random_entry call
                try:
                    tool_result = await self.session.call_tool("generate_and_insert_random_entry", {"table": table_name})
                    processed_result = await self._process_tool_result(query, "generate_and_insert_random_entry", tool_result)
                    return processed_result
                except Exception as e:
                    return f"❌ Error retrieving data from {table_name}: {str(e)}"
            
            return response_text

    async def _process_tool_result(self, original_query: str, tool_name: str, tool_result) -> str:
        """Process tool result and provide a clear, human-readable answer"""
        
        # Extract the actual data from the tool result
        result_text = ""
        if hasattr(tool_result, 'content') and tool_result.content:
            for content in tool_result.content:
                if hasattr(content, 'text'):
                    result_text = content.text
                    break
        
        if not result_text:
            return "Sorry. I couldn't retrieve the data from the tool."
        
        # Parse JSON if it's JSON data
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            return f"Retrieved data: {result_text}"
        
        # Handle specific query patterns directly for better responses
        lower_query = original_query.lower()
        
        # If user asks for specific value/field
        if 'bill of lading' in lower_query and 'objectid' in lower_query:
            # Look for OBJECTID in the query
            import re
            objectid_match = re.search(r'objectid\s*(\d+)', lower_query)
            if objectid_match:
                target_objectid = objectid_match.group(1)
                
                # Search through the returned data
                if 'rows' in data and isinstance(data['rows'], list):
                    for row in data['rows']:
                        if isinstance(row, dict):
                            # Check for OBJECTID match (case insensitive)
                            for key, value in row.items():
                                if key.lower() == 'objectid' and str(value) == target_objectid:
                                    # Found the row, now look for bill of lading
                                    for bol_key, bol_value in row.items():
                                        if 'billof' in bol_key.lower().replace('_', '').replace(' ', '') or 'bol' in bol_key.lower():
                                            return f"The bill of lading for OBJECTID {target_objectid} is: {bol_value}"
                                    return f"Found OBJECTID {target_objectid} but no bill of lading field in the record."
                    return f"No record found with OBJECTID {target_objectid} in the data."
        
        # For other queries, use LLM interpretation but make it more focused
        data_str = json.dumps(data, indent=2)
        if estimate_tokens(data_str) > 3000:
            # If data is large, try to extract relevant parts first
            if 'rows' in data and isinstance(data['rows'], list) and len(data['rows']) > 0:
                # Show first few rows as sample
                sample_data = {"rows": data['rows'][:5], "total_rows": len(data['rows'])}
                data_str = json.dumps(sample_data, indent=2)
            else:
                data_str = truncate_text(data_str, 3000)
            
        interpretation_prompt = f"""
        User asked: "{original_query}"
        Tool returned this data: {data_str}
        
        Extract and provide the EXACT answer to the user's question. 
        - If they asked for a specific value, provide that exact value
        - If they asked for a count, give the number
        - If they asked about existence, answer yes/no
        - Be direct and specific, don't just summarize
        - If the answer isn't in the data, say so clearly
        """
        
        llm = ChatOpenAI(deployment_id=LLM_DEPLOYMENT_ID)
        lc_messages = [HumanMessage(content=interpretation_prompt)]
        interpretation_response = llm.invoke(lc_messages)
        
        return interpretation_response.content

    # === Chat Loop ===
    async def chat_loop(self):
        print("\n🤖 Enhanced S4HANA MCP Client Started — Type your queries or 'quit/exit' to exit.")
        print("💡 Default Table: COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION")
        print("🔧 Context management: Automatically managing memory to prevent token overflow")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ('quit', 'exit'):
                    break

                # Clear memory command
                if query.lower() in ('clear', 'reset', 'clear memory'):
                    self.memory = []
                    print("🧹 Memory cleared!")
                    continue

                # ✅ Handle file upload BEFORE normal processing
                if query.lower().startswith("/upload"):
                    try:
                        parts = query.split()
                        txt_path = parts[1]
                        table = parts[2] if len(parts) > 2 else "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION"
                        upload_response = await self.handle_file_upload(txt_path, table)
                        print("\n💬 Upload Result:\n" + upload_response)
                        continue  # Skip normal processing
                    except Exception as e:
                        print(f"❌ Upload command failed: {e}")
                        continue  # Skip processing

                # 🔄 Otherwise process normally
                response = await self.process_query(query)
                print("\n💬 Response:\n" + response)

            except Exception as e:
                error_msg = str(e)
                if "context_length_exceeded" in error_msg:
                    print(f"\n⚠️ Context length exceeded. Clearing memory and retrying...")
                    self.memory = []
                    try:
                        response = await self.process_query(query)
                        print("\n💬 Response:\n" + response)
                    except Exception as retry_e:
                        print(f"\n❌ Error after retry: {str(retry_e)}")
                else:
                    print(f"\n❌ Error: {error_msg}")

    # === Cleanup ===
    async def cleanup(self):
        await self.exit_stack.aclose()

    async def handle_file_upload(self, txt_file_path: str, table_name: str = "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION") -> str:
        """Read a .txt file with JSON array and insert data into the database"""
        try:
            async with aiofiles.open(txt_file_path, mode='r') as f:
                content = await f.read()
                records = json.loads(content)

            if not isinstance(records, list):
                return "❌ File must contain a JSON array of objects."

            tool_name = "upload_json_records"
            params = {
                "table": table_name,
                "records": records
            }

            tool_result = await self.session.call_tool(tool_name, params)

            # Extract and return message
            if hasattr(tool_result, 'content') and tool_result.content:
                for part in tool_result.content:
                    if hasattr(part, 'text'):
                        return part.text

            return "✅ Upload completed, but no detailed message was returned."

        except Exception as e:
            return f"❌ Failed to upload file: {e}"

    # === Data Deletion ===
    async def handle_data_deletion(self, user_input: str, schema_data: dict):
        """Handle data deletion using schema-aware LLM processing"""
        
        schema_str = json.dumps(schema_data, indent=2)
        if estimate_tokens(schema_str) > 2000:
            schema_str = truncate_text(schema_str, 2000)
            
        schema_prompt = f"""
        You are a database assistant. The user wants to delete data from a database.

        DATABASE SCHEMA (truncated if large): {schema_str}

        USER INPUT: "{user_input}"

        TASK: Create delete_data parameters.

        INSTRUCTIONS:
        1. Identify table (default: "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION" if unclear)
        2. Extract filter conditions
        3. Match columns with schema
        4. Return ONLY: TOOL: delete_data\\nPARAMS: {{"table": "<name>", "where": {{"col": "val"}}}}
        """
        llm = ChatOpenAI(deployment_id=LLM_DEPLOYMENT_ID)
        lc_messages = [HumanMessage(content=schema_prompt)]
        llm_response = llm.invoke(lc_messages)
        response_text = llm_response.content

        tool_name, params = parse_tool_response(response_text)
        if tool_name == "delete_data" and params:
            try:
                tool_result = await self.session.call_tool(tool_name, params)
                return await self._process_deletion_result(user_input, tool_result, params)
            except Exception as e:
                return f"❌ Error deleting data: {str(e)}"
        else:
            return f"❌ Could not parse the deletion request. LLM Response: {response_text}"

    async def _process_deletion_result(self, original_input: str, tool_result, params: dict) -> str:
        """Process deletion result and provide user-friendly feedback"""
        result_text = ""
        if hasattr(tool_result, 'content') and tool_result.content:
            for content in tool_result.content:
                if hasattr(content, 'text'):
                    result_text = content.text
                    break
        table_name = params.get('table', 'table')
        try:
            result_json = json.loads(result_text)
            if (result_json.get('message', '').lower().find('successfully') != -1 or 
                result_json.get('object') == 'delete_result'):
                return f"✅ Successfully deleted record(s) from {table_name}."
            else:
                return f"❌ Error deleting data from {table_name}: {result_text}"
        except json.JSONDecodeError:
            if ("error" in result_text.lower() or "failed" in result_text.lower()):
                return f"❌ Error deleting data from {table_name}: {result_text}"
            else:
                return f"✅ Successfully deleted record(s) from {table_name}."

    # === Data Update ===
    async def handle_data_update(self, user_input: str, schema_data: dict):
        """Handle data update using schema-aware LLM processing"""
        
        schema_str = json.dumps(schema_data, indent=2)
        if estimate_tokens(schema_str) > 2000:
            schema_str = truncate_text(schema_str, 2000)
            
        schema_prompt = f"""
        You are a database assistant. The user wants to update data in a database.

        DATABASE SCHEMA (truncated if large): {schema_str}

        USER INPUT: "{user_input}"

        TASK: Create update_data parameters.

        INSTRUCTIONS:
        1. Identify table (default: "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION" if unclear)
        2. Extract columns to update and new values
        3. Extract filter conditions (where clause)
        4. Match columns with schema
        5. Return ONLY: TOOL: update_data\\nPARAMS: {{"table": "<name>", "data": {{"col": "new_val"}}, "where": {{"col": "match_val"}}}}
        """
        llm = ChatOpenAI(deployment_id=LLM_DEPLOYMENT_ID)
        lc_messages = [HumanMessage(content=schema_prompt)]
        llm_response = llm.invoke(lc_messages)
        response_text = llm_response.content

        tool_name, params = parse_tool_response(response_text)
        if tool_name == "update_data" and params:
            try:
                tool_result = await self.session.call_tool(tool_name, params)
                return await self._process_update_result(user_input, tool_result, params)
            except Exception as e:
                return f"❌ Error updating data: {str(e)}"
        else:
            return f"❌ Could not parse the update request. LLM Response: {response_text}"

    async def _process_update_result(self, original_input: str, tool_result, params: dict) -> str:
        """Process update result and provide user-friendly feedback"""
        result_text = ""
        if hasattr(tool_result, 'content') and tool_result.content:
            for content in tool_result.content:
                if hasattr(content, 'text'):
                    result_text = content.text
                    break
        table_name = params.get('table', 'table')
        try:
            result_json = json.loads(result_text)
            if (result_json.get('message', '').lower().find('successfully') != -1 or 
                result_json.get('object') == 'update_result'):
                return f"✅ Successfully updated record(s) in {table_name}."
            else:
                return f"❌ Error updating data in {table_name}: {result_text}"
        except json.JSONDecodeError:
            if ("error" in result_text.lower() or "failed" in result_text.lower()):
                return f"❌ Error updating data in {table_name}: {result_text}"
            else:
                return f"✅ Successfully updated record(s) in {table_name}."
    

# === Entry Point ===
async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())