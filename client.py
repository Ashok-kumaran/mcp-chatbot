import sys
import os
import asyncio
import json
import logging
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack
from dotenv import load_dotenv
import argparse

# Logging setup with UTF-8
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
        logging.FileHandler('client.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# LLM import
try:
    from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
    from langchain.schema.messages import HumanMessage
    logger.info("Successfully imported GenAI Hub ChatOpenAI")
except Exception as e:
    logger.warning(f"Could not import GenAI Hub ChatOpenAI: {e}. Falling back to run_sql.")
    ChatOpenAI = None

# MCP stdio client
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    logger.info("Successfully imported MCP client modules")
except Exception as e:
    logger.error(f"Failed to import MCP client modules: {e}")
    print(f"Error: MCP client modules not available: {e}", file=sys.stderr)
    sys.exit(1)

load_dotenv()

# Default values
DEFAULT_SCHEMA = os.getenv("HANA_SCHEMA")
DEFAULT_TABLE = os.getenv("HANA_TABLE_NAME", "COM_SIERRA_ECOBRIDGE_COMPLIANCETRANSACTION")
LLM_DEPLOYMENT_ID = os.getenv("LLM_DEPLOYMENT_ID", "d647c85d1614386c")

# Validate environment
if not all([DEFAULT_SCHEMA, DEFAULT_TABLE]):
    logger.error("Missing required environment variables: HANA_SCHEMA or HANA_TABLE_NAME")
    print("Error: Please set HANA_SCHEMA and HANA_TABLE_NAME in .env", file=sys.stderr)
    sys.exit(1)

def parse_tool_response(response_text: str):
    logger.debug(f"Parsing LLM response: {response_text}")
    try:
        txt = response_text.strip()
        if txt.startswith("```"):
            txt = txt.strip("```").strip()
        j = json.loads(txt)
        if isinstance(j, dict) and "TOOL" in j and "PARAMS" in j:
            logger.debug(f"Parsed JSON tool: {j['TOOL']}, params: {j['PARAMS']}")
            return j["TOOL"], j["PARAMS"]
    except Exception as e:
        logger.error(f"Failed to parse JSON response: {e}")
    return None, None

class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.writer = None
        self.tools = []

    async def connect_to_server(self, server_script_path: str):
        logger.info(f"Connecting to server: {server_script_path}")
        is_py = server_script_path.endswith(".py")
        if not is_py and not server_script_path.endswith(".js"):
            raise ValueError("server_script_path must be a .py or .js file")

        command = "python" if is_py else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        try:
            logger.debug("Creating stdio transport")
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.writer = stdio_transport
            logger.debug("Initializing client session")
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.writer))
            logger.debug("Sending initialize request")
            await asyncio.wait_for(self.session.initialize(), timeout=180.0)  # Increased to 180s
            logger.info("Client session initialized")
            print("Connected to server", file=sys.stderr)
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
            print(f"Error: Failed to connect to server: {e}", file=sys.stderr)
            raise

        try:
            logger.debug("Listing available tools")
            tools_resp = await self.session.list_tools()
            self.tools = tools_resp.tools if hasattr(tools_resp, "tools") else []
            logger.info(f"Available tools: {[t.name for t in self.tools]}")
            print(f"Available tools: {[t.name for t in self.tools]}", file=sys.stderr)
        except Exception as e:
            logger.error(f"Could not list tools: {e}")
            print(f"Warning: Could not list tools: {e}", file=sys.stderr)

    async def process_query(self, query: str):
        logger.info(f"Processing query: {query}")
        tools_list = [t.name for t in self.tools] if self.tools else ["run_sql", "get_data", "get_schema", "get_table_summary"]
        prompt = f"""
You are a database assistant for table {DEFAULT_TABLE} with 292 columns, including FUELONWARDSMATERIALDOCUMENTYEAR and FUEL_TYPE.
The user asked: "{query}"

Available tools: {', '.join(tools_list)}
- For counts or aggregations, use TOOL: run_sql PARAMS: {{"query": "{query}"}} 
- For schema info, use TOOL: get_schema PARAMS: {{}}
- For sample data, use TOOL: get_data PARAMS: {{"limit": 10}}
- For table summary (row count, columns), use TOOL: get_table_summary PARAMS: {{}}

Return ONLY:
{{
  "TOOL": "<tool_name>",
  "PARAMS": <json params>
}}
"""
        if not ChatOpenAI:
            logger.warning("LLM unavailable, falling back to run_sql")
            return await self._call_tool_and_extract("run_sql", {"query": query})

        llm = ChatOpenAI(deployment_id=LLM_DEPLOYMENT_ID, temperature=0.0)
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            text = resp.content
            logger.debug(f"LLM response: {text}")
            tool, params = parse_tool_response(text)
            if not tool or params is None:
                logger.warning(f"LLM failed to produce valid TOOL/PARAMS, falling back to run_sql. LLM output:\n{text}")
                return await self._call_tool_and_extract("run_sql", {"query": query})
            return await self._call_tool_and_extract(tool, params)
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return await self._call_tool_and_extract("run_sql", {"query": query})

    async def _call_tool_and_extract(self, tool_name: str, params: Dict[str, Any]):
        logger.debug(f"Calling tool: {tool_name} with params: {params}")
        try:
            tool_result = await self.session.call_tool(tool_name, params)
            logger.debug(f"Raw tool result: {tool_result}")
            text = None
            if hasattr(tool_result, "content") and tool_result.content:
                for part in tool_result.content:
                    if hasattr(part, "text"):
                        text = part.text
                        break
            if text is None:
                return tool_result if isinstance(tool_result, (dict, list)) else {"raw": str(tool_result)}
            try:
                result = json.loads(text)
                logger.debug(f"Parsed tool result: {result}")
                return result
            except Exception:
                logger.warning(f"Tool result not JSON, returning raw: {text}")
                return {"raw_text": text}
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return {"error": f"Tool call failed: {e}"}

    async def chat_loop(self):
        logger.info("Starting chat loop")
        print("\nMCP Chatbot - type 'exit' or 'quit' to stop.")
        print(f"Default table: {DEFAULT_TABLE} | Default schema: {DEFAULT_SCHEMA}")
        while True:
            try:
                q = input("\nYou: ").strip()
                if q.lower() in ("exit", "quit"):
                    break
                if not q:
                    continue
                result = await self.process_query(q)
                print("\nResult:\n", json.dumps(result, indent=2, ensure_ascii=False))
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Chat loop error: {e}")
                print(f"Error: {e}", file=sys.stderr)

    async def cleanup(self):
        logger.info("Cleaning up client session")
        await self.exit_stack.aclose()

async def main():
    parser = argparse.ArgumentParser(description="MCP Chatbot Client")
    parser.add_argument("server_script", help="Path to server script (e.g., server.py)")
    parser.add_argument("--query", help="Run a single query and exit")
    args = parser.parse_args()

    client = MCPClient()
    try:
        await client.connect_to_server(args.server_script)
        if args.query:
            result = await client.process_query(args.query)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            await client.chat_loop()
    except Exception as e:
        logger.error(f"Main error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())