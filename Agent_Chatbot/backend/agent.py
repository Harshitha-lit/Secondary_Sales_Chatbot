import os
import json
import logging
from litellm import completion
from tools import AVAILABLE_TOOLS, execute_tool

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("LLM_MODEL", "mistral/mistral-large-latest")

SYSTEM_PROMPT = """
You are an intelligent Business Assistant for forecasting and churn risk.
You have access to tools that can pull real-time data from the backend.

CRITICAL BUSINESS CONTEXT & TOOL LIMITATIONS:
1. "Secondary Sales" represent sales from distributors to **Retailers / Outlets**. When asked about "customers" or "churn", focus on Retailers/Outlets using the churn tools.
2. The forecast tool (`query_forecast_model`) now fully supports generating an overall aggregate forecast if you leave both `distributor_sk` and `sku_sk` empty.
3. For ANY general questions about past data (e.g. "historical secondary sales data of February 2026", "how many zones", "comparisons between months"), you MUST use the `query_secondary_data` tool to execute a SQL query on the raw data. Do NOT use the forecast tool for past data.
4. **DATE CONTEXT:** The current business year for this dataset is **2026**. The historical data only covers **January 2026 to April 2026**. If the user asks about "this year" or "current year", you MUST use 2026 in your SQL queries and text explanations. Do NOT assume the year is 2024.

IMPORTANT INSTRUCTIONS:
- To get data, you MUST invoke the provided tools using native tool calling. 
- ALWAYS FORMAT YOUR DATA AS A MARKDOWN TABLE (`| Column | Column |`). DO NOT use raw HTML tags (`<table>`). ReactMarkdown will automatically render the Markdown into a beautiful HTML table. CRITICAL: When showing churn risk or customer lists, you MUST ALWAYS include a dedicated "Months Inactive" column in your table.
- NO GRAPHS: You MUST always set `"render_chart": false` in the JSON output and leave the `"data"` array empty `[]`. Do not attempt to generate or render any charts.
- CRITICAL: Keep your text explanation extremely concise. Do not write long paragraphs. Less is more.
- CRITICAL NUMBER FORMATTING: You MUST format all large numbers using the Indian numbering system but use abbreviations (e.g., 'K' for thousands, 'L' for Lakhs, and 'Cr' for Crores). DO NOT use Millions or Billions. (For example, use ₹1.5L instead of ₹1.5 Lakhs or ₹150k, ₹3.14Cr instead of ₹3.14 Crores, and 50K instead of 50,000). If a number is a quantity, just append 'L' or 'Cr' and DO NOT write the word 'units' in the table.
- CONFIDENCE SCORE RULE: You MUST dynamically calculate the "confidence" (between 0.0 and 1.0) based on data availability and accuracy. DO NOT hardcode or blindly copy the example confidence.
- PROVENANCE RULE: In the `evidence` object, `provenance` MUST explain *why* we are getting that answer (the logic or features driving the result), NOT *where* we got the data from.
- CRITICAL EVIDENCE FORMATTING: DO NOT write the words "Evidence", "Provenance", or "Confidence" inside the `text_answer` string. You MUST put these details strictly inside the separate `evidence` JSON object. Do not duplicate them in the text.
- EVIDENCE CARD RULE: If the user asks a definition question or a general question NOT related to secondary sales data, you MUST leave the `evidence.sources` array empty `[]`.
- ONLY AFTER you have received the data, provide your final answer to the user in a STRICT JSON object matching this schema. You MUST include ALL keys below, especially `text_answer`:
{
  "text_answer": "Your detailed explanation here. Include names and figures pulled from the tools.",
  "evidence": {
    "sources": ["Name the specific ML model or tool used."],
    "confidence": 0.85,
    "provenance": "State exactly what numbers or logic you used to derive this answer from the tool."
  },
  "chart_data": {
    "render_chart": false,
    "type": "bar",
    "data": []
  },
  "suggested_questions": ["What is the forecast for SKU 123 at Distributor 456?", "List out the customers who will churn."]
}
CRITICAL: EVEN IF THE TOOL FAILS OR YOU NEED TO ASK THE USER FOR MORE INFO, YOU MUST STILL OUTPUT YOUR RESPONSE ENTIRELY AS A STRICT JSON OBJECT MATCHING THE SCHEMA ABOVE. DO NOT OUTPUT RAW CONVERSATIONAL TEXT.
"""

async def run_agent_loop(messages: list, max_iterations: int = 8) -> dict:
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        
    for _ in range(max_iterations):
        logger.info(f"Calling LLM with {len(messages)} messages.")
        import time
        
        # Retry loop to prevent Mistral Rate Limit crashes WITHOUT deleting history
        max_api_retries = 5
        for attempt in range(max_api_retries):
            try:
                response = completion(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=AVAILABLE_TOOLS,
                    tool_choice="auto",
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    num_retries=1
                )
                break # Success, break out of retry loop
            except Exception as e:
                if attempt < max_api_retries - 1:
                    logger.warning(f"Mistral API limit hit or error. Waiting 10 seconds before retry. ({e})")
                    time.sleep(10)
                else:
                    raise e
            
        response_message = response.choices[0].message
        messages.append(response_message.model_dump())
        
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                logger.info(f"LLM called tool natively: {function_name} with {arguments}")
                
                try:
                    tool_result = execute_tool(function_name, arguments)
                except Exception as e:
                    tool_result = {"error": str(e)}
                    
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(tool_result)
                })
        else:
            try:
                final_content = response_message.content or ""
                
                if "```json" in final_content:
                    final_content = final_content.split("```json")[1].split("```")[0]
                elif "```" in final_content:
                    final_content = final_content.split("```")[1].split("```")[0]
                    
                parsed_json = json.loads(final_content.strip(), strict=False)
                
                if "tool_call" in parsed_json or "tool_calls" in parsed_json:
                    tc_list = parsed_json.get("tool_calls", [])
                    if not tc_list and "tool_call" in parsed_json:
                        tc_list = [parsed_json["tool_call"]]
                        
                    if tc_list:
                        tc = tc_list[0]
                        name = tc.get("name", "")
                        args = tc.get("arguments") or tc.get("parameters") or {}
                        
                        if "forecast" in name.lower():
                            name = "query_forecast_model"
                            args["query_type"] = "get_forecast"
                            if "horizon_days" in args:
                                args["horizon"] = f"{args['horizon_days']}d"
                        elif "churn" in name.lower():
                            name = "get_top_churn_risks"
                        elif "data" in name.lower() or "sql" in name.lower() or "query" in name.lower():
                            name = "query_secondary_data"
                            
                        # Convert empty strings to None to satisfy Python
                        for k, v in list(args.items()):
                            if v == "":
                                args[k] = None
                            
                        try:
                            tool_result = execute_tool(name, args)
                        except Exception as e:
                            tool_result = {"error": str(e)}
                            
                        messages.append({
                            "role": "user", 
                            "content": f"SYSTEM: You tried to call a tool as JSON text. Here is the tool output:\n{json.dumps(tool_result)}\n\nNow, output your final answer strictly in the required JSON schema (text_answer, evidence, etc)."
                        })
                        continue 

                return parsed_json
                
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON. Attempting manual recovery.")
                import re
                
                extracted_text = response_message.content or "The LLM did not return text."
                
                # Extract text_answer more robustly
                text_match = re.search(r'"text_answer"\s*:\s*"(.*?)"\s*,\s*"(?:evidence|chart_data|suggested_questions)"', extracted_text, re.DOTALL)
                text_answer = text_match.group(1).strip() if text_match else extracted_text
                
                # Extract evidence dynamically by isolating the evidence JSON block
                confidence = 0.0
                provenance = "No provenance extracted."
                sources = ["AI Analysis"]
                
                # Use a more forgiving regex that allows nested brackets if LLM hallucinated them
                evidence_match = re.search(r'"evidence"\s*:\s*(\{.*?\})\s*(?:,|"chart_data"|"suggested_questions"|})', extracted_text, re.DOTALL)
                if evidence_match:
                    try:
                        import json as local_json
                        evidence_dict = local_json.loads(evidence_match.group(1))
                        confidence = float(evidence_dict.get("confidence", 0.0))
                        provenance = evidence_dict.get("provenance", "No provenance extracted.")
                        sources = evidence_dict.get("sources", ["AI Analysis"])
                    except Exception as e:
                        logger.error(f"Failed to parse evidence block: {e}")
                        # Fallback regex if json.loads fails on the isolated evidence block
                        conf_m = re.search(r'"confidence"\s*:\s*([\d\.]+)', evidence_match.group(1))
                        if conf_m: confidence = float(conf_m.group(1))
                        prov_m = re.search(r'"provenance"\s*:\s*"(.*?)"', evidence_match.group(1), re.DOTALL)
                        if prov_m: provenance = prov_m.group(1).strip()
                
                return {
                    "text_answer": text_answer,
                    "evidence": {
                        "sources": sources,
                        "confidence": confidence,
                        "provenance": provenance
                    },
                    "chart_data": {"render_chart": False, "type": "bar", "data": []},
                    "suggested_questions": []
                }
                
    logger.warning("Max iterations hit in agent loop.")
    return {
        "text_answer": "I attempted to fetch this data, but the queries were too complex and timed out after multiple attempts.",
        "evidence": {
            "sources": ["Agent Timeout"],
            "confidence": 0.0,
            "provenance": "Exceeded maximum internal tool-calling iterations (8 attempts)."
        },
        "suggested_questions": ["What is the overall business forecast?"]
    }
