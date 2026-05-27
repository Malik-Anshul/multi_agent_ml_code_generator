from agents.prompts import coder_prompt
from utils.llm import call_llm
import json

def generate_code(planned_order, data_processor):
    print("coder")
    coder_input = {
        "planned_order": planned_order,
        "data_processor": data_processor
    }
    response = call_llm(coder_prompt, json.dumps(coder_input), temperature=0.3)

    try:
        if "```" in response:
            lines = response.split("\n")
            cleaned_lines = [line for line in lines if not line.strip().startswith("```")]

            response = "\n".join(cleaned_lines)

        return {"status": "success", "code": response}
        
    except Exception as e:
        return {"status": "error", "reason": str(e)}