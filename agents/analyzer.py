import json
from utils.llm import call_llm
from agents.prompts import analyzer_prompt

def analyze_instruction(user_instruction):
    print("analyzer")
    response = call_llm(analyzer_prompt, user_instruction)
    print(response)

    try:
        start_idx = response.index("{")
        end_idx = response.rindex("}")

        response_dict = response[start_idx:end_idx+1]
        analyzed_data = json.loads(response_dict)

        return analyzed_data
    
    except Exception as e:
        return {"status": "error", "reason": str(e)}
