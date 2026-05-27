from agents.prompts import planner_prompt
from utils.llm import call_llm
import json

def state_order(analyzed_data):
    print("planner")
    response = call_llm(planner_prompt, json.dumps(analyzed_data))
    print(response)
    try:
        start_idx = response.index("{")
        end_idx = response.rindex("}")

        response_dict = response[start_idx:end_idx+1]
        planned_order = json.loads(response_dict)

        return planned_order
    
    except Exception as e:
        return {"status": "error", "reason": str(e)}

