from agents.prompts import reviewer_prompt
from utils.llm import call_llm
import json

def code_reviewer(generated_code, analyzed_data):
    print("review")
    review_input = {
        "generated_code": generated_code,
        "analyzed_data": analyzed_data
    }
    response = call_llm(reviewer_prompt, json.dumps(review_input))

    try:
        if "```" in response:
            lines = response.split("\n")
            cleaned_lines = [line for line in lines if not line.strip().startswith("```")]

            response = "\n".join(cleaned_lines)

        return {"status": "success", "code": response}
        
    except Exception as e:
        return {"status": "error", "reason": str(e)}