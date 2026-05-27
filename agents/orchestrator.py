from agents.analyzer import analyze_instruction
from agents.planner import state_order
from agents.coder import generate_code
from agents.reviewer import code_reviewer
from tools.preprocessing import PreprocessingTemplates

def run_pipeline(user_instruction):
    analyzed_data = analyze_instruction(user_instruction)
    if analyzed_data.get("status") == "error":
        print(f"Planning failed: {analyzed_data['reason']}")
        return

    while True:
        if analyzed_data["status"] == "valid":
            break
        elif analyzed_data["status"] == "correction":
            suggested_algorithm = analyzed_data["suggested_algorithm"]

            print(f"Did you mean: {suggested_algorithm}")
            print("1 -> If satisfied with topic")
            print("0 -> If not satisfied with topic")
            satisfied = input("Enter 0/1: ")

            if satisfied.strip() == "1":
                new_instruction = analyzed_data["corrected_instruction"]
                analyzed_data = analyze_instruction(new_instruction)

            elif satisfied.strip() == "0":
                print("Please describe your ML task clearly.")
                print("\nExample: 'Build a CNN for image classification'")
                new_instruction = input("Please enter the topic again with neat version and topic")
                analyzed_data = analyze_instruction(new_instruction)

            else:
                print("Enter a valid number")

        elif analyzed_data["status"] == "invalid":
            print("Your instruction is not ML related.")
            print(f"\nReason: {analyzed_data['reason']}")

            new_instruction = input("Please enter the topic again with neat version and topic: \n1")
            analyzed_data = analyze_instruction(new_instruction)


    planned_order = state_order(analyzed_data)
    
   # error check stays
    if planned_order.get("status") == "error":
        print(f"Planning failed: {planned_order['reason']}")
        return

    # while loop for missing fields
    required_fields = ["algorithm", "libraries", "sections", "evaluation_metric"]

    while True:
        missing = [field for field in required_fields if field not in planned_order]
        
        if not missing:
            break
        
        print(f"Planner output incomplete. Missing: {missing}. Retrying...")
        planned_order = state_order(analyzed_data)
        
        if planned_order.get("status") == "error":
            print(f"Planning failed: {planned_order['reason']}")
            return
        
    
    category = analyzed_data["category"].strip().lower()
    if "supervised" in category:
        key = "supervised"
    elif "nlp" in category or "text" in category:
        key = "nlp"
    elif "cnn" in category or "image" in category:
        key = "cnn"
    else:
        key = "supervised"  # default fallback
    
    template_name  = PreprocessingTemplates.registry[key]
    data_processor = getattr(PreprocessingTemplates, template_name)

    generated_code = generate_code(planned_order, data_processor)
    # print(generated_code)
    if generated_code.get("status") == "error":
        print(f"Code generation failed: {generated_code['reason']}")
        return
    

    final_corrected_code = code_reviewer(generated_code["code"], analyzed_data)
    if final_corrected_code.get("status") == "error":
        print(f"Code reviewing failed: {final_corrected_code['reason']}")
        return
    
    return final_corrected_code["code"]
        