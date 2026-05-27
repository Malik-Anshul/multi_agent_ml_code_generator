from agents.orchestrator import run_pipeline

print("=" * 50)
print("Welcome to ML Code Generator!")
print("=" * 50)

user_instruction = input("\nEnter the topic for which you want to get code: \n")

code = run_pipeline(user_instruction)

if code is None:
    print("\nPipeline failed. Please try again.")

else:
    # Save generated code to a runnable Python file
    output_file = "generated_ml_code.py"
    
    with open(output_file, "w") as f:
        f.write(code)
    
    print("\n" + "=" * 50)
    print(f"Code successfully generated!")
    print(f"Saved to: {output_file}")
    print("=" * 50)
    print(f"\nCode Preview:\n")
    print(code)