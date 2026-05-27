# Multi-Agent ML Code Generator 🚀

A raw Python multi-agent AI system that generates structured Machine Learning code using agent orchestration, preprocessing-aware generation, and multi-layer code review.

Unlike simple LLM wrappers, this system does not rely entirely on the LLM for preprocessing or generation flow.  
The project combines:
- manually engineered preprocessing templates
- structured planning agents
- validation pipelines
- multi-layer review agents

to generate cleaner and more reliable ML code.

---

# 🔥 Core Features

## ✅ 4-Layer Multi-Agent Pipeline

```text
User Instruction
      ↓
Analyzer Agent
      ↓
Planner Agent
      ↓
Coder Agent
      ↓
Reviewer Agent
      ↓
Final ML Code
```

---

## ✅ Analyzer Agent
Handles:
- vague instructions
- invalid ML requests
- incorrect algorithm-task combinations

Examples:
- CNN for house price prediction
- vague neural network requests
- non-ML instructions

The analyzer validates and corrects instructions before generation begins.

---

## ✅ Planner Agent
Creates a structured blueprint for:
- required libraries
- evaluation metrics
- section ordering
- preprocessing flow
- formatting structure

This prevents random code generation.

---

## ✅ Coder Agent
Generates complete ML code according to:
- planner instructions
- preprocessing templates
- formatting constraints
- architectural rules

The system does NOT rely fully on raw LLM generation.

---

## ✅ Reviewer Agent (4 Review Layers)

### Layer 1 — Syntax Validation
Checks:
- syntax errors
- indentation issues
- formatting problems

### Layer 2 — Logic Validation
Checks:
- variable mismatches
- import problems
- broken references

### Layer 3 — Requirement Validation
Checks whether generated code actually satisfies:
- user objective
- correct algorithm usage
- proper evaluation strategy

### Layer 4 — Final Correction Layer
Fixes:
- syntax issues
- logic issues
- formatting problems
- requirement mismatches

and returns cleaned final code.

---

# ⚙️ Manual Preprocessing System

One of the main features of this project is manually engineered preprocessing templates.

Instead of depending entirely on the LLM, preprocessing pipelines are predefined for:
- supervised learning
- NLP
- CNN workflows

This improves:
- preprocessing quality
- consistency
- reliability

---

# 🧠 Supported Algorithms

The system can generate code for many ML algorithms.

Current strongest support:
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN
- Lasso Regression
- Ridge Regression
- CNN
- FNN

Additional support:
- RNN
- LSTM

The project is currently most stable for:
- supervised learning workflows
- CNN pipelines
- FNN pipelines

---

# 🛠️ Tech Stack

- Python
- Groq API
- llama-3.3-70b-versatile
- PyTorch

---

# 🚫 No Frameworks Used

Built completely using raw Python.

No:
- LangChain
- AutoGen
- CrewAI
- LlamaIndex

The goal was to deeply understand:
- multi-agent orchestration
- prompt engineering
- structured generation
- validation pipelines

without abstraction frameworks.

---

# 📂 Project Structure

```text
ML_CODE_GENERATOR/
│
├── agents/
├── tools/
├── utils/
│
├── main.py
├── requirements.txt
└── generated_ml_code.py
```

---

# 💾 Output

The final generated code is automatically saved as:

```text
generated_ml_code.py
```

for easier testing and implementation.

---

# ▶️ Run Project

```bash
pip install -r requirements.txt
python main.py
```

---

# 💡 Example Instructions

```text
Build an SVM for spam detection
```

```text
Build a CNN for image classification
```

---

# 👨‍💻 Author

Anshul Malik 
Focused on AI Agents and Generative AI systems.
