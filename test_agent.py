from app.agent.agent import run_agent


question = """
I have a customer churn dataset at data/customers.csv.

Create a visualization showing the churn distribution.

Use the appropriate tool to create the chart.
Do not guess the data.
"""


result = run_agent(question, {"customers": "data/customers.csv"})

print("\n==============================")
print("FINAL AGENT RESPONSE")
print("==============================\n")

print(result["answer"])

print("\n==============================")
print("LATENCY BREAKDOWN")
print("==============================\n")

print(f"Total: {result['latency']['total_seconds']}s")
print(f"Tool calls: {result['latency']['tool_calls']}")

for stage in result["latency"]["stages"]:
    print(f"  {stage['stage']:<30} {stage['seconds']}s")