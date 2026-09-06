"""Curated exemplar query representations for semantic embedding-based intent routing."""

from enterprise_agent.schemas.router import QueryIntent

ROUTER_EXEMPLARS: dict[QueryIntent, list[str]] = {
    QueryIntent.DIRECT_CHAT: [
        "Hello, how are you today?",
        "Good morning assistant!",
        "Thanks a lot for your help earlier.",
        "Can you explain recursion in Python with a quick example?",
        "Tell me a short joke about programming.",
        "What is the capital of France?",
        "Hey there! Can you chat for a minute?",
        "Goodbye, talk to you tomorrow.",
    ],
    QueryIntent.RAG_SEARCH: [
        "What is the corporate travel expense reimbursement policy?",
        "How many days of paid parental leave do employees receive?",
        "What are the company guidelines for remote work equipment stipends?",
        "Explain the annual performance review evaluation criteria and timeline.",
        "What is our customer data security and retention protocol?",
        "Where can I find the standard employee code of conduct documentation?",
        "What is the policy regarding unused vacation rollover at year end?",
        "How do I submit an intellectual property disclosure form?",
    ],
    QueryIntent.SQL_DATABASE: [
        "How many employees are currently working in the Engineering department?",
        "What is the average salary of senior engineers across departments?",
        "Which products currently have a stock quantity below 50 units?",
        "What was our total revenue from completed sales orders in Q2?",
        "List all employees hired after January 2022 and their roles.",
        "What is the allocated budget for the Global Marketing department?",
        "Count the total number of hardware products in our inventory.",
        "Who is the employee with the highest salary in Product Management?",
    ],
    QueryIntent.AUTONOMOUS_AGENT: [
        "Look up the travel per diem policy and calculate the total meal allowance for 5 people.",
        "Check the current time, verify if offices are open, and calculate the time difference.",
        "Query the top product from database and draft an executive marketing summary.",
        "Find the server appliance price in database, apply an 18% discount, and format invoice.",
        "Cross-reference travel budget against current flight costs and compute remaining funds.",
        "Search handbook for holiday policy, calculate days remaining this quarter, and report.",
        "Inspect the employee roster and calculate the total payroll for department 1.",
        "Retrieve software license pricing and calculate total cost for 250 enterprise seats.",
    ],
}
