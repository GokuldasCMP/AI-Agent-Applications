from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os

load_dotenv()
llm = LLM(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))

# Agent: Topic Summarizer
summarizer_agent = Agent(
    role="AI Topic Summarizer",
    goal="Create clear, structured summaries of complex data science {topic} for interview preparation",
    backstory=(
        "You are an expert at breaking down technical topics into concise, easy-to-understand summaries. "
        "You highlight key concepts, use bullet points, and maintain interview relevance."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Task: Summarize given topic content
summarize_task = Task(
    description=(
        """
        Your task is to read and understand the following topic content:
        {topic_content}

        Then, generate a concise summary that includes:
        1. One-paragraph overview of the topic
        2. Key points (3-7 bullets)
        3. Real-world example or analogy (if applicable)
        4. A final 1-line takeaway
        """
    ),
    expected_output=(
        """
        A markdown-formatted summary including:
        - **Overview**
        - **Key Points**
        - **Example**
        - **Final Takeaway**
        """
    ),
    agent=summarizer_agent
)


# Create the Crew
crew = Crew(
    agents=[summarizer_agent],
    tasks=[summarize_task],
    verbose=True
)

# Sample Input
topic = "Basics of SQL"
topic_content = """
Structured Query Language (SQL) is the standard language used to communicate with and manipulate relational databases. It allows users to retrieve, insert, update, delete, and manage data with high precision and efficiency. SQL is declarative, meaning users specify what they want, and the database engine figures out how to execute it. It’s the backbone of data analysis, data engineering, and backend development in systems like MySQL, PostgreSQL, Oracle, MS SQL Server, and SQLite. SQL is composed of several sub-languages: DDL (Data Definition Language) for creating or altering tables (CREATE, ALTER, DROP), DML (Data Manipulation Language) for changing data (INSERT, UPDATE, DELETE), DQL (Data Query Language) for querying (SELECT), DCL (Data Control Language) for permissions (GRANT, REVOKE), and TCL (Transaction Control Language) for managing transactions (COMMIT, ROLLBACK). At its core, SQL uses tables, rows, and columns to organize data in a structured, normalized way—often leveraging primary keys, foreign keys, and indexes to optimize relational integrity and performance. Common operations include filtering (WHERE), sorting (ORDER BY), grouping (GROUP BY), joining tables (JOIN, LEFT JOIN, RIGHT JOIN, FULL OUTER JOIN, CROSS JOIN), and aggregations (COUNT, SUM, AVG, MIN, MAX). Advanced queries may use subqueries, window functions (ROW_NUMBER(), RANK(), LEAD(), LAG()), Common Table Expressions (CTEs), recursive queries, and views for abstraction and readability. In analytics and interviews, SQL is a core skill tested for roles like data analyst, data scientist, and business intelligence engineer, especially in scenarios involving customer segmentation, funnel analysis, A/B testing, and cohort tracking. Optimization techniques, such as using appropriate indexes, avoiding N+1 queries, and using EXPLAIN plans, are vital for scaling large datasets. Despite being around since the 1970s, SQL continues to thrive due to its combination of simplicity and power, its ANSI standardization, and its tight integration with modern ecosystems like Python (via libraries like pandasql or SQLAlchemy), data visualization tools (e.g., Tableau, Power BI), and cloud platforms (BigQuery, Snowflake, Redshift). While NoSQL alternatives like MongoDB or Cassandra exist, relational databases and SQL remain dominant in business applications where data consistency, complex querying, and structured schemas are crucial. In interview settings, candidates are often tested on real-world SQL problems involving joins, nested queries, filtering conditions, date manipulation (DATE_TRUNC, DATE_ADD, DATEDIFF), and even writing queries from scratch based on vague business scenarios. Therefore, mastering SQL not only demonstrates data literacy but also the ability to think analytically, debug logically, and optimize performance. Whether you're diagnosing churn patterns, evaluating ad campaign success, or building dashboards, SQL is the fundamental language of structured data and will remain a non-negotiable skill in the data world for years to come.
"""

# Run
result = crew.kickoff(inputs={"topic": topic, "topic_content": topic_content})
print(result)
