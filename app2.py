from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['SERPER_API_KEY'] = os.getenv("SERPER_API_KEY")

# Load LLM
llm = LLM(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))

# Search tool (SerperDev - like Google Search)
search_tool = SerperDevTool(n=10)

# Topic (Optional input from UI later)
topic = "Data Science Interview Questions"

# Agent 1: Interview Question Collector
question_collector = Agent(
    role="Interview Question Scraper",
    goal="Search and collect latest real interview questions for Data Science roles from credible platforms",
    backstory=(
        "You're an expert at scouring the internet for actual interview experiences shared on platforms like LeetCode Discussion, "
        "AmbitionBox, LinkedIn, Glassdoor, and similar. You can extract relevant questions, the company, position, and timeframe, "
        "ensuring the source is trustworthy and recent."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agent 2: Question Formatter
question_formatter = Agent(
    role="Interview Question Formatter",
    goal="Format scraped interview questions into a clean structured table with columns like question, company, position, timeframe, and source.",
    backstory=(
        "You're excellent at converting unstructured data into clean, structured Excel-style tables. "
        "You ensure each row contains a complete and accurate question entry with all relevant metadata."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Task 1: Search for questions
search_task = Task(
    description=(
        """
        Search for real and recent Data Science interview questions shared by candidates online.
        Focus on well-known platforms like:
        - LeetCode Discussion
        - AmbitionBox
        - LinkedIn posts
        - Glassdoor reviews
        - Medium blogs

        For each question, try to collect:
        - The exact interview question
        - Company name
        - Role/Position (e.g., Data Scientist, ML Engineer)
        - Timeframe or post date (at least month/year)
        - Source platform (with link if possible)

        Ensure only recent posts from the past 6–12 months are included.
        Collect at least 10 relevant examples.
        """
    ),
    expected_output=(
        """
        A raw list of 10+ real interview questions with:
        - Question
        - Company
        - Position
        - TimeFrame
        - Source (preferably include URL)
        """
    ),
    agent=question_collector
)

# Task 2: Format into Excel-style Table
formatting_task = Task(
    description=(
        """
        Take the collected interview questions and format them into a markdown table or CSV-style layout
        with the following columns:
        - Question
        - Company
        - Position
        - TimeFrame
        - Source

        Ensure consistency in formatting and make sure each field is complete.
        Prefer tabular formatting for easy export to Excel later.
        """
    ),
    expected_output=(
        """
        A clean markdown-style or CSV-style table with headers:
        Question | Company | Position | TimeFrame | Source
        """
    ),
    agent=question_formatter
)

# Define the Crew
crew = Crew(
    agents=[question_collector, question_formatter],
    tasks=[search_task, formatting_task],
    verbose=True
)

# Run the Crew
result = crew.kickoff(inputs={"topic": topic})

# Show the result
print(result)
