import streamlit as st
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import nest_asyncio

# Load environment variables
load_dotenv()

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Initialize Google Gemini AI
llm = ChatGoogleGenerativeAI(
    api_key=os.getenv('GOOGLE_API_KEY'),
    model="models/gemini-pro"  # Replace with the correct model name
)

# Define Agents
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article about the topic: {topic}. You collect information that helps the audience learn something and make informed decisions. Your work is the basis for the Content Writer to write an article on this topic.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write a compelling and well-structured blog post on {topic}.",
    backstory="You're a writer who uses the content plan to create a detailed blog post. Ensure it aligns with SEO best practices and the brand's voice.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization.",
    backstory="You are an editor who receives a blog post from the Content Writer. Your goal is to review the blog post to ensure that it follows journalistic best practices, provides balanced viewpoints when providing opinions or assertions, and also avoids major controversial topics or opinions when possible.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# Define Tasks
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering their interests and pain points.\n"
        "3. Develop a detailed content outline including an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources.\n"
        "5. The plan should be detailed and cover all aspects of the topic."
    ),
    expected_output="A comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named in an engaging manner.\n"
        "4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and alignment with the brand's voice.\n"
        "6. Each section should have 2 or 3 paragraphs.\n"
        "7. The entire blog post should be at least 1000 words."
    ),
    expected_output="A well-written blog post in markdown format, ready for publication.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given blog post for grammatical errors and alignment with the brand's voice."),
    expected_output="A polished and error-free blog post in markdown format, ready for publication.",
    agent=editor
)

# Define Crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2
)

# Streamlit App
def main():
    st.title("Content Creation Assistant")

    # Input for topic
    topic = st.text_input("Enter the topic for the blog post:")

    if st.button("Generate Content"):
        if topic:
            with st.spinner('Generating content...'):
                try:
                    result = crew.kickoff(inputs={"topic": topic})
                    st.markdown(result)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.error("Please enter a topic.")

if __name__ == "__main__":
    main()
