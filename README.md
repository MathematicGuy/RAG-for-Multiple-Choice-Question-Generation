# RAG for multiple choice questions generation

This project aims to build a RAG model for generating MCQ (multiple choice questions) from a PDF file


# Set Up LangSmith for debugging
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY='your_langsmith_api_key'
LANGSMITH_PROJECT='rag-aio-project'
HUGGING_FACE_API_KEY = 'your_huggingface_api_key'

# Folder Structure
RAG - Folder for main code file
Streamlit - Testing RAG with Streamlit FE
TestingFeatures - Testing features related to RAG.

### Dummy text for testing
dummy_text = """
In statistics, degrees of freedom (df) represents the number of independent pieces of information available to estimate a parameter or to calculate a statistic. It essentially reflects the "wiggle room" in your data after accounting for constraints. A higher degree of freedom generally implies more information, while a lower degree of freedom indicates less.
Here's a more detailed explanation:

    Independent Observations:
    Degrees of freedom indicate how many values in a dataset can vary freely without violating any imposed restrictions or assumptions.

Estimating Parameters:
When calculating statistics like the mean or standard deviation, you use up degrees of freedom because these values are derived from the data.
Constraints:
Each parameter estimated reduces the number of available degrees of freedom. For example, if you know the mean of a sample, you have one less degree of freedom because one value can be derived if you know the rest.
Example:
If you have a sample of 5 numbers and calculate the mean, you have 5-1 = 4 degrees of freedom. Knowing the mean doesn't tell you the exact values of all 5 numbers, but it does constrain one of them.
Why it matters:
Degrees of freedom are crucial for determining the appropriate statistical tests and interpreting results accurately.

In simpler terms:

    Think of it like a see-saw:
    You can freely move the first two people on a see-saw, but the third person's position is determined by where the first two are to keep it balanced, thus only two have "freedom".

Imagine a coin flip:
If you know the results of 3 coin flips (e.g., H, H, T), you don't know the fourth flip's result, but if you also know the overall percentage of heads, you can infer the fourth flip (e.g., H). This extra information reduces the degrees of freedom.

Essentially, degrees of freedom help us understand how much information we have in our data and how it can be used to make inferences about a larger population.
"""
