import os 
from dotenv import load_dotenv 

from deepeval import evaluate 
from deepeval.test_case import LLMTestCase 
from openai import OpenAI
from deepeval.metrics import AnswerRelevancyMetric 
from deepeval.models import GPTModel 


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHATBOT_MODEL = "gpt-4.1-mini"
JUDGE_MODEL = "gpt-4.1-mini"

client = OpenAI(api_key=OPENAI_API_KEY)

def chatbot(question: str) -> str:

    response = client.responses.create(
       model = CHATBOT_MODEL,
       input = question

    )

    return response.output_text



def main(): 

    question = "What is the capital of India?"

    answer = chatbot(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

    test_case = LLMTestCase(
        input=question,
        actual_output=answer
    )

    judge_model = GPTModel(
        model=JUDGE_MODEL,
        api_key=OPENAI_API_KEY
    )
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=judge_model
    )

    evaluate(
        test_cases=[test_case],
        metrics=[metric]
    )



if __name__ == "__main__":
    main()