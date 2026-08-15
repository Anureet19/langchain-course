import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")

    information = """
    Guru Nanak Dev Ji (1469–1539) was the founder of Sikhism and the first of the ten Sikh Gurus. He was born in Rai 
    Bhoi Ki Talwandi, now known as Nankana Sahib in Pakistan. From a young age, he believed in the oneness of God and 
    taught people to live with truth, kindness, humility, and compassion. He travelled widely to share his message and 
    encourage people to rise above religious and social divisions.
    
    Guru Nanak Dev Ji taught three important principles: **Naam Japna** (remembering and meditating on God), **Kirat 
    Karni** (earning an honest living), and **Vand Chhakna** (sharing with others and helping those in need). 
    He strongly opposed caste discrimination and believed that all people are equal. His teachings continue to guide 
    Sikhs around the world and are an important foundation of Sikh faith and values.
    """
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],template=summary_template
    )
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
