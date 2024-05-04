import openai
from typing import List
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# constant
COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 1500
SEPARATOR = "\n* "
separator_len = 3

def construct_prompt(most_relevant_sections: List[str]) -> str:
    
    chosen_sections = []
    chosen_sections_len = 0
    for section_index in most_relevant_sections:
        
        chosen_sections_len += len(section_index) + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + section_index)
    
    header = """請根據以下的資料回答問題，如果無法從下方的資料找到答案，則回答"無法回答"\n\n資料:\n"""
    
    return header + "".join(chosen_sections)

def answer_query_with_context(
    query: str,
    docs: List[str]
) -> str:
    prompt = construct_prompt(docs)

    logger.info(f'尋找到的相關資料:{prompt}')

    response = openai.ChatCompletion.create(
        model=COMPLETIONS_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        temperature = 0,
        max_tokens = 800
    )

    return response["choices"][0]["message"]["content"]
