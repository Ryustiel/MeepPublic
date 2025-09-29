
from typing import List

import os, langchain_openai, langchain_google_genai

from dotenv import load_dotenv
load_dotenv()

if True:  # Custom Models

    # Used in activity.py
    DECISION = langchain_openai.ChatOpenAI(
            model_name="gpt-4.1-nano",
            api_key=os.environ["OPENAI_API_KEY"],
        )

    # Used in summarize.py
    THINK = langchain_openai.ChatOpenAI(
            model_name="gpt-5-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            reasoning_effort="medium"
        )

    # Used in vision.py
    SUMMARIZE = langchain_openai.ChatOpenAI(
            model_name="gpt-5-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            reasoning_effort="low"
        )

    # Used in vision.py
    IMAGE = langchain_openai.ChatOpenAI(
            model_name="gpt-5-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            reasoning_effort="low"
        )

    # Used in converse.py
    CUSTOM = langchain_openai.ChatOpenAI(
            model_name="ft:gpt-4.1-2025-04-14:ima-benelux:meep:BfpLJrFI",
            api_key=os.environ["OPENAI_API_KEY"],
        )

elif False:  # GPT 5 Custom account

    DECISION = langchain_openai.ChatOpenAI(
            model_name="gpt-5-nano",
            api_key=os.environ["TEMP_OPENAI_KEY"],
        )
    THINK = DECISION
    SUMMARIZE = DECISION
    
    CUSTOM = langchain_openai.ChatOpenAI(
            model_name="gpt-5-mini",
            api_key=os.environ["TEMP_OPENAI_KEY"],
            reasoning=None,
        )
    IMAGE = CUSTOM
    
elif True:  # GPT 5

    DECISION = langchain_openai.ChatOpenAI(
            model_name="gpt-4.1-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            # reasoning_effort="minimal"
        )
    THINK = DECISION
    SUMMARIZE = DECISION
    
    CUSTOM = langchain_openai.ChatOpenAI(
            model_name="gpt-4.1-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            # reasoning_effort="minimal"
        )
    IMAGE = langchain_openai.ChatOpenAI(
            model_name="gpt-4.1-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0.0,
        )

else:  # Gemini
    
    # XXX : Uses service account, need to prevent that

    DECISION = langchain_google_genai.ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0.0
        )
    THINK = DECISION
    SUMMARIZE = DECISION
    
    CUSTOM = langchain_google_genai.ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0.7
        )
    IMAGE = CUSTOM
    
    # Because Google genai attempts a file read to get its own version
    print("Pre-emptively initializing Google GenAI clients...")
    try:
        _ = DECISION.async_client
        _ = CUSTOM.async_client
        print("Client initialization successful.")
    except Exception as e:
        print(f"An error occurred during client initialization: {e}")
