import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import Memory

load_dotenv()

config = {
    "version": "v1.1",
    "embedder":{
        "provider":"openai",
        "config": {"api_key": os.getenv("OPENAI_API_KEY"), "model": "text-embedding-3-small"}
    },
    "llm":{
        "provider":"openai",
        "config": {"api_key": os.getenv("OPENAI_API_KEY"), "model": "gpt-4.1-mini"}
    },
    "graph_store":{
        "provider":"neo4j",
        "config": {
            "url": os.getenv("NEO4J_CONNECTION_URL"),
            "username": os.getenv("NEO4J_USERNAME"),
            "password": os.getenv("NEO4J_PASSWORD")
        }
    },
    "vector_store":{
        "provider":"qdrant",
        "config": {
            "host": "localhost",
            "port": 6333
        }
    }
}

memory_client = Memory.from_config(config)
memory_result = None

client = OpenAI()

while True:

    user_input = input("\n> ")

    memory_result = None

    try:
        memory_search = memory_client.search(query = user_input, user_id = "SS")
        if (len(memory_search.get("results")) > 0):
            match_score = [m.get("score") for m in memory_search.get("results")][0]
            print(match_score)
            if (match_score >= 0.29):
                memory_result = [m.get("memory") for m in memory_search.get("results")][0]
    except:
        dummy = 0


    SYSTEM_PROMPT = f"""Use the below provided context about the user:
    {memory_result}
    """

    response = client.chat.completions.create(
            model = "gpt-4.1-mini",
            messages = [
                {"role": "user", "content": user_input},
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
    )

    ai_response = response.choices[0].message.content
    print(f"\n🤖: {ai_response}")

    memory_client.add(
        user_id = "SS",
        messages = [
            {"role": "user", "content": user_input}
            # {"role": "assistant", "content": ai_response}
        ]
    )