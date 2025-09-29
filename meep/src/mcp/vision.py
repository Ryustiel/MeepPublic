
from typing import List
from langchain_core.messages import HumanMessage, AIMessage

import graphs._llm as llm

IMAGE_PROCESSING_LLM = llm.IMAGE

async def process_image(url: str) -> str:
    try:
        desc: AIMessage = await IMAGE_PROCESSING_LLM.ainvoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Describe this image in details"
                        }, 
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url
                            }
                        }
                    ]
                )
            ]   
        )
        return desc.content
    except Exception as e:
        return f"Describe image failed. Error={str(e)}"
