import os
import openai
import asyncio
import json
import time
 
from langgraph.checkpoint.memory import InMemorySaver
from deep_research.research_agent_full import deep_researcher_builder
import warnings

openai.api_key = os.environ["OPENAI_API_KEY"]

warnings.filterwarnings("ignore")
checkpointer = InMemorySaver()
full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)

from langchain_core.messages import HumanMessage

async def main(prompt):
    config = {
        "configurable": {"thread_id": "1"},
        "recursion_limit": 50
    }
    try:
        result = await (
            full_agent.ainvoke(
                {"messages": [HumanMessage(
                    content=prompt)]}, 
                    #content="Could you provide information on recent developments in cloud-based train control systems for urban rail transit? I'm also interested in understanding the key technologies involved.")]}, 
                config=config)
        )
    except Exception as e:
        print(f"Error: {e}")
        return "Error in main function"
    return result

def serialize_message(msg):
    """Convert LangChain message objects to dictionaries"""
    if hasattr(msg, 'dict'):
        # LangChain v0.1+
        return msg.dict()
    elif hasattr(msg, 'to_dict'):
        # Older versions
        return msg.to_dict()
    elif hasattr(msg, '__dict__'):
        # Fallback: convert to dict manually
        return {
            'type': msg.__class__.__name__,
            'content': getattr(msg, 'content', ''),
            'additional_kwargs': getattr(msg, 'additional_kwargs', {}),
            'response_metadata': getattr(msg, 'response_metadata', {}),
            'id': getattr(msg, 'id', ''),
            'tool_calls': getattr(msg, 'tool_calls', []),
        }
    else:
        return str(msg)

def serialize_result(obj):
    """Recursively serialize the result object"""
    if isinstance(obj, dict):
        return {k: serialize_result(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_result(item) for item in obj]
    elif hasattr(obj, '__class__') and 'Message' in obj.__class__.__name__:
        # It's a LangChain message object
        return serialize_message(obj)
    else:
        return obj

if __name__ == "__main__":
    file_path = "ydc_results_2.jsonl"
    input_questions_path = "/Users/zairah/su-sea/deep_research_bench/data/prompt_data/query.jsonl"

    with open(input_questions_path, "r") as f:
        # each row looks like this
        # {"id": 74, "topic": "Education & Jobs", "language": "en", "prompt": "Please conduct a study and prepare a report on the 'Construction and Application of a Sports Intelligent Tutoring and Learning Guidance System Driven by Multimodal Data Fusion.'"}
        input_questions = [json.loads(line.strip()) for line in f]
    
    for i, input_question in enumerate(input_questions):
        if input_question['language'] == 'en':
            # if i == 50 or i == 51:
            #     continue
            # with open(file_path, 'r') as f:
            #     for line in f:
            #         data = json.loads(line)
            #         print(data['messages'][0]['content'])
            #         if input_question == data['messages'][0]['content']:
            #             # result found, skip
            #             print(f"Result found for question: {input_question['prompt']}, skipping...")
            #             continue
            print(f"Processing question: {input_question['prompt']}")
            start_time = time.time()
            result = asyncio.run(main(input_question['prompt']))
            #result = asyncio.run(main("Nvidia latest news"))
            end_time = time.time()
            time_taken = (end_time - start_time)/60
            print(f"Time taken: {time_taken} minutes")
            
            print("******************")
            print(type(result))
            if isinstance(result, dict):
                print(result.keys())

            if result != "Error in main function":
                # Serialize the result
                serialized_result = serialize_result(result)
                serialized_result['time_taken'] = time_taken
                # Now save it
                with open(file_path, "a") as f:
                    f.write(json.dumps(serialized_result) + "\n")
            else:
                with open(file_path, "a") as f:
                    f.write(json.dumps({"error": "Error in main function"}) + "\n")
            print("Saved successfully!")
            print("\n\n")
