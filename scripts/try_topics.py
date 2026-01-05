import asyncio
from pathlib import Path
import json
from pprint import pprint
import fire
from gibbon.llm_ops.find_root import send_to_llm
from prompt_tools.topic_cats import TopicsOnly

OLLAMA_URL = "http://192.168.100.242:11434"
MODEL = "mistral:7b-instruct"

class TopicOps:

    def run_prompt(self, draft, use_tool_calling=True, print_prompts=False):
        yaml_path = Path(__file__).parent / "essay.yaml"
        prompt = TopicsOnly.make_prompt(draft, yaml_path, use_tool_calling=use_tool_calling)
        full_res = asyncio.run(send_to_llm(prompt, OLLAMA_URL, MODEL, use_tool_calling=use_tool_calling))
        if print_prompts:
            print('-'*50 + " prompt " + '-'*50)
            print(prompt)
        pprint(full_res.__dict__)
        matches = TopicsOnly.parse_llm_response(full_res)
        print('-'*50 + " matches " + '-'*50)
        pprint(matches)
        

    def new_topic(self, use_tool_calling=True, print_promts=False):
        draft = "Create new topic book_"
        yaml_path = Path(__file__).parent / "essay.yaml"
        prompt = TopicsOnly.make_prompt(draft, yaml_path, use_tool_calling=use_tool_calling)
        full_res = asyncio.run(send_to_llm(prompt, OLLAMA_URL, MODEL, use_tool_calling=use_tool_calling))
        print('-'*50 + " result " + '-'*50)
        pprint(full_res.__dict__)
        matches = TopicsOnly.parse_llm_response(full_res)
        print('-'*50 + " matches " + '-'*50)
        pprint(matches)


    def new_essay(self, use_tool_calling=True):
        draft = "Record an essay"
        yaml_path = Path(__file__).parent / "essay.yaml"
        prompt = TopicsOnly.make_prompt(draft, yaml_path, use_tool_calling=use_tool_calling)
        full_res = asyncio.run(send_to_llm(prompt, OLLAMA_URL, MODEL, use_tool_calling=use_tool_calling))
        print('-'*50 + " prompt " + '-'*50)
        print(prompt)
        print('-'*50 + " result " + '-'*50)
        pprint(full_res.__dict__)
        matches = TopicsOnly.parse_llm_response(full_res)
        print('-'*50 + " matches " + '-'*50)
        pprint(matches)

    def do_vague(self, use_tool_calling=True):
        draft = "essay"
        yaml_path = Path(__file__).parent / "essay.yaml"
        prompt = TopicsOnly.make_prompt(draft, yaml_path, use_tool_calling=use_tool_calling)
        full_res = asyncio.run(send_to_llm(prompt, OLLAMA_URL, MODEL, use_tool_calling=use_tool_calling))
        print('-'*50 + " prompt " + '-'*50)
        print(prompt)
        print('-'*50 + " result " + '-'*50)
        pprint(full_res.__dict__)
        matches = TopicsOnly.parse_llm_response(full_res)
        print('-'*50 + " matches " + '-'*50)
        pprint(matches)

if __name__ == '__main__':
    fire.Fire(Prompter)

