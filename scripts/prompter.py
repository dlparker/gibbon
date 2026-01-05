import asyncio
from pathlib import Path
import json
from pprint import pprint
import fire
from gibbon.llm_ops.find_root import send_to_llm
from prompt_tools.topic_cats import TopicsOnly
from trial_sinks import SinkRegistry, TreeNavSink, NotesSink


OLLAMA_URL = "http://192.168.100.242:11434"
#MODEL = "llama3.1:8b-instruct-q4_K_M"
#MODEL = "mistral:7b"
MODEL = "mistral:7b-instruct"

class Prompter:

    def __init__(self):
        self.regy = SinkRegistry()
        self.regy.register(TreeNavSink())
        self.regy.register(NotesSink())

    def do_n_select(self, use_tool_calling=True):
        draft = "navigation select tree bob"
        sink_topics = self.regy.get_topics_for_prompt(as_list=True)
        prompt = TopicsOnly.make_prompt_2(draft, sink_topics, use_tool_calling=use_tool_calling)
        full_res = asyncio.run(send_to_llm(prompt, OLLAMA_URL, MODEL, use_tool_calling=use_tool_calling))
        print('-'*50 + " prompt " + '-'*50)
        pprint(prompt)
        print('-'*50 + " result " + '-'*50)
        pprint(full_res.__dict__)
        matches = TopicsOnly.parse_llm_response(full_res)
        print('-'*50 + " matches " + '-'*50)
        pprint(matches)
            
    def do_t_select(self, use_tool_calling=True):
        draft = "select topic bob"
        sink_topics = self.regy.get_topics_for_prompt(as_list=True)
        prompt = TopicsOnly.make_prompt_2(draft, sink_topics, use_tool_calling=use_tool_calling)
        full_res = asyncio.run(send_to_llm(prompt, OLLAMA_URL, MODEL, use_tool_calling=use_tool_calling))
        print('-'*50 + " prompt " + '-'*50)
        pprint(prompt)
        print('-'*50 + " result " + '-'*50)
        pprint(full_res.__dict__)
        matches = TopicsOnly.parse_llm_response(full_res)
        print('-'*50 + " matches " + '-'*50)
        pprint(matches)

    def do_continue(self, use_tool_calling=True):
        draft = "continue with tree bob"
        sink_topics = self.regy.get_topics_for_prompt(as_list=True)
        prompt = TopicsOnly.make_prompt_2(draft, sink_topics, use_tool_calling=use_tool_calling)
        full_res = asyncio.run(send_to_llm(prompt, OLLAMA_URL, MODEL, use_tool_calling=use_tool_calling))
        print('-'*50 + " prompt " + '-'*50)
        pprint(prompt)
        print('-'*50 + " result " + '-'*50)
        pprint(full_res.__dict__)
        matches = TopicsOnly.parse_llm_response(full_res)
        print('-'*50 + " matches " + '-'*50)
        pprint(matches)
            
    def new_topic(self, use_tool_calling=True):
        draft = "Create a new topic"
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

