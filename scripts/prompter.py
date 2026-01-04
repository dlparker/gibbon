import asyncio
import fire
from  prompt_styles import do_draft_prompt
from pprint import pprint
class Prompter:
    def do_cats_1(self):
        draft = "Iron mines"
        yaml_path = "topo.yaml"
        style = "yaml_cats_1"
        res = asyncio.run(do_draft_prompt(draft, yaml_path, style))
        print('-'*50 + " prompt " + '-'*50)
        print(res['prompt'])
        print('-'*50 + " result " + '-'*50)
        pprint(res['full_res'].__dict__)
        print('-'*50 + " matches " + '-'*50)
        pprint(res['matches'])

    def do_two_step(self):
        draft1 = "Metal ore production"
        yaml_path = "topo.yaml"
        style = "yaml_cats_1"
        res = asyncio.run(do_draft_prompt(draft1, yaml_path, style))
        print('-'*50 + " prompt " + '-'*50)
        print(res['prompt'])
        print('-'*50 + " result " + '-'*50)
        pprint(res['full_res'].__dict__)
        print('-'*50 + " matches " + '-'*50)
        pprint(res['matches'])
        print("+"*100)
        context = res['matches'][0]
        draft2 = "Iron Mine"
        res2 = asyncio.run(do_draft_prompt(draft2, yaml_path, style, use_context=context))
        print('-'*50 + " prompt " + '-'*50)
        print(res2['prompt'])
        print('-'*50 + " result " + '-'*50)
        pprint(res2['full_res'].__dict__)
        print('-'*50 + " matches " + '-'*50)
        pprint(res2['matches'])
        

    def do_flat_1(self):
        draft = "Iron mines"
        yaml_path = "topo.yaml"
        style = "flat_yaml_1"
        res = asyncio.run(do_draft_prompt(draft, yaml_path, style))
        print('-'*50 + " prompt " + '-'*50)
        print(res['prompt'])
        print('-'*50 + " result " + '-'*50)
        pprint(res['full_res'].__dict__)
        print('-'*50 + " matches " + '-'*50)
        pprint(res['matches'])


if __name__ == '__main__':
    fire.Fire(Prompter)

