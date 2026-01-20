from pathlib import Path
import json
from typing import Optional
from dataclasses import dataclass, asdict, field
import jsonlines

from palaver_shared.draft_events import Draft
from gibbon.flow_types import AimTool, AimDef, AimToolResponse, MatchResult, DraftContext

@dataclass
class Task:
    name: str
    project_name: str
    description:str
    
@dataclass
class Project:
    name: str
    description:str
    match_examples: list[str]
    repo_dir: str
    preferred_words: Optional[list[str]] = None
    tasks: list[Task] = field(default_factory=list[Task])

init_projects =  {
    "palaver": Project(name= 'palaver',
                       description="Voice to text service",
                       match_examples=['Palaver',],
                       repo_dir="/home/dparker/projects/assist/palaver"),
    "gibbon": Project(name= 'gibbon',
                      description="Intent tree service",
                      match_examples=['Gibbon','Aim tree', 'Intent Tree'],
                      repo_dir="/home/dparker/projects/assist/gibbon"),
    "move-out-prep": Project(name= 'move-out-prep',
                        description="House sale prep",
                        match_examples=['House move','House sale prep', 'sell house'],
                        repo_dir="/home/dparker/projects/kboard"),
}

def load_projects(file_path='projects.jsonl'):
    filepath = Path(file_path)
    res = []
    if filepath.exists():
        with jsonlines.open(filepath) as reader:
            for proj in reader:
                project = Project(**proj)
                res.append(project)
        return res
    save_projects(init_projects)
    res = list(init_projects.values())
    return res
    
def save_projects(projects, file_path='projects.jsonl'):
    filepath = Path(file_path)
    with jsonlines.open(filepath, mode='w') as writer:
        for project in projects.values():
            wproj = asdict(project)
            writer.write(wproj)    

@dataclass
class OpState:
    is_current_tree: bool = False
    current_project_name: Optional[str] = None
    current_task_name: Optional[str] = None
    running_buffer: Optional[str] = ""
    draft: Optional[Draft] = None
    
class KBoardTool(AimTool):

    def __init__(self):
        self.op_state = OpState()
        self.projects = {}
        self.root_aim_def = AimDef(
            unique_name = 'project_management',
            description =  'User wants to interact with tasks or projects - create, search, modify, or query status of tasks/projects.',
            examples = '"Add task to fix door", "What tasks mention garage?", "Show project status", "Mark call as done"'
        )
        self.current_aim_defs = [self.root_aim_def,]
        self.select_project_aim_defs = []
        for project in load_projects():
            self.projects[project.name] = project
            aim_def = AimDef(unique_name=f"kboard-{project.name}",
                             description = project.description,
                             examples = project.match_examples,
                             preferred_words = project.preferred_words)
            self.select_project_aim_defs.append(aim_def)

        self.project_aim_defs = []
        self.project_aim_defs.append(AimDef(unique_name='kboard-create-task',
                                            description = "Create a new task for a project",
                                            examples = ["Make a new task", "Create a task",
                                                        "Add a task"]))
                                            
    def get_aim_defs(self):
        return self.current_aim_defs

    async def note_match(self, draft_context:DraftContext, match_res:MatchResult):
        self.op_state.is_current_tree = True
        reprocess_partial = None
        if match_res.intent_key == self.root_aim_def.unique_name:
            rescan_start = match_res.excerpt_pos
            rescan_end = len(draft_context.text)
            reprocess_partial = (rescan_start, rescan_end)
            self.current_aim_defs = self.select_project_aim_defs
        else:
            prefix = "kboard-"
            key_name = match_res.intent_key[len(prefix):]
            if match_res.intent_key in [aim.unique_name for aim in self.project_aim_defs]:
                if key_name == "create-task":
                    rescan_start = match_res.excerpt_pos + len(match_res.matched_phrase)
                    si = None
                    tmp = draft_context.text[rescan_start:].split()
                    for index, word in enumerate(tmp[:10]):
                        if word in ['named', 'called']:
                            si = index + 1
                            break
                    if si:
                        # we are going to assume that the speaker pauses before
                        # continuing after the name so that we see this
                        # as a text event and everything after the key phrase is the name
                        name = ' '.join(tmp[si:])
                        print(f"\n\nNew task name '{name}'\n\n")
            elif key_name in self.projects:
                project = self.projects[key_name]
                # look after the match
                rescan_start = match_res.excerpt_pos + len(match_res.matched_phrase)
                rescan_end = len(draft_context.text)
                reprocess_partial = (rescan_start, rescan_end)
                self.current_aim_defs = self.project_aim_defs

        response = AimToolResponse(True, True,
                                   reprocess_partial=reprocess_partial,
                                   reprocess_tool=self)
        return response
        

if __name__ == "__main__":
    foo()
    
