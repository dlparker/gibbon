from pathlib import Path
import json
from typing import Optional
from dataclasses import dataclass, asdict, field
import jsonlines

from palaver_shared.draft_events import Draft
from aim_select import AimTool, AimDef, AimToolResponse

@dataclass
class Task:
    name: str
    project_name: str
    description:str
    
@dataclass
class Project:
    name: str
    description:str
    repo_dir: str
    tasks: list[Task] = field(default_factory=list[Task])
    
init_projects=  {
    "palaver": Project(name= 'palaver',
                       description="Voice to text service",
                       repo_dir= "/home/dparker/projects/assist/palaver"),
    "gibbon": Project(name= 'gibbon',
                      description="Intent tree service",
                      repo_dir= "/home/dparker/projects/assist/gibbon"),
    "move out": Project(name= 'move out',
                        description="House sale prep",
                        repo_dir= "/home/dparker/projects/kboard"),
}

def load_projects(file_path='projects.jsonl'):
    filepath = Path(file_path)
    res = []
    with jsonlines.open(filepath) as reader:
        for proj in reader:
            project = Project(**proj)
            res.append(project)
    return res
    
def save_projects(file_path='projects.jsonl'):
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
        for project in load_projects():
            self.projects[project.name] = project
        
    def get_aim_defs(self):
        if not self.op_state.is_current_tree:
            return [
                AimDef(
                    unique_name = 'project_management',
                    description =  'User wants to interact with tasks or projects - create, search, modify, or query status of tasks/projects.',
                    examples = '"Add task to fix door", "What tasks mention garage?", "Show project status", "Mark call as done"'
                )
            ]
        raise Exception('not done')

    def start_draft(self, draft):
        self.draft = draft
        return AimToolResponse(success=True, can_continue=True)

    def end_draft(self, draft):
        self.ops.append(draft.full_text)
        return AimToolResponse(success=True, can_continue=True)

    def continuation_draft(self, draft):
        if not draft.end_text:
            self.draft = draft
            return AimToolResponse(success=False, can_continue=True)
        self.ops.append(draft.full_text)
        return AimToolResponse(success=True, can_continue=True)

if __name__ == "__main__":
    foo()
    
