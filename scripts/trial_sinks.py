from typing import  Optional, Protocol
from dataclasses import dataclass

from gibbon.store.models import Draft

@dataclass
class SinkResponse:
    success: bool
    can_continue: bool
    reprocess_partial: Optional[tuple[int,int]]

@dataclass
class TopicDef:
    key: str
    description: str
    root_key: Optional[str] = None
    parent: Optional['TopicDef'] = None

    @property
    def path(self):
        path_parts = [self.key]
        parent = self.parent
        while parent:
            path_parts.append(parent.key)
            parent = parent.parent
        return reversed(path_parts)

    def __post_init__(self):
        if self.parent and not self.root_key:
            parent = self.parent
            while parent:
                tmp = parent.key
                parent = parent.parent
            root_key = tmp
        elif not self.parent and self.root_key:
            raise Exception('root key cannot be set when parent is None')
        elif not self.parent:
            self.root_key = self.key
        
    
class SinkRegistry:

    def __init__(self):
        self._sinks: dict[DraftSink] = {}
        self._sink_index: dict[DraftSink] = {}

    def register(self, sink: 'DraftSink', overwrite=False):
        sink.set_registry(self)
        all_topics = sink.get_all_topics()
        first = all_topics[0]
        if not overwrite and first.root_key in self._sinks:
            raise Exception(f"{first.root_key} already registered")
        self._sinks[first.root_key] = sink
        for topic in all_topics:
            pname = ':'.join(topic.path)
            if not overwrite and pname in self._sink_index:
                raise ValueError(f"{pname} already registered")
            elif pname in self._sink_index and self._sink_index[pname] != sink:
                raise ValueError(f"{pname} registered to other sink {self._sink_index[pname]}")
            self._sink_index[pname] = sink

    def get_topics(self):
        # rebuild every time so that sinks can use state
        # to filter or select
        topics = []
        for name, sink in self._sinks.items():
            topics += sink.get_topics()
        return topics

    def get_topics_for_prompt(self, as_list=False):
        lines = []
        for topic in self.get_topics():
            lines.append(f"- key: {topic.key}")
            lines.append(f"  path: {' -> '.join(topic.path)}")
            lines.append(f"  description: {topic.description}")
        if as_list:
            return lines
        return "\n".join(lines)
    
    def dispatch(self, draft, matches):
        pass

class DraftSink(Protocol):

    def set_registry(self, registry: SinkRegistry) -> None: ...
    
    def get_all_topics(self) -> list[TopicDef]: ...
    """ get all topics that the sink can ever supply"""
    def get_topics(self) -> list[str]: ...
    """ construct a list of topics from the available match patterns and possibly the current state"""
    
    def matched_draft(self, draft, match_result) -> SinkResponse: ...
    """ called when sink topic detected, match_result says which one"""

    def continue_draft(self, draft) -> SinkResponse: ...
    """if previous  matched_draft call returned can_continue
    and new draft comes in that has not aim match, then
    the unmatched draft will be passed in to this call"""

    def get_name(self) -> str: ...
    """Return human-readable name for this sink."""


class TreeNavSink(DraftSink):

    def __init__(self):
        self.registry = None
        self.root = TopicDef(description="Topic Tree Navigation",
                             key="topic_tree_navigation")
        self.nav_select_tree = TopicDef(description="Navigation select tree",
                                    key="navigation_select_tree",
                                    parent=self.root)
        self.continue_with = TopicDef(description="continue with tree",
                                    key="continue_tree",
                                    parent=self.root)
        self.select_topic = TopicDef(description="Select topic",
                                    key="select_topic",
                                    parent=self.root)
        self.all_topics = [
            self.root,
            self.nav_select_tree,
            self.continue_with,
            self.select_topic,
        ]
                           

    def set_registry(self, registry: SinkRegistry) -> None:
        self.registry = registry
        
    def get_all_topics(self) -> list[TopicDef]:
        """ get all topics that the sink can ever supply"""
        return self.all_topics
    
    def get_topics(self) -> list[TopicDef]:
        """ get the topics that make sense given current state"""
        return self.all_topics
    
    def matched_draft(self, draft, match_result) -> SinkResponse:
        top_match = match_result[0]
        cat_name = top_match['category_name']
        match_string = top_match['matching_excerpt']
        index = draft.full_text.find(match_string)
        tree_bits = draft.full_text[index:].split()
        tree_name = "_".join(tree_bits).lower()
        result = SinkResponse(success=True, can_continue=False, reprocess_partial=None)
        return result
    
    def continue_draft(self, draft) -> SinkResponse:
        raise Exception("can't do this!")

    def get_name(self) -> str: 
        return "tree navigation"
    
class NotesSink(DraftSink):

    def __init__(self):
        self.registry = None
        self.root = TopicDef(description="General Notes",
                             key="general_notes")

        self.shopping = TopicDef(description="Shopping",
                                    key="shopping",
                                    parent=self.root)
        self.online = TopicDef(description="online shopping",
                                    key="online",
                                    parent=self.shopping)
        self.amazon =   TopicDef(description="Amazon Shopping",
                                    key="amazon",
                                    parent=self.online)
        self.heb =   TopicDef(description="H.E.B. Shopping",
                                    key="heb",
                                    parent=self.shopping)
                           
        self.all_topics = [
            self.root,
            self.shopping,
            self.online,
            self.amazon,
            self.heb
        ]
        
    def set_registry(self, registry: SinkRegistry) -> None:
        self.registry = registry
        
    def get_all_topics(self) -> list[TopicDef]:
        """ get all topics that the sink can ever supply"""
        return self.all_topics
    
    def get_topics(self) -> list[TopicDef]:
        """ get the topics that make sense given current state"""
        return self.all_topics
    
    def matched_draft(self, draft, match_result) -> SinkResponse:
        top_match = match_result[0]
        cat_name = top_match['category_name']
        match_string = top_match['matching_excerpt']
        index = draft.full_text.find(match_string)
        tree_bits = draft.full_text[index:].split()
        tree_name = "_".join(tree_bits).lower()
        result = SinkResponse(success=True, can_continue=False, reprocess_partial=None)
        return result
    
    def continue_draft(self, draft) -> SinkResponse:
        raise Exception("can't do this!")

    def get_name(self) -> str: 
        return "tree navigation"
    
