from trial_sinks import SinkRegistry, TreeNavSink, NotesSink

regy = SinkRegistry()
regy.register(TreeNavSink())
regy.register(NotesSink())
lines = regy.get_topics_for_prompt()
print(lines)
