import yaml
from rapidfuzz import fuzz

class Arborist:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.tree = self.load_tree()
        self.flattened = self.flatten_tree(self.tree['roots'])  # Adjust if top-level changes

    def load_tree(self):
        with open(self.yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def flatten_tree(self, node, path=[], descriptions={}):
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = path + [key]
                if 'description' in value:
                    desc = value['description']
                    desc_key = '.'.join(new_path)
                    descriptions[desc_key] = desc
                if 'children' in value:
                    self.flatten_tree(value['children'], new_path, descriptions)
        return descriptions

    def sanity_check(self, threshold=80):
        """Flag high-similarity descriptions in unrelated branches."""
        similar_pairs = []
        for key1, desc1 in self.flattened.items():
            branch1 = key1.split('.')[0]
            for key2, desc2 in self.flattened.items():
                if key1 != key2:
                    branch2 = key2.split('.')[0]
                    if branch1 != branch2:
                        score = fuzz.partial_ratio(desc1.lower(), desc2.lower())
                        if score > threshold:
                            similar_pairs.append({
                                'key1': key1, 'desc1': desc1,
                                'key2': key2, 'desc2': desc2,
                                'score': score
                            })
        if similar_pairs:
            print("Sanity warnings: High cross-branch similarity")
            for pair in similar_pairs:
                print(f" - {pair['key1']} ({pair['desc1']}) vs {pair['key2']} ({pair['desc2']}) | Score: {pair['score']}")
            return similar_pairs
        print("Sanity check passed: No high cross-branch overlaps.")
        return []

    def validate_new_description(self, new_desc, new_branch, threshold=75):
        """Check if new description overlaps too much with unrelated branches."""
        for key, existing_desc in self.flattened.items():
            existing_branch = key.split('.')[0]
            if existing_branch != new_branch:
                score = fuzz.partial_ratio(new_desc.lower(), existing_desc.lower())
                if score > threshold:
                    return False, f"Reject: High overlap with {key} ({existing_desc}) | Score: {score}"
        return True, "Accepted"

    def prune_for_transcript(self, transcript, min_score=50):
        """Conservative prune: Keep subtree if any description matches transcript >= min_score."""
        transcript_lower = transcript.lower()
        pruned = {'roots': {}}
        for root_key, root_node in self.tree['roots'].items():
            subtree_flattened = self.flatten_tree({root_key: root_node})
            has_match = False
            for desc in subtree_flattened.values():
                score = fuzz.partial_ratio(transcript_lower, desc.lower())
                if score >= min_score:
                    has_match = True
                    break
            if has_match:
                pruned['roots'][root_key] = root_node
        if not pruned['roots']:
            print("Warning: No matches — falling back to full tree.")
            return self.tree
        return pruned

    def flatten_paths(self, include_non_leaves=True, concat_parents=False):
        paths = []
        def recurse(node, current_path=[], parent_desc=''):
            if isinstance(node, dict):
                for key, value in node.items():
                    new_path = current_path + [key]
                    desc = value.get('description', '')
                    full_desc = f"{parent_desc} -> {desc}" if concat_parents and parent_desc else desc
                    path_str = ' -> '.join(new_path)
                    # Optional: Prefix with DB ID if you query it from SQLModel
                    # path_str = f"ID:{db_id} | {path_str}"
                    if include_non_leaves or not value.get('children'):  # Leaves only if flag
                        paths.append({'path': path_str, 'description': full_desc})
                    if 'children' in value:
                        recurse(value['children'], new_path, full_desc)
        for root_key, root_node in self.tree['roots'].items():
            recurse({root_key: root_node})
        return paths

    def get_topic_paths(self, keywords_file=None):
        topic_paths = []
        def recurse(node, current_path=[]):
            if isinstance(node, dict):
                for key, value in node.items():
                    new_path = current_path + [key]
                    is_topic = value.get('is_topic', False) or value.get('topic', False)  # Handle both variants
                    if is_topic:
                        topic_paths.append({
                            'key': key,  # ← This is the critical short key!
                            'path': ' -> '.join(new_path),
                            'description': value.get('description', '')
                        })
                    if 'children' in value:
                        recurse(value['children'], new_path)
        for root_key, root_node in self.tree['roots'].items():
            recurse({root_key: root_node})
        return topic_paths

    def get_topic_paths_old(self, keywords_file=None):
        """
        Returns only paths marked as topics (is_topic=true).
        Skips structural nodes.
        Optionally merges keywords from separate file.
        """
        topic_paths = []

        def recurse(node, current_path=[]):
            if isinstance(node, dict):
                for key, value in node.items():
                    new_path = current_path + [key]

                    # Check if this is a topic
                    is_topic = value.get('is_topic', False)

                    if is_topic:
                        path_dict = {
                            'path': ' -> '.join(new_path),
                            'description': value.get('description', ''),
                        }
                        topic_paths.append(path_dict)

                    # Recurse into children regardless
                    if 'children' in value:
                        recurse(value['children'], new_path)

        for root_key, root_node in self.tree['roots'].items():
            recurse({root_key: root_node})

        # Optionally merge keywords from separate file
        if keywords_file:
            try:
                with open(keywords_file, 'r') as f:
                    keywords_dict = yaml.safe_load(f)
                    if keywords_dict:
                        for topic in topic_paths:
                            if topic['path'] in keywords_dict:
                                topic['keywords'] = keywords_dict[topic['path']]
            except FileNotFoundError:
                pass  # Keywords file is optional

        return topic_paths

# Example usage
if __name__ == "__main__":
    arbor = Arborist('topo.yaml')
    arbor.sanity_check()  # Run on load

    # Test adding new
    ok, msg = arbor.validate_new_description("Iron mining operations", new_branch='root2')
    print(msg)

    # Test pruning
    pruned = arbor.prune_for_transcript("Iron ore mine")
    print(yaml.dump(pruned, sort_keys=False))
