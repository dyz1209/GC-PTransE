
class GlobalValue:
  # some train parameters
  vector_len = 100 # the embedding vector dimension
  learning_rate = 0.02
  margin = 0.1
  margin_relation = 0.2

  relation_num = 0
  entity_num = 0
  relation2id = {}
  entity2id = {}
  id2relation = {}
  id2entity = {}
  left_entity = {}
  right_entity = {}
  left_num = {}
  right_num = {}

  relation_vec = []
  entity_vec = []

  path_confidence = {}

