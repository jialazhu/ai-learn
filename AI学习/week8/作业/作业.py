import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from TorchCRF import CRF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """实体类"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.0

@dataclass
class FAQItem:
    """FAQ项目"""
    id: int
    question: str
    answer: str
    category: str = ""

@dataclass
class QueryResult:
    """查询结果"""
    answer: str
    entities: List[Entity]
    similar_questions: List[Tuple[str, float]]
    confidence: float

class NERModel:
  """命名实体识别模型"""

  def __init__(self, model_type: str = 'bilstm_crf'):
    self.model_type = model_type
    self.model = None
    self.word_to_ix = {}
    self.tag_to_ix = {}
    self.ix_to_tag = {}
    # 标签集合
    self.labels = ['O', 'B-PRODUCT', 'I-PRODUCT', 'B-ORDER', 'I-ORDER',
                   'B-TIME', 'I-TIME', 'B-LOCATION', 'I-LOCATION',
                   'B-PRICE', 'I-PRICE']

    for i, label in enumerate(self.labels):
      self.tag_to_ix[label] = i
      self.ix_to_tag[i] = label

  def prepare_data(self, data: List[Dict]) -> List[Tuple[List[str], List[str]]]:
    """准备训练数据"""
    training_data = []

    for item in data:
      text = item['text']
      entities = item['entities']

      # 字符级别标注
      chars = list(text)
      labels = ['O'] * len(chars)

      # 标注实体
      for entity in entities:
        start, end = entity['start'], entity['end']
        label = entity['label']

        if start < len(labels):
          labels[start] = f'B-{label}'
          for i in range(start + 1, min(end, len(labels))):
            labels[i] = f'I-{label}'

      training_data.append((chars, labels))

    return training_data

  def build_vocab(self, training_data: List[Tuple[List[str], List[str]]]):
    for chars,labels in  training_data:
      for char in chars :
        if char not in self.word_to_ix:
          self.word_to_ix[char] = len(self.word_to_ix)

    self.word_to_ix["<UNK>"] = len(self.word_to_ix)
    self.word_to_ix["<PAD>"] = len(self.word_to_ix)

  def train_bilstm_crf(self, training_data: List[Tuple[List[str], List[str]]]):

    class BiLstmCRF(nn.Module):
      def __init__(self, vocab_size, tagset_size, embedding_dim = 10, hidden_dim = 20):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_idx = tagset_size
        self.tagset_size = len(tagset_size)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True,batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim,self.tagset_size)
        self.crf = CRF(self.tagset_size)

      def forward(self,sentence):
        mebeds = self.word_embeds(sentence).unsqueeze(0)
        lstm_out ,_ = self.lstm(mebeds)
        eminssions = self.hidden2tag(lstm_out)
        return self.crf.decode(eminssions.transpose(0,1))[0]

      def neg_log_likelihood(self, sentence, tags):
        embeds = self.word_embeds(sentence).unsqueeze(0)
        lstm_out, _ =  self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return -self.crf(emissions.transpose(0,1),tags.unsqueeze(1))

    self.model = BiLstmCRF(len(self.word_to_ix),self.tag_to_ix)
    optimizer = optim.Adam(self.model.parameters(),lr=0.01)

    for epoch in range(100):
      total_loss = 0
      for chars,labels in training_data[:10]:
        char_ids = torch.tensor([self.word_to_ix.get( c, self.word_to_ix['<UNK>']) for c in chars])
        label_ids = torch.tensor([self.tag_to_ix[l] for l in labels])

        self.model.zero_grad()
        loss = self.model.neg_log_likelihood(char_ids,label_ids)
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()

      logging.info(f"Epoch:{epoch} Loss:{total_loss:.4f}")

  def predict(self,text:str) -> List[Entity]:
      if self.model is None:
        logger.warning("模型未训练，返回空结果")
        return []
      chars = list(text)
      char_ids = torch.tensor([self.word_to_ix.get(c,self.word_to_ix['<UNK>']) for c in chars])

      with torch.no_grad():
        predict_tags = self.model(char_ids)

      entities = []
      current_entity = None

      for i,tag_id in enumerate(predict_tags):
        tag = self.ix_to_tag[tag_id]
        if tag.startswith("B-"):
          if current_entity:
            entities.append(current_entity)
          current_entity = Entity(text=text[i],label=tag[2:],start=i,end=i+1,confidence=0.8)
        elif tag.startswith("I-") and current_entity:
          current_entity.text += chars[i]
          current_entity.end = i +1
        else:
          if current_entity:
            entities.append(current_entity)
            current_entity = None

      if current_entity:
        entities.append(current_entity)

      return entities

  def evaluate(self,test_data: List[Dict]) -> Dict[str,float]:
    correct =0
    total = 0
    for item in test_data[:5]:
      predicted_entities = self.predict(item['text'])
      true_entities = item['entities']

      for pred_entity in predicted_entities:
        for true_entity in true_entities:
          if (pred_entity.text == true_entity['text'] and
            pred_entity.label == true_entity['label']) :
            correct += 1
            break

      total += len(true_entities)

    precision = correct / max(total , 1)

    return {
      "precision": precision,
      "recall":precision,
      "f1":precision
    }

  def train(self, tran_data: List[Dict],val_data: List[Dict] = None):
    training_data = self.prepare_data(tran_data)
    self.build_vocab(training_data)
    if self.model_type == "bilstm_crf":
      self.train_bilstm_crf(training_data)
    else:
      print(f"不支持模型类型:{self.model_type}")


class TextMatcher:
  def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    self.model_name = model_name
    self.model = SentenceTransformer(model_name)

  def encode_texts(self, texts: List[str]) -> np.ndarray:
    return self.model.encode(texts)

  def find_similar(self,query:str, candidates:List[str],top_k:int = 5) -> List[Tuple[str, float]]:
    if not candidates:
      return []

    all_texts = [query] + candidates
    embeddings = self.encode_texts(all_texts)
    query_embedding = embeddings[0:1]
    candidate_embeddings = embeddings[1:]
    similarities = cosine_similarity(query_embedding,candidate_embeddings)[0]

    results = [(candidates[i],float(sim)) for i,sim in enumerate(similarities) ]
    results.sort(key = lambda  x:x[1],reverse=True)
    return results[:top_k]

  def batch_search(self, queries: List[str], candidates: List[str]) -> List[List[Tuple[str, float]]]:
    return [self.find_similar(q,candidates) for q in queries]


class IntelligentQASystem:
  def __init__(self):
    self.ner_model = NERModel()
    self.text_matcher = TextMatcher()
    self.faq_dataBase = []
    self.faq_questions = []

  def load_faq(self,faq_data:List[Dict]):
    self.faq_dataBase = [FAQItem(**item) for item in faq_data]
    self.faq_questions = [item.question for item in self.faq_dataBase]

  def train_model(self,train_data:List[Dict]):
    self.ner_model.train(train_data)

  def generate_answer(self, entities:List[Entity],similar_questions:List[Tuple[str,float]]) -> str:
    if not similar_questions:
      return "抱歉，没有找到答案。"

    best_question,similarity = similar_questions[0]

    for faq in self.faq_dataBase:
      if faq.question == best_question:
        answer = faq.answer

        if entities:
          entity_info = "、".join([f"{e.label}: {e.text}" for e in entities])
          answer += f"\n\n检测到相关信息：{entity_info}"
        return answer

    return "抱歉，没有找到答案。"

  def calculate_confidence(self,entities:List[Entity],similar_questions:List[Tuple[str,float]]) -> float:
    if not similar_questions:
      return 0.0

    max_similarity = similar_questions[0][1]
    emtity_bonus = min(len(entities)*0.1,0.3)
    return min(max_similarity + emtity_bonus,1.0)
  def process_query(self,query:str) -> QueryResult:
    entities = self.ner_model.predict(query)

    if self.faq_questions:
      similar_questions = self.text_matcher.find_similar(query,self.faq_questions,top_k = 3)
    else:
      similar_questions = []

    answer = self.generate_answer(entities,similar_questions)
    confidence = self.calculate_confidence(entities,similar_questions)

    return QueryResult(answer,entities,similar_questions,confidence)

def main():
  with open("dataSet.json","r",encoding="utf-8") as f:
    data = json.load(f)
  train_data = data["questions"]
  faq_data = data["faq"]

  system = IntelligentQASystem()
  system.load_faq(faq_data)
  system.train_model(train_data)
  result = system.process_query("一加12什么时候发货?")
  print(f"识别实体: {[(e.text, e.label) for e in result.entities]}")
  print(f"相似问题: {result.similar_questions[:3]}")
  print(f"置信度: {result.confidence:.2f}")
  print(f"答案: {result.answer}")

  metrics = system.ner_model.evaluate(train_data)
  print(f"NER模型性能: {metrics}")

if __name__ == "__main__":
  main()