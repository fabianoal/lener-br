
from model.ner_model import NERModel
from model.config import Config
from nltk import word_tokenize

config = Config()

model = NERModel(config)
model.build()
model.restore_session(config.dir_model)

def predict(words):
    words_tokens = word_tokenize(words, language='portuguese')
    return {'tokens': words_tokens, 'tags': model.predict(words_tokens)}

