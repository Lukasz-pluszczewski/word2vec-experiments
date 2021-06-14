import os
import sys
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq import hub_utils

model_path = os.path.join(os.getcwd(), "../data/roberta-transforms/roberta_large_fairseq")
loaded = hub_utils.from_pretrained(
    model_name_or_path=model_path,
    data_name_or_path=model_path,
    bpe="sentencepiece",
    sentencepiece_vocab=os.path.join(model_path, "sentencepiece.bpe.model"),
    load_checkpoint_heads=True,
    archive_map=RobertaModel.hub_models(),
    cpu=True
)
roberta = RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])
roberta.eval()
sentences = sys.argv[1:]

for sentence in sentences:
    print(roberta.fill_mask(sentence, topk=1))

#[('Atak na World Trade Center miał miejsce w 2001 roku.', 0.5372974872589111, ' 2001')]
#[('Ludzie nie boją się zmian.', 0.11347772181034088, ' zmian')]
#[('Druga wojna światowa zakończyła się w 1945 roku.', 0.9345270991325378, ' 1945')]
#[('Ludzie najbardziej boją się śmierci.', 0.14140743017196655, ' śmierci')]