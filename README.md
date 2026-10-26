This project trains a TF-IDF classifier with regex features to score whether sentences from sustainability reports contain claims. It assigns each sentence an auto-accept, review, or discard decision, but the current results require human validation because testing found both false positives and missed claims.

## Web app

`app.py` is a small Flask app: upload a PDF and it runs the sentence extractor
from the sibling [Sentence-Extraction](../Sentence-Extraction) project, scores
the classifier-ready sentences with `claim_classifier.joblib`, and shows the
results. You can filter by decision, search, click a sentence to see its page
with the sentence highlighted, and download the classified and not-classified
CSVs.

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt pip  # spacy download needs pip
.venv\Scripts\python.exe -m spacy download en_core_web_trf
.venv\Scripts\python.exe app.py
```

Then open http://127.0.0.1:5000. Set `SENTENCE_EXTRACTION_DIR` if the
Sentence-Extraction checkout is not next to this one, or `CLAIM_MODEL_PATH` to
use another saved model. The first upload takes longer while the spaCy model
loads. Uploaded files go in `uploads/`, which is cleared each time the app starts.
