from spacy.tokens import Doc, Span
from spacy import displacy
import spacy
from flask import Flask, jsonify, render_template, request, make_response
# Use a pipeline as a high-level helper
from transformers import pipeline

# Make a Flask App
app = Flask(__name__)

# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("LampOfSocrates/bert-base-cased-sourav")
model = AutoModelForTokenClassification.from_pretrained("LampOfSocrates/bert-base-cased-sourav")

# Load the NER model and tokenizer from Hugging Face
#ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
ner_pipeline = pipeline("ner", model=model, tokenizer = tokenizer)
nlp = spacy.load("en_core_web_sm") # only for rendering

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['message']
        entities = ner_pipeline(text)
        print(entities)
        # Convert HF entities to SpaCy's format
        doc = nlp(text)
        ents = []
        for entity in entities:
            start = entity['start']
            end = entity['end']
            label = entity['entity']
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents

        # Render using displacy
        html = displacy.render(doc, style='ent', page=False)
        print(html)
        return render_template('index.html', original_text=text, styled_text=html, entities=entities)
    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)