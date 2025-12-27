import spacy

nlp = spacy.load("trained_model")

text = "Infosys reported a net profit of Rs 1,619 crore in Q4."

doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
