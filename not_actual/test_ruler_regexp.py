import spacy
from spacy_labeling import build_spacy_ruler, remove_overlapping_ents

def test_text(text):
    nlp = spacy.blank("ru")
    build_spacy_ruler(nlp)                # подтягиваем и паттерны, и validate
    doc = nlp(text)
    # вот что вы не делали в тесте:
    doc.ents = remove_overlapping_ents(list(doc.ents))

    print(f"\nТекст: {text}\n")
    for ent in doc.ents:
        print(f"{ent.label_:>25}: {ent.text} ({ent.start_char}–{ent.end_char})")
"""
def test_text(text):
    nlp = spacy.blank("ru")
    doc = nlp(text)
    print("Токены:")
    for token in doc:
        print(f"[{token.i}] {token.text}")
"""

if __name__ == "__main__":
    test_text("9 месяцев Фолиевая кислота табл п/о плен 400 мкг х30")
