import spacy
from ruler_patterns import get_ruler_patterns

def create_test_nlp():
    nlp = spacy.blank("ru")  # без модели
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(get_ruler_patterns())
    return nlp

def test_text(text):
    nlp = create_test_nlp()
    doc = nlp(text)
    print(f"\nТекст: {text}\n")
    for ent in doc.ents:
        print(f"{ent.label_}: {ent.text} (позиции {ent.start_char}–{ent.end_char})")

if __name__ == "__main__":
    test_text("9 месяцев Фолиевая кислота табл п/о плен 400 мкг х30")
