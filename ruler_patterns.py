def get_ruler_patterns():
    return [
        {
            "label": "ПотребительскаяУпаковкаКолво",
            "pattern": [{"TEXT": {"REGEX": r"^(№\d+|[xх×]\d+)$"}}]
        },
        {
            "label": "Дозировка",
            "pattern": [
                {"TEXT": {"REGEX": r"^\d+([.,]\d+)?$"}},
                {"TEXT": {"REGEX": r"^(мг|мкг|г|мл|ед|МЕ|%)$"}}
            ]
        },
        {
            "label": "ЛекФорма",
            "pattern": [
                {"TEXT": {"REGEX": r"^(табл|капс|р-р|сусп|мазь|крем|гель|амп|саше|шпр|спрей|порошок)$"}},
                {"TEXT": {"REGEX": r"^(п|в)$"}, "OP": "?"},
                {"TEXT": "/"},  
                {"TEXT": {"REGEX": r"^(о|в|м)$"}, "OP": "?"},
                {"TEXT": {"REGEX": r"^(плен|жев)$"}, "OP": "?"}
            ]
        },
        {
            "label": "ТорговоеНаименование",
            "pattern": [
                {"IS_ALPHA": True},
                {"IS_ALPHA": True, "OP": "?"},
                {"IS_ALPHA": True, "OP": "?"},
                {"IS_ALPHA": True, "OP": "?"},
                {"IS_ALPHA": True, "OP": "?"}
            ]
        },
        {
            "label": "ТорговоеНаименование",
            "pattern": [
                {"TEXT": {"REGEX": r"^\d+$"}},
                {"IS_ALPHA": True},
                {"IS_ALPHA": True, "OP": "?"},
                {"IS_ALPHA": True, "OP": "?"},
                {"IS_ALPHA": True, "OP": "?"}
            ]
        }
    ]
