IGNORE_WORDS = [

    "morning",

    "night",

    "continue",

    "call",

    "between",

    "etc",

    "bp",

    "x"

]

def is_medicine(text):

    t = text.lower()

    for word in IGNORE_WORDS:

        if word in t:
            return False

    return True