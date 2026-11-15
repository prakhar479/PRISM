"""Template families for E13 (task behaviour per language/script) and E14: object tracking (OT), running sum (RS) and
people-cities binding (PC), written in English (a parallel baseline), Hindi (Devanagari), romanized Hindi, Hinglish and French.

STATUS: every pack is a DRAFT written by the assistant and **unvalidated**: no native speaker has checked grammar or
naturalness (handoff 6: "to be checked by a native speaker and against the tokenizer"). `tests/harness/test_langfamilies.py`
checks the tokenizer-side invariants; `scripts/validate_templates.py` writes the review file for the humans. Rows built from
these families carry `validated: false` until a human signs the pack in `data/lang/templates_review.md`.

Design rules, each forced by a measured Gemma tokenizer fact (docs/PLAN_COMPLETION.md section 4/5):
  * Answers are one token: box labels are ASCII letters, sums are ASCII digits (as the pilot), cities are Latin-script names
    (in Gemma's vocabulary 1 of 45 Devanagari city names is a single token, and none of the Tamil or Bengali ones tested), so the
    Hindi people-cities pack has Latin-script city answers inside a Devanagari sentence: a caveat, stated in the report.
  * Clean and corrupt prompts must have equal length: the objects an OT pair swaps all have the same token count (word lists
    below are one length class each, verified by the test); people appear identically in A and B, so their length is free.
  * No gender or number agreement depends on the filled-in word: Hindi frames use passive / copula (है, रखा गया) and
    'X का घर', French frames start with the object phrase (capitalised, article included) and use 'va', 'est'.
  * Every string is NFC-stable (checked), because nukta letters such as ड़ have a composed form that NFC rewrites.
"""
from __future__ import annotations

from typing import Dict

from harness.families import OTFamily, PCFamily, RSFamily

LABELS = tuple("ABCDEFGH")
CITIES = ("Pune", "Delhi", "Mumbai", "Paris", "London", "Rome", "Berlin", "Tokyo", "Madrid", "Dubai", "Cairo", "Lima", "Oslo",
          "Vienna", "Prague", "Athens")                      # all single tokens in Gemma-2 (verified in the test)
PEOPLE_LATN = ("Asha", "Ravi", "Leo", "Nina", "Omar", "Priya", "Arjun", "Maya", "Sara", "Tom", "Anil", "Mira")
PEOPLE_DEVA = ("आशा", "रवि", "लीला", "नीना", "उमर", "प्रिया", "माया", "सारा", "अनिल", "मीरा")

OT = {f.id: f for f in [
    OTFamily("ot_en", ("hat", "shoe", "bell", "lamp", "fork", "doll", "drum", "vase"), LABELS,
             "The {o} is in box {b}.", "The {o} is put in box {b}.", "Now the {o} is in box",
             r"The (\w+) is in box (\w)\.", r"The (\w+) is put in box (\w)\.", r"Now the (\w+) is in box\s*$"),
    OTFamily("ot_hi", ("पैसा", "फूल", "दूध", "चाय", "दवा", "बैग", "ढोल", "छाता", "ताला", "दीया"), LABELS,
             "{o} डिब्बे {b} में है।", "{o} को डिब्बे {b} में रखा गया।", "{o} अब किस डिब्बे में है? उत्तर: डिब्बा",
             r"(\S+) डिब्बे ([A-Z]) में है।", r"(\S+) को डिब्बे ([A-Z]) में रखा गया।", r"(\S+) अब किस डिब्बे में है\? उत्तर: डिब्बा\s*$"),
    OTFamily("ot_hi_latn", ("chabi", "sikka", "kalam", "thaila", "chammach", "rassi", "diya", "bartan", "patthar", "phool"), LABELS,
             "{o} dabbe {b} mein hai.", "{o} ko dabbe {b} mein rakha gaya.", "{o} ab kis dabbe mein hai? Uttar: dabba",
             r"(\S+) dabbe ([A-Z]) mein hai\.", r"(\S+) ko dabbe ([A-Z]) mein rakha gaya\.", r"(\S+) ab kis dabbe mein hai\? Uttar: dabba\s*$"),
    OTFamily("ot_hinglish", ("hat", "shoe", "bell", "lamp", "fork", "doll", "drum", "vase"), LABELS,
             "{o} box {b} mein hai.", "{o} ko box {b} mein rakha gaya.", "{o} ab kis box mein hai? Answer: box",
             r"(\w+) box ([A-Z]) mein hai\.", r"(\w+) ko box ([A-Z]) mein rakha gaya\.", r"(\w+) ab kis box mein hai\? Answer: box\s*$"),
    OTFamily("ot_fr", ("Le chapeau", "Le livre", "La clé", "La balle", "Le stylo", "La lampe", "Le sac", "La tasse", "Le miroir", "La cloche"), LABELS,
             "{o} est dans la boîte {b}.", "{o} va dans la boîte {b}.", "{o} est maintenant dans la boîte",
             r"((?:Le|La) \w+) est dans la boîte ([A-Z])\.", r"((?:Le|La) \w+) va dans la boîte ([A-Z])\.", r"((?:Le|La) \w+) est maintenant dans la boîte\s*$"),
]}

RS = {f.id: f for f in [
    RSFamily("rs_en", "calc", openers=("Start at {s}.", "Begin with {s}.", "Initially {s}."), step="Add {a}.",
             closers=("Total =", "Answer:", "Result:"),
             start_re=r"^(?:Start at|Begin with|Initially) (\d)\.", add_re=r"Add (\d)\.", query_re=r"(Total =|Answer:|Result:)\s*$"),
    RSFamily("rs_hi", "calc", openers=("शुरुआत {s} से करें।", "आरंभिक मान {s} है।", "पहले {s} लें।"), step="{a} जोड़ें।",
             closers=("कुल =", "उत्तर:", "योग:"),
             start_re=r"^(?:शुरुआत|आरंभिक मान|पहले) (\d)(?= (?:से करें|है|लें)।)", add_re=r"(\d) जोड़ें।", query_re=r"(कुल =|उत्तर:|योग:)\s*$"),
    RSFamily("rs_hi_latn", "calc", openers=("Shuruaat {s} se karein.", "Aarambhik maan {s} hai.", "Pehle {s} lein."), step="{a} jodein.",
             closers=("Kul =", "Uttar:", "Yog:"),
             start_re=r"^(?:Shuruaat|Aarambhik maan|Pehle) (\d)(?= (?:se karein|hai|lein)\.)", add_re=r"(\d) jodein\.", query_re=r"(Kul =|Uttar:|Yog:)\s*$"),
    RSFamily("rs_hinglish", "calc", openers=("Start {s} se karo.", "Pehle {s} lo.", "Initial value {s} hai."), step="{a} add karo.",
             closers=("Total =", "Answer:", "Sum:"),
             start_re=r"^(?:Start|Pehle|Initial value) (\d)(?= (?:se karo|lo|hai)\.)", add_re=r"(\d) add karo\.", query_re=r"(Total =|Answer:|Sum:)\s*$"),
    RSFamily("rs_fr", "calc", openers=("Commencez à {s}.", "Valeur initiale {s}.", "Au début {s}."), step="Ajoutez {a}.",
             closers=("Réponse :", "Total =", "Résultat :"),
             start_re=r"^(?:Commencez à|Valeur initiale|Au début) (\d)\.", add_re=r"Ajoutez (\d)\.", query_re=r"(Réponse :|Total =|Résultat :)\s*$"),
]}

PC = {f.id: f for f in [
    PCFamily("pc_en", PEOPLE_LATN, CITIES, "{p}'s home is in {c}.", "{p}'s home is in",
             r"(\w+)'s home is in (\w+)\.", r"(\w+)'s home is in\s*$"),
    PCFamily("pc_hi", PEOPLE_DEVA, CITIES, "{p} का घर {c} में है।", "{p} का घर किस शहर में है? उत्तर:",
             r"(\S+) का घर (\S+) में है।", r"(\S+) का घर किस शहर में है\? उत्तर:\s*$"),
    PCFamily("pc_hi_latn", PEOPLE_LATN, CITIES, "{p} ka ghar {c} mein hai.", "{p} ka ghar kis shehar mein hai? Uttar:",
             r"(\w+) ka ghar (\w+) mein hai\.", r"(\w+) ka ghar kis shehar mein hai\? Uttar:\s*$"),
    PCFamily("pc_hinglish", PEOPLE_LATN, CITIES, "{p} ka home {c} mein hai.", "{p} ka home kis city mein hai? Answer:",
             r"(\w+) ka home (\w+) mein hai\.", r"(\w+) ka home kis city mein hai\? Answer:\s*$"),
    PCFamily("pc_fr", PEOPLE_LATN, CITIES, "La maison de {p} est à {c}.", "La maison de {p} est à",
             r"La maison de (\w+) est à (\w+)\.", r"La maison de (\w+) est à\s*$"),
]}

FAMILIES: Dict[str, dict] = {"entity_track": OT, "add_chain": RS, "people_cities": PC}

# family id -> (language code, script, how the language is written)
LANG_OF = {
    "ot_en": ("en", "Latn"), "rs_en": ("en", "Latn"), "pc_en": ("en", "Latn"),
    "ot_hi": ("hi", "Deva"), "rs_hi": ("hi", "Deva"), "pc_hi": ("hi", "Deva (city answers Latin)"),
    "ot_hi_latn": ("hi_latn", "Latn (romanized Hindi)"), "rs_hi_latn": ("hi_latn", "Latn (romanized Hindi)"),
    "pc_hi_latn": ("hi_latn", "Latn (romanized Hindi)"),
    "ot_hinglish": ("hinglish", "Latn (code-mixed)"), "rs_hinglish": ("hinglish", "Latn (code-mixed)"),
    "pc_hinglish": ("hinglish", "Latn (code-mixed)"),
    "ot_fr": ("fr", "Latn"), "rs_fr": ("fr", "Latn"), "pc_fr": ("fr", "Latn"),
}
LANGUAGES = ("en", "hi", "hi_latn", "hinglish", "fr")
VALIDATED: Dict[str, bool] = {k: False for k in LANG_OF}      # flipped only by a human, in data/lang/templates_review.md


def family_id(task: str, lang: str) -> str:
    pre = {"entity_track": "ot", "add_chain": "rs", "people_cities": "pc"}[task]
    return f"{pre}_{lang}"
