#!/usr/bin/env python3
"""Export a Hassaniya language pack for acon96/home-llm's synthetic data generator.

home-llm (https://github.com/acon96/home-llm, the "Local LLM Conversation" HA integration
+ Home models) builds its multilingual training data from CSV "piles" in
data/piles/<language>/. This writes a `hassaniya` pile from our evidence-checked lexicon so
their generator, integration and training configs work for Hassaniya with no other changes
except the small prompting.py patch in integrations/home-llm/prompting_hassaniya.patch.py.

    python3 scripts/export_home_llm.py              # -> integrations/home-llm/piles/hassaniya/
    # then, inside a home-llm checkout:
    cp -r <Gg>/integrations/home-llm/piles/hassaniya data/piles/
    python3 <Gg>/integrations/home-llm/apply_prompting_patch.py data/prompting.py
    cd data && python3 generate_data.py --train --small --language hassaniya
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hassaniya import ROOT, count_with_unit, load_lexicon, strip_al  # noqa: E402

LEX = load_lexicon()
OUT = ROOT / "integrations" / "home-llm" / "piles" / "hassaniya"
R = random.Random(5)
V = LEX["verbs"]
SOFT = ["", "", "حگل ", "الله يخليك "]          # attested softeners/politeness, prefixed
SOFT_POST = ["", "", " حگل", " الله يخليك"]


def imp(key, n=3):
    v = V[key]
    return list(dict.fromkeys(v["imp_m"][:n] + v["imp_f"][:1] + v["req"][:1]))


# --------------------------------------------------------------- device names
AREAS = [(a["id"], a["ar"][0]) for a in LEX["areas"]]
DEV_TYPES = {  # home-llm type -> (our lexicon item canonical words, id stem)
    "light": [("ظو", "dhaw"), ("لامبة", "lamba"), ("ثريّا", "thurayya")],
    "fan": [("مروحة", "marwaha"), ("فانتيلو", "ventilo")],
    "climate": [("كليم", "klim")],
    "media_player": [("تلفزة", "tilfza"), ("راديو", "radio"), ("سبيكر", "speaker")],
    "switch": [("شوفو", "chauffe_eau"), ("بومبة", "pompe"), ("پريز", "prise"), ("فريجو", "frigo")],
    "blinds": [("ريدو", "rideau"), ("طاگة", "taga")],
    "vacuum": [("روبو", "robot")],
}


def device_rows():
    rows = []
    for t, items in DEV_TYPES.items():
        for word, stem in items:
            for aid, aname in AREAS:
                rows.append({"device_name": f"{t}.{aid}_{stem}", "description": f"{word} {aname}"})
    rows += [{"device_name": "garage_door.garage", "description": "باب الگاراج"},
             {"device_name": "garage_door.hawch", "description": "باب الحوش"},
             {"device_name": "lock.bab_ed_dar", "description": "باب الدار"},
             {"device_name": "lock.bab_el_hawch", "description": "باب الحوش"},
             {"device_name": "lock.bab_el_madkhal", "description": "باب المدخل"},
             {"device_name": "lock.garage", "description": "كادنة الگاراج"},
             {"device_name": "timer.atay", "description": "منبّه أتاي"},
             {"device_name": "timer.kitchen", "description": "منبّه الكوزينة"},
             {"device_name": "timer.salat", "description": "منبّه الصلاة"},
             {"device_name": "todo.shopping_list", "description": "ليستة الشرا"},
             {"device_name": "todo.tasks", "description": "ليستة الخدمة"}]
    return rows


# --------------------------------------------------------------- templated actions
def templated():
    T = []

    def add(dt, svc, phrases, mult=1):
        for p in dict.fromkeys(phrases):
            T.append({"device_type": dt, "service": svc, "phrase": p, "multiplier": mult})

    on = imp("turn_on", 4)
    on_dev = imp("turn_on_device", 4)
    off = imp("turn_off", 3)
    opn = imp("open", 2)
    cls = imp("close", 3)
    for dt, onv in (("light", on), ("switch", on_dev), ("fan", on_dev), ("media_player", on_dev)):
        add(dt, "turn_on", [f"{s}{v} <device_name>" for v in onv for s in ("", "حگل ")] +
            [f"{v} <device_name>{p}" for v in onv[:2] for p in SOFT_POST[2:]] +
            [f"نبغيك {V['turn_on']['req'][0]} <device_name>"], 2 if dt == "light" else 1)
        add(dt, "turn_off", [f"{s}{v} <device_name>" for v in off for s in ("", "حگل ")] +
            [f"{v} <device_name> حگل" for v in off[:1]] + [f"نبغيك {V['turn_off']['req'][0]} <device_name>"],
            2 if dt == "light" else 1)
        add(dt, "toggle", ["بدّل <device_name>", "ردّ <device_name> للحالة الاخرى", "<device_name>، اگديه الا كان طافي واطفيه الا كان گادي"])
    add("light", "turn_on", [f"{v} <device_name> ف <brightness>%" for v in ["دير", "ردّ", "اعدل"]] +
        ["<device_name> <brightness> فالمية", "حگل دير <device_name> على <brightness> فالمية"], 2)
    add("light", "turn_on", [f"{v} <device_name> <color>" for v in ["دير", "ردّ", "اعدل"]] +
        ["حگل ردّ <device_name> بلون <color>"], 2)
    add("fan", "increase_speed", ["زيد <device_name>", "زيد سرعة <device_name>", "<device_name> زيدها شوي", "حگل زيد <device_name>"])
    add("fan", "decrease_speed", ["نگّص <device_name>", "نگّص سرعة <device_name>", "<device_name> نگّصها شوي", "حگل نگّص <device_name>"])
    for dt in ("blinds", "garage_door"):
        add(dt, "open_cover", [f"{s}{v} <device_name>" for v in opn for s in ("", "حگل ")] + ["نبغيك تفتح لي <device_name>"])
        add(dt, "close_cover", [f"{s}{v} <device_name>" for v in cls for s in ("", "حگل ")] + ["اگفل <device_name> حگل"])
        add(dt, "stop_cover", ["وگّف <device_name>", "وگّف <device_name> ف بلاصتو", "حگل وگّف <device_name>"])
        add(dt, "toggle", ["بدّل <device_name>", "الا كان <device_name> مفتوح اگفلو والا افتحو"])
    add("lock", "lock", ["اگفل <device_name>", "اگفل <device_name> حگل", "اگفل <device_name> بالمفتاح", "سكّر <device_name>", "نبغيك تگفل لي <device_name>"])
    add("lock", "unlock", ["افتح <device_name>", "افتح <device_name> حگل", "حلّ <device_name>", "نبغيك تفتح لي <device_name>"])
    add("media_player", "media_play", ["كمّل <device_name>", "ردّ <device_name> يخدم", "حگل كمّل <device_name>"])
    add("media_player", "media_pause", ["حبّس <device_name> شوي", "حبّس <device_name>", "وگّف <device_name> شوي"])
    add("media_player", "media_stop", ["وگّف <device_name>", "وگّف <device_name> گاع"])
    add("media_player", "media_next_track", ["الّي بعدو ف <device_name>", "دوّر للّي بعدو ف <device_name>"])
    add("media_player", "media_previous_track", ["ردّ للّي گبل ف <device_name>", "رجّع الّي فات ف <device_name>"])
    add("media_player", "volume_up", ["زيد صوت <device_name>", "طلّع صوت <device_name> شوي", "ما نسمع شي، زيد صوت <device_name>"])
    add("media_player", "volume_down", ["نگّص صوت <device_name>", "صوت <device_name> عالي ياسر، نگّصو", "هبّط صوت <device_name>"])
    add("media_player", "volume_mute", ["سكّت <device_name>", "گطع صوت <device_name>", "حگل سكّت <device_name>"])
    add("climate", "set_temperature", ["دير <device_name> على <temp_c> درجة", "اعدل <device_name> ف <temp_c> درجة",
                                       "نبغي <device_name> <temp_c> درجة", "حگل ردّ <device_name> <temp_c>",
                                       "<device_name> على <temp_c> الله يخليك"], 2)
    add("vacuum", "start", ["خلّي <device_name> ينظّف", "اگدي <device_name>", "حگل خلّي <device_name> يشطّب الدار"])
    add("vacuum", "stop", ["وگّف <device_name>", "حبّس <device_name>"])
    add("vacuum", "pause", ["حبّس <device_name> شوي", "خلّي <device_name> يصبر شوي"])
    add("vacuum", "return_to_base", ["ردّ <device_name> لبلاصتو", "رجّع <device_name> للشارجور"])
    add("timer", "start", ["دير <device_name> ل <duration>", "حطّ <device_name> <duration>", "نبّهني بعد <duration> ب <device_name>",
                           "اعدل <device_name> ل <duration> حگل"], 2)
    add("timer", "cancel", ["الغي <device_name>", "وگّف <device_name>", "اگطع <device_name>"])
    add("todo", "add_item", ["زيد <todo> ل <device_name>", "اكتب <todo> ف <device_name>", "حگل زيد <todo> ف <device_name>",
                             "ما ننساو <todo>، زيدو ف <device_name>"], 2)
    # two-device commands (short responses)
    for a, sa, b, sb, p in [("light", "turn_off", "fan", "turn_off", "طفّي <device_name1> و<device_name2>"),
                            ("light", "turn_on", "blinds", "close_cover", "اگدي <device_name1> واگفل <device_name2>"),
                            ("media_player", "turn_off", "light", "turn_off", "طفّي <device_name1> و<device_name2> حگل"),
                            ("lock", "lock", "light", "turn_off", "اگفل <device_name1> وطفّي <device_name2>"),
                            ("blinds", "open_cover", "light", "turn_off", "افتح <device_name1> وطفّي <device_name2>")]:
        T.append({"device_type": f"{a}|{b}", "service": f"{sa}|{sb}", "phrase": p, "multiplier": 1})
    return T


# --------------------------------------------------------------- responses
RESP = {  # service: (starting, confirmed) variants; <vars> must match the template's vars
    "light.turn_on": [("ايوه، لاهي نگدي <device_name>.", "گدّيت <device_name>."), ("ايوه، نگدي <device_name> ظرك.", "<device_name> گادي ظرك."),
                      ("ايوه، هاني نگدي <device_name>.", "گدّيت <device_name>.")],
    "light.turn_off": [("ايوه، لاهي نطفّي <device_name>.", "طفّيت <device_name>."), ("ايوه، نطفّي <device_name> ظرك.", "<device_name> مطفي ظرك.")],
    "light.toggle": [("ايوه، نبدّل <device_name>.", "بدّلت <device_name>.")],
    "switch.turn_on": [("ايوه، لاهي نگدي <device_name>.", "گدّيت <device_name>."), ("ايوه، نگدي <device_name>.", "گدّيت <device_name>.")],
    "switch.turn_off": [("ايوه، لاهي نطفّي <device_name>.", "طفّيت <device_name>.")],
    "switch.toggle": [("ايوه، نبدّل <device_name>.", "بدّلت <device_name>.")],
    "fan.turn_on": [("ايوه، نگدي <device_name>.", "گدّيت <device_name>.")],
    "fan.turn_off": [("ايوه، نطفّي <device_name>.", "طفّيت <device_name>.")],
    "fan.toggle": [("ايوه، نبدّل <device_name>.", "بدّلت <device_name>.")],
    "fan.increase_speed": [("ايوه، نزيد <device_name>.", "زدت سرعة <device_name>.")],
    "fan.decrease_speed": [("ايوه، ننگّص <device_name>.", "نگّصت سرعة <device_name>.")],
    "media_player.turn_on": [("ايوه، نگدي <device_name>.", "گدّيت <device_name>.")],
    "media_player.turn_off": [("ايوه، نطفّي <device_name>.", "طفّيت <device_name>.")],
    "media_player.toggle": [("ايوه، نبدّل <device_name>.", "بدّلت <device_name>.")],
    "media_player.media_play": [("ايوه، نكمّل <device_name>.", "<device_name> رجع يخدم.")],
    "media_player.media_pause": [("ايوه، نحبّس <device_name>.", "حبّست <device_name>.")],
    "media_player.media_stop": [("ايوه، نوگّف <device_name>.", "وگّفت <device_name>.")],
    "media_player.media_next_track": [("ايوه، الّي بعدو ف <device_name>.", "هاك الّي بعدو ف <device_name>.")],
    "media_player.media_previous_track": [("ايوه، نردّ للّي گبل ف <device_name>.", "رجّعت للّي گبل ف <device_name>.")],
    "media_player.volume_up": [("ايوه، نزيد صوت <device_name>.", "زدت صوت <device_name>.")],
    "media_player.volume_down": [("ايوه، ننگّص صوت <device_name>.", "نگّصت صوت <device_name>.")],
    "media_player.volume_mute": [("ايوه، نسكّت <device_name>.", "سكّتت <device_name>.")],
    "blinds.open_cover": [("ايوه، نفتح <device_name>.", "فتحت <device_name>.")],
    "blinds.close_cover": [("ايوه، نگفل <device_name>.", "گفلت <device_name>.")],
    "blinds.stop_cover": [("ايوه، نوگّف <device_name>.", "وگّفت <device_name>.")],
    "blinds.toggle": [("ايوه، نبدّل <device_name>.", "بدّلت <device_name>.")],
    "garage_door.open_cover": [("ايوه، نفتح <device_name>.", "فتحت <device_name>.")],
    "garage_door.close_cover": [("ايوه، نگفل <device_name>.", "گفلت <device_name>.")],
    "garage_door.stop_cover": [("ايوه، نوگّف <device_name>.", "وگّفت <device_name>.")],
    "garage_door.toggle": [("ايوه، نبدّل <device_name>.", "بدّلت <device_name>.")],
    "lock.lock": [("ايوه، نگفل <device_name>.", "گفلت <device_name>."), ("ايوه، لاهي نگفل <device_name>.", "<device_name> مگفول ظرك.")],
    "lock.unlock": [("ايوه، نفتح <device_name>.", "فتحت <device_name>.")],
    "climate.set_temperature": [("ايوه، ندير <device_name> على <temp_c> درجة.", "<device_name> ظرك على <temp_c> درجة.")],
    "vacuum.start": [("ايوه، <device_name> لاهي يبدا ينظّف.", "<device_name> بدا ينظّف.")],
    "vacuum.stop": [("ايوه، نوگّف <device_name>.", "وگّفت <device_name>.")],
    "vacuum.pause": [("ايوه، نحبّس <device_name>.", "حبّست <device_name>.")],
    "vacuum.return_to_base": [("ايوه، نردّ <device_name> لبلاصتو.", "<device_name> راجع لبلاصتو.")],
    "timer.start": [("ايوه، ندير <device_name> ل <duration>.", "<device_name> بدا: <duration>.")],
    "timer.cancel": [("ايوه، نلغي <device_name>.", "الغيت <device_name>.")],
    "todo.add_item": [("ايوه، نزيد <todo> ل <device_name>.", "زدت <todo> ل <device_name>.")],
}
VAR_RESP = {  # extra variants that carry template vars
    ("light.turn_on", "brightness"): [("ايوه، ندير <device_name> ف <brightness>%.", "<device_name> ظرك ف <brightness>%.")],
    ("light.turn_on", "color"): [("ايوه، نردّ <device_name> <color>.", "<device_name> ظرك <color>.")],
}


def responses():
    rows = []
    for svc, pairs in RESP.items():
        for s, c in pairs:
            for short in (0, 1):
                if short:
                    s, c = s.replace("لاهي ", ""), c
                rows.append({"service": svc, "response_starting": s, "response_confirmed": c, "persona": "assistant", "short": short})
    for (svc, _), pairs in VAR_RESP.items():
        for s, c in pairs:
            rows.append({"service": svc, "response_starting": s, "response_confirmed": c, "persona": "assistant", "short": 0})
    return rows


# --------------------------------------------------------------- status requests
def status():
    S = []
    for dt in ("light", "fan", "switch", "media_player"):
        S += [{"device_type": dt, "state": "on", "phrase": p, "assistant_response": r} for p, r in [
            ("ياك <device_name> گادي؟", "ايوه، <device_name> گادي ظرك."), ("<device_name> گادي ولّا طافي؟", "<device_name> گادي."), ("<device_name> گادي ولّا مطفي؟", "<device_name> گادي."),
            ("شوف لي <device_name>", "<device_name> گادي ظرك."), ("<device_name> كيف حالو؟", "<device_name> گادي.")]]
        S += [{"device_type": dt, "state": "off", "phrase": p, "assistant_response": r} for p, r in [
            ("ياك <device_name> گادي؟", "لا، <device_name> مطفي."), ("<device_name> گادي ولّا طافي؟", "<device_name> مطفي."),
            ("شوف لي <device_name>", "<device_name> مطفي ظرك.")]]
    S += [{"device_type": "light", "state": "on;<color>;<brightness>", "phrase": "<device_name> كيف حالو؟",
           "assistant_response": "<device_name> گادي، لونو <color> وف <brightness>%."}]
    for dt in ("blinds", "garage_door"):
        S += [{"device_type": dt, "state": "open", "phrase": p, "assistant_response": r} for p, r in [
            ("<device_name> مفتوح؟", "ايوه، <device_name> مفتوح."), ("شوف لي <device_name>", "<device_name> مفتوح ظرك.")]]
        S += [{"device_type": dt, "state": "closed", "phrase": p, "assistant_response": r} for p, r in [
            ("<device_name> مفتوح؟", "لا، <device_name> مگفول."), ("ياك <device_name> مگفول؟", "ايوه، <device_name> مگفول.")]]
    S += [{"device_type": "lock", "state": "locked", "phrase": p, "assistant_response": r} for p, r in [
        ("ياك <device_name> مگفول؟", "ايوه، <device_name> مگفول."), ("<device_name> مگفول ولّا مفتوح؟", "<device_name> مگفول.")]]
    S += [{"device_type": "lock", "state": "unlocked", "phrase": p, "assistant_response": r} for p, r in [
        ("ياك <device_name> مگفول؟", "لا، <device_name> مفتوح. تبغي نگفلو؟"), ("<device_name> مگفول ولّا مفتوح؟", "<device_name> مفتوح.")]]
    S += [{"device_type": "media_player", "state": "playing;<media>;<volume>", "phrase": "شنه يخدم ف <device_name>؟",
           "assistant_response": "<device_name> يخدم <media>، والصوت ف <volume>."},
          {"device_type": "media_player", "state": "paused", "phrase": "<device_name> يخدم؟", "assistant_response": "<device_name> محبوس ظرك."},
          {"device_type": "timer", "state": "active;<duration>;<remaining>", "phrase": "كم باگي ف <device_name>؟",
           "assistant_response": "<device_name> امدّتو <duration>، وباگي <remaining>."}]
    return S


def refusals():
    rows = []
    for svc, dev, fn, phrase in [("lock.lock", "bab_el_kouzina", "باب الكوزينة", "اگفل <device_name>"),
                                 ("light.turn_on", "dhaw_es_stah", "ظو السطح", "اگدي <device_name>"),
                                 ("fan.turn_on", "marwaha_el_khayma", "مروحة الخيمة", "اگدي <device_name> حگل"),
                                 ("climate.set_temperature", "klim_el_hawch", "كليم الحوش", "طفّي <device_name>"),
                                 ("switch.turn_on", "pompe_es_stah", "بومبة السطح", "اگدي <device_name>"),
                                 ("blinds.open_cover", "rideau_el_bureau", "ريدو البيرو", "افتح <device_name>")]:
        rows.append({"reason_type": "not_available", "service_name": svc, "device_name": dev, "friendly_name": fn, "desired_state": "",
                     "phrase": phrase, "response": f"ما لگيت حتى {fn} ف الدار."})
    for svc, dev, fn, st, phrase, resp in [("light.turn_on", "dhaw_es_salon", "ظو الصالون", "on", "اگدي <device_name>", "ظو الصالون گادي من گبيل."),
                                           ("light.turn_off", "dhaw_el_kouzina", "ظو الكوزينة", "off", "طفّي <device_name>", "ظو الكوزينة مطفي من گبيل."),
                                           ("lock.lock", "bab_ed_dar", "باب الدار", "locked", "اگفل <device_name>", "باب الدار مگفول من گبيل."),
                                           ("switch.turn_off", "chauffe_eau", "شوفو بيت الما", "off", "طفّي <device_name>", "شوفو بيت الما مطفي من گبيل.")]:
        rows.append({"reason_type": "already_state", "service_name": svc, "device_name": dev, "friendly_name": fn, "desired_state": st,
                     "phrase": phrase, "response": resp})
    return rows


def failed_calls():
    rows = []
    for svc, good, fn, bad, phrase in [("light.turn_on", "light.salon_dhaw", "ظو الصالون", "light.salon_daw", "اگدي <device_name>"),
                                       ("fan.turn_off", "fan.bedroom_marwaha", "مروحة بيت الرگاد", "fan.bedrom_marwaha", "طفّي <device_name>"),
                                       ("lock.lock", "lock.bab_ed_dar", "باب الدار", "lock.bab_dar", "اگفل <device_name>"),
                                       ("switch.turn_on", "switch.yard_pompe", "بومبة الحوش", "switch.yard_pomp", "اگدي <device_name> حگل")]:
        rows.append({"service_name": svc, "correct_device_name": good, "correct_friendly_name": fn, "bad_device_name": bad, "phrase": phrase,
                     "error_result": f"Error: entity {bad} not found", "retry_prompt": f"نعاود ب {fn}."})
    return rows


def write(name, rows, fields=None):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    write("pile_of_device_names.csv", device_rows())
    write("pile_of_templated_actions.csv", templated())
    write("pile_of_responses.csv", responses())
    write("pile_of_status_requests.csv", status())
    write("pile_of_refusals.csv", refusals())
    write("pile_of_failed_tool_calls.csv", failed_calls())
    # specific actions: concrete phrasings using a real friendly name
    spec = []
    for dt, svc, verbs in (("light", "turn_on", imp("turn_on", 3)), ("light", "turn_off", imp("turn_off", 2)),
                           ("fan", "turn_on", imp("turn_on_device", 2)), ("blinds", "close_cover", imp("close", 2))):
        word, stem = DEV_TYPES[dt][0]
        for aid, aname in AREAS:
            spec.append({"service_name": f"{dt}.{svc}", "device_name": f"{aid}_{stem}", "phrase": f"{R.choice(verbs)} {word} {aname}{R.choice(SOFT_POST)}"})
    # home-llm derives the friendly name of a specific action from its ASCII id ("Salon Dhaw"),
    # which would pair Arabic phrases with Latin names, so this pile is left header-only.
    del spec
    write("pile_of_specific_actions.csv", [], fields=["service_name", "device_name", "phrase"])
    write("pile_of_system_prompts.csv", [{"persona": "assistant", "prompt":
          "انت «دار»، مساعد ذكي يتحكم ف آلات الدار. اعدل الخدمة الّي يطلبها صاحب الدار ولّا جاوب على سؤالو بالمعلومات الّي عندك وبس. "
          "جاوب بالحسانية، كلام قصير."}])
    durs = [("00:01:00", 1, "minute"), ("00:02:00", 2, "minute"), ("00:03:00", 3, "minute"), ("00:05:00", 5, "minute"),
            ("00:07:00", 7, "minute"), ("00:10:00", 10, "minute"), ("00:15:00", 15, "minute"), ("00:20:00", 20, "minute"),
            ("00:30:00", 30, "minute"), ("00:45:00", 45, "minute"), ("01:00:00", 1, "hour"), ("02:00:00", 2, "hour"),
            ("00:00:30", 30, "second")]
    write("pile_of_durations.csv", [{"duration": d, "name": {"00:30:00": "نص ساعة", "00:15:00": "ربع ساعة"}.get(d) or count_with_unit(n, u, random.Random(1))}
                                    for d, n, u in durs])
    (OUT / "pile_of_and_words.csv").write_text("و\nومن بعد\n", encoding="utf-8")
    (OUT / "pile_of_todo_items.txt").write_text("\n".join([i["ar"] for i in LEX["shopping_items"]] +
                                                          ["نعيّط لخالي", "نخلّص الكوران", "نشري لكريدي", "نغسل الوطو", "نمشي للسوگ"]) + "\n", encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT), sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
