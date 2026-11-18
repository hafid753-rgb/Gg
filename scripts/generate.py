#!/usr/bin/env python3
"""Generate the Hassaniya Home-Assistant dataset from data/lexicon/lexicon.yaml.

Outputs (all under data/generated/):
  nlu/{train,dev,test}.jsonl   utterance -> HA intent + slots (+ noisy ASR & Arabizi variants)
  sft/{train,dev,test}.jsonl   chat/tool-calling conversations grounded in a random smart home,
                               with dynamic, state-aware Hassaniya replies (for LoRA fine-tuning)
  asr/recording_prompts.csv    diverse prompts for native speakers to record (Whisper fine-tune)
  stats.json                   counts per intent / split
and homeassistant/custom_sentences/ar/hassaniya.yaml (fast local path, via hassil).

    python3 scripts/generate.py --nlu 30000 --sft 15000 --seed 7
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hassaniya import (  # noqa: E402
    MONTHS, ROOT, WEEKDAYS, asr_noise, count_with_unit, load_lexicon, normalize,
    obj_suffix, pp, say_number, say_time, strip_al, strip_tashkeel, to_arabizi,
)

LEX = load_lexicon()
OUT = ROOT / "data" / "generated"
TOOLS = json.loads((ROOT / "data" / "schema" / "ha_tools.json").read_text(encoding="utf-8"))

SYSTEM_PROMPT = (
    "You are \"Dar\" (دار), a local voice assistant for Home Assistant in a Mauritanian home.\n"
    "The user speaks Hassaniya Arabic (Mauritanian dialect), sometimes mixed with French words, "
    "and the text comes from speech recognition so it may contain spelling errors.\n"
    "Always answer in short, natural Hassaniya written in Arabic script. "
    "Use the tools to control devices; never claim an action you did not perform. "
    "If a request is ambiguous ask one short question. If a device does not exist, say so.\n"
)


# =============================================================== primitives
def pick(seq, rng):
    return rng.choice(seq)


def verb(key: str, rng: random.Random, style: str | None = None) -> tuple[str, str]:
    v = LEX["verbs"][key]
    style = style or rng.choices(["m", "f", "pl", "req", "want"], [55, 14, 5, 14, 12])[0]
    if style == "m":
        return pick(v["imp_m"], rng), style
    if style == "f":
        return pick(v["imp_f"], rng), style
    if style == "pl":
        return pick(v["imp_pl"], rng), style
    if style == "req":
        return pick(v["req"], rng), style
    return f"{pick(LEX['want']['ar'], rng)} {pick(v['req'], rng)}", style


def wrap(core: str, rng: random.Random, question: bool = False) -> str:
    parts = []
    if rng.random() < 0.10:
        parts.append(pick(LEX["fillers"]["ar"], rng) + "،")
    if rng.random() < 0.08:
        parts.append(pick(LEX["wake_words"]["ar"], rng) + "،")
    if rng.random() < 0.08:
        parts.append(pick(LEX["politeness"]["pre"], rng))
    soft = rng.random() < 0.15 and not question  # حگل: confirmed very natural, before or after the command
    if soft and rng.random() < 0.5:
        parts.append("حگل")
    parts.append(core)
    if soft and "حگل" not in parts:
        parts.append("حگل")
    if rng.random() < 0.10 and not question:
        parts.append(pick(LEX["now"]["ar"], rng))
    if rng.random() < 0.10:
        parts.append(pick(LEX["politeness"]["post"], rng))
    s = " ".join(parts)
    return s + ("؟" if question else "")


def in_area(area: dict, rng) -> str:
    return f"{pick(LEX['in']['ar'], rng)} {pick(area['ar'], rng)}"


def dev_items(domain_key: str):
    return LEX["devices"][domain_key]["items"]


AREAS = LEX["areas"]
AREA_BY_ID = {a["id"]: a for a in AREAS}


@dataclass
class Dev:
    key: str          # lexicon domain key (light, fan, ...)
    domain: str       # HA domain
    item: dict        # lexicon item
    surface: str = ""  # how it was said

    @property
    def g(self):
        return self.item.get("g", "m")

    @property
    def canon(self):
        return self.item["ar"][0]


def rand_dev(rng, keys=None) -> Dev:
    keys = keys or list(LEX["devices"].keys())
    k = pick(keys, rng)
    item = pick(dev_items(k), rng)
    return Dev(k, LEX["devices"][k]["domain"], item)


@dataclass
class Utt:
    text: str
    intent: str
    slots: dict
    kind: str = "command"          # command | query | implicit | multi | social | oos | clarify
    fast_path: bool = True         # expected to be handled by HA sentence matching
    template: str = ""
    calls: list = field(default_factory=list)   # [(tool_name, args)]
    meta: dict = field(default_factory=dict)


# =============================================================== utterances
ACTION_VERB = {
    ("on", "light"): "turn_on", ("on", "other"): "turn_on_device", ("off", None): "turn_off",
    ("open", None): "open", ("close", None): "close",
}


def onoff_verb(action: str, dev: Dev) -> str:
    if action == "on":
        return "turn_on" if dev.key == "light" else "turn_on_device"
    if action == "off":
        return "turn_off"
    return {"open": "open", "close": "close", "lock": "close", "unlock": "open"}[action]


def utt_onoff(action: str, dev: Dev, area: dict | None, rng, all_flag=False, plain=False) -> Utt:
    if not dev.canon.startswith("ال"):
        area = None  # fixed-name device, e.g. "باب الگاراج"
    vkey = onoff_verb(action, dev)
    vs, style = verb(vkey, rng)
    d = pick(dev.item["ar"], rng)
    dev.surface = d
    loc = f" {in_area(area, rng)}" if area else ""
    tpl = rng.random()
    single_word_verb = " " not in vs and style in ("m", "f", "pl")
    if action == "lock":
        core = f"{vs} {d}{loc} {pick(['بالمفتاح', 'بالساروت', ''], rng)}".strip()
        tid = "lock"
    elif all_flag:
        core = f"{vs} {d} {pick(LEX['all']['ar'], rng)}{loc}" if rng.random() < 0.6 else \
            f"{vs} {pick(LEX['all']['ar'], rng)} {d}{loc}"
        tid = "all"
    elif area and tpl < 0.15:   # construct state: "اشعل ضو الصالون"
        core = f"{vs} {strip_al(d)} {pick(area['ar'], rng)}"
        tid = "construct"
    elif single_word_verb and tpl < 0.27:  # topicalized: "الضو ف الصالون اشعلو"
        core = f"{d}{loc} {vs}{obj_suffix(dev.g) if style != 'pl' else ''}"
        tid = "topic"
    else:
        core = f"{vs} {d}{loc}"
        tid = "basic"
    intent = "HassTurnOn" if action in ("on", "open", "lock") else "HassTurnOff"
    slots = {"domain": dev.domain}
    if area:
        slots["area"] = area["ar"][0]
    return Utt(core if plain else wrap(core, rng), intent, slots, template=f"onoff.{action}.{tid}",
               meta={"dev": dev, "area": area, "action": action})


def utt_brightness(dev: Dev, area, rng, value=None) -> Utt:
    n = value if value is not None else pick([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100], rng)
    vs, _ = verb("set", rng)
    d = pick(dev.item["ar"], rng)
    loc = f" {in_area(area, rng)}" if area else ""
    unit = pick(LEX["units"]["percent"][:3], rng)
    core = pick([
        f"{vs} {d}{loc} {pick(['ف', 'على', 'ل'], rng)} {say_number(n, LEX, rng)} {unit}",
        f"{d}{loc} {say_number(n, LEX, rng)} {unit}",
        f"{vs} {d}{loc} {say_number(n, LEX, rng)} {unit}",
    ], rng)
    slots = {"brightness": n}
    if area:
        slots["area"] = area["ar"][0]
    return Utt(wrap(core, rng), "HassLightSet", slots, template="light.brightness",
               meta={"dev": dev, "area": area})


def utt_color(dev: Dev, area, rng) -> Utt:
    c = pick(LEX["colors"], rng)
    vs, _ = verb("set", rng)
    d = pick(dev.item["ar"], rng)
    loc = f" {in_area(area, rng)}" if area else ""
    cw = pick(c["ar"], rng)
    core = pick([f"{vs} {d}{loc} {cw}", f"{vs} {d}{loc} بلون {cw}", f"{vs} {d}{loc} {cw} الله يخليك"], rng)
    slots = {"color": c["en"]}
    if area:
        slots["area"] = area["ar"][0]
    return Utt(wrap(core, rng), "HassLightSet", slots, template="light.color",
               meta={"dev": dev, "area": area, "color_ar": c["ar"][0]})


def utt_relative_light(dev, area, rng, up: bool) -> Utt:
    vs, _ = verb("increase" if up else "decrease", rng)
    amt = pick(LEX["a_little"]["ar"] + LEX["very"]["ar"] + [""], rng)
    d = pick(dev.item["ar"], rng)
    loc = f" {in_area(area, rng)}" if area else ""
    core = pick([f"{vs} {d}{loc} {amt}", f"{d}{loc} {vs}{obj_suffix(dev.g)} {amt}" if " " not in vs else f"{vs} {d}{loc}"], rng)
    step = 40 if amt in LEX["very"]["ar"] else 20
    return Utt(wrap(core.strip(), rng), "HassLightSet", {"brightness_step": step if up else -step},
               kind="relative", fast_path=False, template="light.relative",
               meta={"dev": dev, "area": area, "step": step if up else -step})


def utt_climate_set(area, rng, value=None) -> Utt:
    n = value if value is not None else rng.randint(16, 28)
    dev = Dev("climate", "climate", pick(dev_items("climate"), rng))
    vs, _ = verb("set", rng)
    d = pick(dev.item["ar"], rng)
    loc = f" {in_area(area, rng)}" if area else ""
    deg = pick(LEX["units"]["degree"] + [""], rng)
    core = pick([
        f"{vs} {d}{loc} {pick(['ف', 'على'], rng)} {say_number(n, LEX, rng)} {deg}",
        f"{vs} {d}{loc} {say_number(n, LEX, rng)} {deg}",
        f"{pick(LEX['want']['ar'], rng)} السخانة{loc} {say_number(n, LEX, rng)} {deg}",
        f"{d}{loc} {say_number(n, LEX, rng)} {deg}",
    ], rng)
    slots = {"temperature": n}
    if area:
        slots["area"] = area["ar"][0]
    return Utt(wrap(core.strip(), rng), "HassClimateSetTemperature", slots,
               template="climate.set", meta={"dev": dev, "area": area})


def utt_climate_get(area, rng) -> Utt:
    loc = f" {in_area(area, rng)}" if area else ""
    hm = pick(LEX["q"]["how_much"]["ar"], rng)
    core = pick([f"{hm} السخانة{loc}", f"السخانة{loc} {hm}", f"ش السخانة{loc} {pick(LEX['now']['ar'], rng)}",
                 f"{pick(LEX['verbs']['tell']['imp_m'], rng)} السخانة{loc}",
                 f"{hm} درجة الحرارة{loc}"], rng)
    slots = {"area": area["ar"][0]} if area else {}
    return Utt(wrap(core, rng, question=True), "HassClimateGetTemperature", slots,
               kind="query", template="climate.get", meta={"area": area})


def utt_get_state(dev: Dev, area, rng) -> Utt:
    st = {"light": "on", "fan": "on", "climate": "on", "media_player": "on", "switch": "on",
          "cover": pick(["open", "closed"], rng), "lock": pick(["locked", "unlocked"], rng),
          "vacuum": "on"}[dev.key]
    if dev.key in ("lock", "vacuum") or not dev.canon.startswith("ال"):
        area = None
    d = pick(dev.item["ar"], rng)
    loc = f" {in_area(area, rng)}" if area else ""
    p = pp(st, dev.g)
    if st == "on" and rng.random() < 0.25:  # شاعل is also understood; گادي is the preferred form
        p = "شاعلة" if dev.g == "f" else "شاعل"
    if rng.random() < 0.25 and dev.key in ("light", "fan", "switch"):
        core = f"ش من {strip_al(d)} {p}{loc}"
        tid = "which"
    else:
        core = pick([f"{pick(LEX['q']['is_it']['ar'], rng)} {d}{loc} {p}", f"{d}{loc} {p}",
                     f"{pick(LEX['verbs']['check']['imp_m'], rng)} {pick(LEX['q']['is_it']['ar'], rng)} {d}{loc} {p}"], rng)
        tid = "yesno"
    slots = {"domain": dev.domain, "state": st}
    if area:
        slots["area"] = area["ar"][0]
    return Utt(wrap(core, rng, question=True), "HassGetState", slots, kind="query",
               template=f"state.{tid}", meta={"dev": dev, "area": area, "state": st})


def utt_cover_position(area, rng) -> Utt:
    dev = Dev("cover", "cover", dev_items("cover")[0])  # curtains
    d = pick(dev.item["ar"], rng)
    loc = f" {in_area(area, rng)}" if area else ""
    if rng.random() < 0.4:
        n = 50
        vs, _ = verb("open", rng)
        core = f"{vs} {d}{loc} {pick(['النصّ', 'انصّو', 'نصّ'], rng)}"
    else:
        n = pick([10, 20, 25, 30, 40, 60, 70, 75, 80, 90], rng)
        vs, _ = verb("set", rng)
        core = f"{vs} {d}{loc} {pick(['ف', 'على'], rng)} {say_number(n, LEX, rng)} {pick(LEX['units']['percent'][:3], rng)}"
    slots = {"position": n}
    if area:
        slots["area"] = area["ar"][0]
    return Utt(wrap(core, rng), "HassSetPosition", slots, template="cover.position",
               meta={"dev": dev, "area": area})


def utt_media(rng, dev=None) -> Utt:
    dev = dev or Dev("media_player", "media_player", pick(dev_items("media_player"), rng))
    d = pick(dev.item["ar"], rng)
    which = pick(["pause", "unpause", "next", "previous", "volume", "mute", "unmute"], rng)
    if which == "mute":
        core = pick([f"سكّت {d}", f"گطع صوت {strip_al(d)}", "سكّت الصوت", f"{d} سكّتها"], rng)
        return Utt(wrap(core, rng), "HassMediaPlayerMute", {}, template="media.mute", meta={"dev": dev})
    if which == "unmute":
        core = pick([f"ردّ صوت {strip_al(d)}", "ردّ الصوت", f"خلّي {d} يتكلم"], rng)
        return Utt(wrap(core, rng), "HassMediaPlayerUnmute", {}, template="media.unmute", meta={"dev": dev})
    if which == "pause":
        core = pick([f"{pick(['حبّس', 'وگّف'], rng)} {d} {pick(LEX['a_little']['ar'], rng)}",
                     f"حبّس {pick([d, 'الموسيقى', 'الغنا', 'المديح'], rng)}", "اعدل pause"], rng)
        return Utt(wrap(core, rng), "HassMediaPause", {}, template="media.pause", meta={"dev": dev})
    if which == "unpause":
        core = pick([f"كمّل {pick([d, 'الموسيقى', 'المديح'], rng)}", f"ردّ {d} يخدم", "خلّيه يكمّل",
                     f"{pick(['شعّل', 'حلّ'], rng)} الموسيقى من جديد"], rng)
        return Utt(wrap(core, rng), "HassMediaUnpause", {}, template="media.unpause", meta={"dev": dev})
    if which == "next":
        core = pick(["الّي بعدو", "دوّر للّي بعدو", "الأغنية الجاية", "جيب الّي بعدها", "next الله يخليك"], rng)
        return Utt(wrap(core, rng), "HassMediaNext", {}, template="media.next", meta={"dev": dev})
    if which == "previous":
        core = pick(["ردّ للّي گبل", "الأغنية الّي فاتت", "رجّع الّي گبلها", "ردّ اللّي فات"], rng)
        return Utt(wrap(core, rng), "HassMediaPrevious", {}, template="media.previous", meta={"dev": dev})
    n = pick([10, 20, 25, 30, 40, 50, 60, 70, 80, 100], rng)
    vs, _ = verb("set", rng)
    core = pick([f"{vs} الصوت {pick(['ف', 'على'], rng)} {say_number(n, LEX, rng)}",
                 f"الصوت {say_number(n, LEX, rng)} {pick(LEX['units']['percent'][:2], rng)}",
                 f"{vs} صوت {strip_al(d)} ف {say_number(n, LEX, rng)}"], rng)
    return Utt(wrap(core, rng), "HassSetVolume", {"volume_level": n}, template="media.volume",
               meta={"dev": dev})


def utt_volume_relative(rng, up) -> Utt:
    vs, _ = verb("increase" if up else "decrease", rng)
    amt = pick(LEX["a_little"]["ar"] + LEX["very"]["ar"] + [""], rng)
    core = pick([f"{vs} الصوت {amt}", f"الصوت {pick(['عالي', 'ياسر'], rng)}" if not up else f"{vs} الصوت {amt}",
                 f"ما نسمع شي، {verb('increase', rng)[0]} الصوت" if up else f"{vs} الصوت {amt}"], rng)
    step = (25 if amt in LEX["very"]["ar"] else 10) * (1 if up else -1)
    return Utt(wrap(core.strip(), rng), "HassSetVolume", {"volume_step": step}, kind="relative",
               fast_path=False, template="media.volume_rel", meta={"step": step})


def utt_fan_speed(area, rng) -> Utt:
    dev = Dev("fan", "fan", pick(dev_items("fan"), rng))
    n = pick([20, 25, 30, 40, 50, 60, 70, 75, 80, 100], rng)
    vs, _ = verb("set", rng)
    d = pick(dev.item["ar"], rng)
    loc = f" {in_area(area, rng)}" if area else ""
    core = pick([f"{vs} {d}{loc} {pick(['ف', 'على'], rng)} {say_number(n, LEX, rng)} {pick(LEX['units']['percent'][:3], rng)}",
                 f"{d}{loc} {say_number(n, LEX, rng)} {pick(LEX['units']['percent'][:3], rng)}"], rng)
    slots = {"percentage": n}
    if area:
        slots["area"] = area["ar"][0]
    return Utt(wrap(core, rng), "HassFanSetSpeed", slots, template="fan.speed", meta={"dev": dev, "area": area})


def utt_timer(rng) -> Utt:
    unit = rng.choices(["minute", "hour", "second"], [75, 15, 10])[0]
    n = {"minute": pick([1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 45], rng),
         "hour": pick([1, 2, 3], rng), "second": pick([10, 20, 30, 45], rng)}[unit]
    amount = count_with_unit(n, unit, rng)
    if unit == "minute" and n == 30 and rng.random() < 0.5:
        amount = pick(LEX["units"]["half_hour"], rng)
    if unit == "minute" and n == 15 and rng.random() < 0.5:
        amount = pick(LEX["units"]["quarter_hour"], rng)
    vs, _ = verb("start_timer", rng)
    tw = pick(["منبّه", "تايمر", "عدّاد", "مينوتري"], rng)
    purpose = pick(["", "", " ل أتاي", " ل الروز", " ل الفرن", " ل الصلاة"], rng)
    core = pick([f"{vs} {tw} ل {amount}{purpose}", f"{vs} {tw} {amount}{purpose}",
                 f"نبّهني بعد {amount}{purpose}", f"گول لي بعد {amount}{purpose}",
                 f"{amount}{purpose}، {vs} {tw}"], rng)
    slots = {f"{unit}s": n}
    return Utt(wrap(core, rng), "HassStartTimer", slots, template="timer.start")


def utt_timer_other(rng) -> Utt:
    tw = pick(["المنبّه", "التايمر", "العدّاد"], rng)
    if rng.random() < 0.15:
        vs, _ = verb("cancel", rng)
        core = pick([f"{vs} {pick(['المنبّهات', 'التايمرات'], rng)} {pick(LEX['all']['ar'], rng)}", f"{vs} {pick(LEX['all']['ar'], rng)} المنبّهات"], rng)
        return Utt(wrap(core, rng), "HassCancelAllTimers", {}, template="timer.cancel_all")
    if rng.random() < 0.55:
        vs, _ = verb("cancel", rng)
        return Utt(wrap(f"{vs} {tw}", rng), "HassCancelTimer", {}, template="timer.cancel")
    core = pick([f"{pick(LEX['q']['how_much']['ar'], rng)} باگي ف {tw}", f"{tw} {pick(LEX['q']['how_much']['ar'], rng)} باگي لو",
                 f"{pick(LEX['q']['how_much']['ar'], rng)} باگي من الوگت"], rng)
    return Utt(wrap(core, rng, question=True), "HassTimerStatus", {}, kind="query", template="timer.status")


TIME_Q = ["شنه الساعة", "ش الساعة", "الساعة كم", "كم الساعة ظرك", "گول لي الساعة", "شوف لي الساعة", "الوكت كم"]
DATE_Q = ["ش اليوم من نهار", "اليوم ش من يوم", "ش التاريخ اليوم", "اليوم كم ف الشهر", "گول لي التاريخ", "ذ اليوم ش هو"]
WEATHER_Q = ["كيف الجو اليوم", "ش حالة الجو", "لاهي تصبّ اليوم", "فم عجاج اليوم", "السخانة كم برّا", "كيف الجو ظرك", "كيف الطقس", "واش الدنيا سخونة برّا", "ياك ما لاهي تجي الريح", "شوف لي الجو اليوم"]
NEVERMIND_Q = ["خلّيه", "ماشي شي", "انساه", "لا لا ماهو ذاك", "سامحني ماني باغي شي", "خلاص", "ما عندي شي", "لا خلّي عنّك"]


def utt_time_date_weather(rng) -> Utt:
    w = pick(["time", "time", "date", "weather", "weather"], rng)
    if w == "time":
        core = pick(TIME_Q, rng)
        return Utt(wrap(core, rng, question=True), "HassGetCurrentTime", {}, kind="query", template="time")
    if w == "date":
        core = pick(DATE_Q, rng)
        return Utt(wrap(core, rng, question=True), "HassGetCurrentDate", {}, kind="query", template="date")
    core = pick(WEATHER_Q, rng)
    return Utt(wrap(core, rng, question=True), "HassGetWeather", {}, kind="query", template="weather")


def utt_shopping(rng) -> Utt:
    it = pick(LEX["shopping_items"], rng)
    vs, _ = verb("add", rng)
    lst = pick(["ليستة الشرا", "لاليست", "ليستة التسوّگ", "ليستة لكورس", "الليستة"], rng)
    core = pick([f"{vs} {it['ar']} ل {lst}", f"{vs} {it['ar']} ف {lst}", f"{it['ar']} {vs}ه ف {lst}" if " " not in vs else f"{vs} {it['ar']}",
                 f"ما ننساو {it['ar']}، {vs}ه ف {lst}" if " " not in vs else f"{vs} {it['ar']} ف {lst}"], rng)
    return Utt(wrap(core, rng), "HassShoppingListAddItem", {"item": it["ar"]}, template="shopping.add")


def utt_vacuum(rng) -> Utt:
    if rng.random() < 0.6:
        vs, _ = verb("clean", rng)
        core = pick([f"{vs} الدار", f"خلّي الروبو {pick(['ينظّف', 'يشطّب'], rng)}", f"{pick(['حلّ', 'شعّل'], rng)} الروبو",
                     f"{vs} {pick(AREAS, rng)['ar'][0]}"], rng)
        return Utt(wrap(core, rng), "HassVacuumStart", {}, template="vacuum.start")
    vs, _ = verb("return", rng)
    core = pick([f"{vs} الروبو {pick(['لبلاصتو', 'لامكانو', 'للشارجور'], rng)}", f"{vs} المكنسة للشارجور",
                 "خلّي الروبو يرجع لبلاصتو"], rng)
    return Utt(wrap(core, rng), "HassVacuumReturnToBase", {}, template="vacuum.return")


def utt_nevermind(rng) -> Utt:
    core = pick(NEVERMIND_Q, rng)
    return Utt(core, "HassNevermind", {}, kind="social", template="nevermind")


SOCIAL = {
    "greet": (["السلام عليكم", "اشطاري", "كيف اصبحت", "لاباس؟", "ش اخبارك", "صباح الخير", "مساء الخير", "كيف حالك"],
              ["وعليكم السلام، لاباس الحمد لله. ش نعدل لك؟", "لاباس، الحمد لله. ش تبغي نعدل لك؟",
               "الحمد لله، ما فم باس. انا هون، شنه نعدل لك؟", "صباح النور، لباس. شطاري عندك؟"]),
    "thanks": (["شكرا", "الله يجازيك بالخير", "بارك الله فيك", "يعطيك العافية", "مرسي"],
               ["العفو، ذاك واجب.", "مرحبا بيك ف أي وكت.", "ما عدلت شي، ذاك خدمتي.", "وفيك بارك الله."]),
    "who": (["شكون انت", "انت ش", "ش اسمك", "ش تعرف تعدل"],
            ["انا دار، مساعد الدار. نگدر نشعل ونطفي الضو، نعدل الكليم، نگول لك الساعة والجو، ونحطّ لك منبّه.",
             "اسمي دار. نتحكّم ف الضو والكليم والتلفزيون والريدوات وغيرهم، كلّو هون ف الدار بلا انترنت."]),
}

OOS = [
    ("اعدل لي أتاي", "ما نگدر نعدل أتاي بيدي، ولاكن نگدر نحطّ لك منبّه ل أتاي ولّا نشعل البولّوار إلا كان عندك."),
    ("جيب لي الما", "ما نگدر نجيب شي، انا ما عندي يدين. نگدر نشعل لك البومبة إلا كانت مربوطة."),
    ("اتصل بأمّي", "التليفون ماه مربوط بيّ ظرك، ما نگدر نعيّط."),
    ("گول لي نكتة", "واحد گال لصاحبو: ش تعدل؟ گال لو: نشرب أتاي. گال: وغدوه؟ گال: نكمّل الكاس الثالث."),
    ("ش الخبار ف البلاد", "ما عندي انترنت، نخدم هون ف الدار وبس. نگدر نعاونك ف الضو والكليم والجو."),
    ("ابغيت نشري سيارة", "ذاك ماهو من خدمتي، ولاكن نگدر نزيد شي ل ليستة الشرا إلا تبغي."),
]

UNCLEAR = ["اممم", "ذاك الّي", "اشعل", "اطفي ذاك", "هيه", "ذ ذ", "اعدل لي"]


def utt_social(rng) -> Utt:
    k = pick(list(SOCIAL), rng)
    q, a = SOCIAL[k]
    return Utt(pick(q, rng), f"Social.{k}", {}, kind="social", fast_path=False, template=f"social.{k}",
               meta={"reply": pick(a, rng)})


def utt_oos(rng) -> Utt:
    q, a = pick(OOS, rng)
    return Utt(q, "OutOfScope", {}, kind="oos", fast_path=False, template="oos", meta={"reply": a})


# ---------------------------------------------- implicit / indirect requests (LLM path)
IMPLICIT = {
    "dark":   ["{area} ظلمة", "ما نشوف شي {in_area}", "الدار ظلمة", "فم ظلام {in_area}"],
    "hot":    ["الدار حامية ياسر", "حامي ياسر اليوم", "الدار سخونة ياسر", "السخانة ياسر {in_area}", "سخنت", "الحرّ ياسر اليوم", "انا نتعرّگ {in_area}"],
    "cold":   ["عندي البرد", "البرد ياسر {in_area}", "الدار باردة ياسر", "بردت"],
    "sleep":  ["لاهي نرگد", "تصبح على خير", "انا ماشي نرگد", "ساعة الرگاد"],
    "leave":  ["انا خارج", "ماشي للخدمة", "انا ماشي، ردّ بالك على الدار", "خارجين گاع"],
    "arrive": ["جيت للدار", "ها انا وصلت", "انا جيت", "رجعت للدار"],
    "shower": ["نبغي نغتسل", "نبغي نتوضّأ بالما السخون", "الما بارد ف الدوش"],
    "water":  ["الما ما ينزل", "البيدونات فارغين", "ما فم ما ف الروبيني"],
}


def utt_implicit(kind: str, area, rng) -> Utt:
    t = pick(IMPLICIT[kind], rng)
    if "{area}" in t and not area:
        t = "الدار ظلمة"
    t = t.replace("{area}", pick(area["ar"], rng) if area else "").replace(
        "{in_area}", in_area(area, rng) if area else "").strip()
    return Utt(t, f"Implicit.{kind}", {"area": area["ar"][0]} if area and area["ar"][0] in _area_words(t) else {},
               kind="implicit", fast_path=False, template=f"implicit.{kind}", meta={"area": area})


def _area_words(t):
    return t


# =============================================================== NLU sampler
def sample_nlu(rng) -> Utt:
    r = rng.random()
    area = pick(AREAS, rng) if rng.random() < 0.6 else None
    if r < 0.30:
        dev = rand_dev(rng, ["light", "fan", "climate", "media_player", "switch", "light", "light"])
        return utt_onoff(pick(["on", "off"], rng), dev, area, rng, all_flag=rng.random() < 0.08)
    if r < 0.36:
        dev = rand_dev(rng, ["cover"])
        return utt_onoff(pick(["open", "close"], rng), dev, area if dev.canon.startswith("ال") else None, rng)
    if r < 0.39:
        dev = rand_dev(rng, ["lock"])
        return utt_onoff(pick(["lock", "unlock"], rng), dev, None, rng)
    if r < 0.45:
        return utt_brightness(rand_dev(rng, ["light"]), area, rng)
    if r < 0.48:
        return utt_color(rand_dev(rng, ["light"]), area, rng)
    if r < 0.52:
        return utt_relative_light(rand_dev(rng, ["light"]), area, rng, rng.random() < 0.5)
    if r < 0.57:
        return utt_climate_set(area, rng)
    if r < 0.60:
        return utt_climate_get(area, rng)
    if r < 0.64:
        return utt_get_state(rand_dev(rng), area, rng)
    if r < 0.66:
        return utt_cover_position(area, rng)
    if r < 0.71:
        return utt_media(rng)
    if r < 0.73:
        return utt_volume_relative(rng, rng.random() < 0.5)
    if r < 0.75:
        return utt_fan_speed(area, rng)
    if r < 0.79:
        return utt_timer(rng)
    if r < 0.81:
        return utt_timer_other(rng)
    if r < 0.86:
        return utt_time_date_weather(rng)
    if r < 0.88:
        return utt_shopping(rng)
    if r < 0.90:
        return utt_vacuum(rng)
    if r < 0.91:
        return utt_nevermind(rng)
    if r < 0.93:
        return utt_social(rng)
    if r < 0.94:
        return utt_oos(rng)
    return utt_implicit(pick(list(IMPLICIT), rng), area, rng)


# =============================================================== smart home simulator
DOMAIN_PLAN = [  # (lexicon key, probability per area)
    ("light", 0.95), ("fan", 0.35), ("climate", 0.45), ("media_player", 0.3), ("cover", 0.3),
]
FIXED = [("lock", 0.9, "entrance"), ("switch", 0.6, "bathroom"), ("switch", 0.45, "yard"),
         ("cover", 0.3, "garage"), ("vacuum", 0.3, "salon"), ("switch", 0.25, "kitchen")]


def make_home(rng) -> dict:
    n_areas = rng.randint(3, 7)
    areas = rng.sample([a for a in AREAS if a["id"] not in ("garage", "entrance")], n_areas)
    ents = []
    used = set()

    def add(key, item, area):
        domain = LEX["devices"][key]["domain"]
        base = strip_al(item["ar"][0])
        name = f"{base} {area['ar'][0]}" if item["ar"][0].startswith("ال") else item["ar"][0]
        if name in used:
            return
        used.add(name)
        eid = f"{domain}.{area['id']}_{key}_{len(ents)}"
        e = {"name": name, "entity_id": eid, "domain": domain, "key": key, "item": item,
             "g": item.get("g", "m"), "area": area}
        if domain == "light":
            on = rng.random() < 0.4
            e.update(state="on" if on else "off", attrs={"brightness": pick([20, 40, 60, 80, 100], rng) if on else 0})
        elif domain == "fan":
            on = rng.random() < 0.4
            e.update(state="on" if on else "off", attrs={"percentage": pick([33, 50, 66, 100], rng) if on else 0})
        elif domain == "climate":
            on = rng.random() < 0.5
            e.update(state="cool" if on else "off", attrs={"temperature": rng.randint(18, 26),
                                                           "current_temperature": rng.randint(24, 38)})
        elif domain == "media_player":
            e.update(state=pick(["playing", "paused", "off", "idle"], rng),
                     attrs={"volume_level": pick([20, 30, 40, 50, 60], rng)})
        elif domain == "cover":
            op = rng.random() < 0.5
            e.update(state="open" if op else "closed", attrs={"current_position": 100 if op else 0})
        elif domain == "lock":
            e.update(state=pick(["locked", "unlocked"], rng), attrs={})
        elif domain == "switch":
            e.update(state=pick(["on", "off"], rng), attrs={})
        elif domain == "vacuum":
            e.update(state=pick(["docked", "cleaning"], rng), attrs={})
        ents.append(e)

    for a in areas:
        for key, p in DOMAIN_PLAN:
            if rng.random() < p:
                items = dev_items(key)
                item = items[0] if key != "cover" else items[0]
                if key == "light" and rng.random() < 0.3:
                    item = items[1]
                if key == "media_player":
                    item = pick(items[:2], rng)
                add(key, item, a)
    for key, p, aid in FIXED:
        if rng.random() < p:
            items = dev_items(key)
            a = AREA_BY_ID[aid]
            if key == "lock":
                item = items[0]
            elif key == "switch":
                item = {"bathroom": items[1], "yard": items[0], "kitchen": items[3]}[aid]
            elif key == "cover":
                item = items[2]
            else:
                item = items[0]
            add(key, item, a)
    all_areas = {a["id"]: a for a in areas}
    for e in ents:
        all_areas[e["area"]["id"]] = e["area"]
    return {"areas": list(all_areas.values()), "entities": ents,
            "time": (rng.randint(5, 23), pick([0, 5, 10, 15, 20, 30, 40, 45, 50], rng)),
            "date": (rng.randint(1, 28), rng.randint(1, 12), rng.randint(0, 6))}


def home_context(home) -> str:
    h, m = home["time"]
    d, mo, wd = home["date"]
    lines = [f"Current time is {h:02d}:{m:02d}. Today is {WEEKDAYS[wd]} {d} {MONTHS[mo - 1]} 2026.",
             "Static Context: An overview of the areas and the devices in this smart home:"]
    for e in home["entities"]:
        lines.append(f"- names: {e['name']}\n  domain: {e['domain']}\n  state: '{e['state']}'\n  areas: {e['area']['ar'][0]}")
        attrs = {k: v for k, v in e["attrs"].items() if v not in (None, 0)}
        if attrs:
            lines.append("  attributes:\n" + "\n".join(f"    {k}: {v}" for k, v in attrs.items()))
    return SYSTEM_PROMPT + "\n" + "\n".join(lines)


def find(home, domain=None, area=None, key=None):
    out = []
    for e in home["entities"]:
        if domain and e["domain"] != domain:
            continue
        if key and e["key"] != key:
            continue
        if area and e["area"]["id"] != area["id"]:
            continue
        out.append(e)
    return out


def target_args(ents, area, domain):
    if len(ents) == 1:
        return {"name": ents[0]["name"]}
    a = {"domain": domain}
    if area:
        a["area"] = area["ar"][0]
    return a


def tool_result(ents):
    return json.dumps({"response_type": "action_done",
                       "data": {"success": [{"name": e["name"], "type": "entity", "id": e["entity_id"]} for e in ents],
                                "failed": []}}, ensure_ascii=False)


DONE = {"on": ["گدّيت", "شعّلت"], "off": ["طفّيت", "سكّرت"], "open": ["فتحت", "حلّيت"], "close": ["گفلت", "سكّرت"],
        "lock": ["گفلت"], "unlock": ["فتحت"]}  # corpus: گديت 14, گفلت 2
STATE_OF = {"on": "on", "off": "off", "open": "open", "close": "closed", "lock": "locked", "unlock": "unlocked"}
HA_ON = {"on", "playing", "paused", "idle", "cool", "heat", "cleaning", "open", "unlocked"}


def is_state(e, st):
    s = e["state"]
    if st == "on":
        return s in ("on", "playing", "paused", "idle", "cool", "heat", "cleaning")
    if st == "off":
        return s in ("off", "docked")
    if st == "open":
        return s == "open"
    if st == "closed":
        return s == "closed"
    return s == st


def names_join(ents):
    names = [e["name"] for e in ents]
    if len(names) == 1:
        return names[0]
    return "، ".join(names[:-1]) + " و" + names[-1]


def reply_onoff(action, ents, rng):
    st = STATE_OF[action]
    if len(ents) == 1:
        e = ents[0]
        return pick([f"اهيه، {e['name']} {pp(st, e['g'])} ظرك.", f"{pick(DONE[action], rng)} {e['name']}.",
                     f"تمّ، {e['name']} {pp(st, e['g'])}.", f"هاني {pick(DONE[action], rng)} {e['name']}."], rng)
    return pick([f"اهيه، {pick(DONE[action], rng)} {len(ents)} {('اضواو' if ents[0]['key'] == 'light' else 'آلات')}: {names_join(ents)}.",
                 f"تمّ، {names_join(ents)} گاع {pp(st, 'f') if len(ents) > 1 else ''}.".replace("  ", " ")], rng)


# =============================================================== SFT scenario builder
_OK_RNG = random.Random(11)


def vary_ok(text: str) -> str:
    """Corpus-attested acknowledgements instead of one fixed word (اهيه 188, ايوه 99, زين 238)."""
    if text.startswith("اهيه"):
        return _OK_RNG.choice(["اهيه", "اهيه", "ايوه", "زين"]) + text[4:]
    return text


def conv(home, user, calls_and_results, final, extra_turns=None):
    msgs = [{"role": "system", "content": home.get("_ctx") or home_context(home)}, {"role": "user", "content": user}]
    if calls_and_results:
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}}
                                    for n, a, _ in calls_and_results]})
        for n, _, res in calls_and_results:
            msgs.append({"role": "tool", "name": n, "content": res})
    msgs.append({"role": "assistant", "content": vary_ok(final)})
    for t in extra_turns or []:
        msgs.append(t)
    return msgs


def apply_state(ents, action):
    for e in ents:
        e["state"] = {"on": "cool" if e["domain"] == "climate" else ("playing" if e["domain"] == "media_player" else
                                                                     ("cleaning" if e["domain"] == "vacuum" else "on")),
                      "off": "off", "open": "open", "close": "closed", "lock": "locked", "unlock": "unlocked"}[action]


def sft_onoff(home, rng):
    action = rng.choices(["on", "off", "open", "close", "lock", "unlock"], [35, 35, 8, 8, 7, 7])[0]
    keys = {"on": ["light", "light", "fan", "climate", "media_player", "switch"],
            "off": ["light", "light", "fan", "climate", "media_player", "switch"],
            "open": ["cover"], "close": ["cover"], "lock": ["lock"], "unlock": ["lock"]}[action]
    key = pick(keys, rng)
    cands = find(home, key=key)
    scenario = rng.choices(["ok", "missing", "ambiguous", "already"], [62, 10, 14, 14])[0]
    if scenario == "missing" or not cands:
        # ask for a device type/area combination that does not exist
        absent_areas = [a for a in AREAS if not find(home, key=key, area=a)]
        if not absent_areas:
            return None
        area = pick(absent_areas, rng)
        dev = Dev(key, LEX["devices"][key]["domain"], pick(dev_items(key), rng))
        u = utt_onoff(action, dev, area, rng)
        avail = cands[:3]
        tail = f" عندك {names_join(avail)}، تبغي نعدل ش فيهم؟" if avail else ""
        return conv(home, u.text, [], f"ما لگيت حتى {dev.canon} {pick(LEX['in']['ar'][:1], rng)} {area['ar'][0]}.{tail}"), u
    e = pick(cands, rng)
    if key != "light":
        cands = [x for x in cands if x["item"] is e["item"]]
    area = e["area"]
    dev = Dev(key, e["domain"], e["item"])
    same_type = [x for x in cands if x["area"]["id"] == area["id"]]
    if scenario == "ambiguous" and len({x["area"]["id"] for x in cands}) >= 2 and key != "light":
        u = utt_onoff(action, dev, None, rng)
        areas_txt = " ولّا ".join(sorted({x["area"]["ar"][0] for x in cands})[:3])
        q = f"عندك {dev.canon} ف {areas_txt}. ف أي بيت؟" if rng.random() < 0.5 else f"أي {strip_al(dev.canon)}؟ الّي ف {areas_txt}؟"
        # second turn: user disambiguates
        ans = pick([f"{pick(LEX['in']['ar'], rng)} {pick(area['ar'], rng)}", f"الّي ف {area['ar'][0]}", pick(area["ar"], rng)], rng)
        before = copy.deepcopy(same_type)
        apply_state(same_type, action)
        call = [("HassTurnOn" if action in ("on", "open", "lock") else "HassTurnOff",
                 target_args(same_type, area, e["domain"]), tool_result(same_type))]
        extra = [{"role": "user", "content": ans},
                 {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": call[0][0], "arguments": call[0][1]}}]},
                 {"role": "tool", "name": call[0][0], "content": call[0][2]},
                 {"role": "assistant", "content": reply_onoff(action, same_type, rng)}]
        del before
        return conv(home, u.text, [], q, extra), u
    # area mention: omit if the device type is unique in the house
    unique = len(cands) == 1
    use_area = (not unique) or rng.random() < 0.5
    if key == "lock":
        use_area = False
    u = utt_onoff(action, dev, area if use_area else None, rng,
                  all_flag=(key == "light" and len(same_type) > 1 and rng.random() < 0.3))
    targets = same_type if use_area or unique else same_type
    want_state = STATE_OF[action]
    if scenario == "already" and all(is_state(t, want_state) for t in targets):
        t = targets[0]
        return conv(home, u.text, [], pick([f"{t['name']} {pp(want_state, t['g'])} من گبيل.",
                                           f"{t['name']} راهو {pp(want_state, t['g'])}، ما يحتاج شي."], rng)), u
    apply_state(targets, action)
    name = "HassTurnOn" if action in ("on", "open", "lock") else "HassTurnOff"
    return conv(home, u.text, [(name, target_args(targets, area, e["domain"]), tool_result(targets))],
                reply_onoff(action, targets, rng)), u


def sft_light_set(home, rng):
    lights = find(home, key="light")
    if not lights:
        return None
    e = pick(lights, rng)
    area = e["area"]
    dev = Dev("light", "light", e["item"])
    mode = pick(["abs", "color", "rel", "rel"], rng)
    if mode == "abs":
        u = utt_brightness(dev, area, rng)
        args = {"name": e["name"], "brightness": u.slots["brightness"]}
        e["state"], e["attrs"]["brightness"] = "on", u.slots["brightness"]
        rep = pick([f"اهيه، {e['name']} ظرك ف {u.slots['brightness']} فالمية.", f"اعدلت {e['name']} ف {u.slots['brightness']}%."], rng)
    elif mode == "color":
        u = utt_color(dev, area, rng)
        args = {"name": e["name"], "color": u.slots["color"]}
        e["state"] = "on"
        rep = pick([f"اهيه، {e['name']} ردّيتو {u.meta['color_ar']}.", f"تمّ، {e['name']} ظرك {u.meta['color_ar']}."], rng)
    else:
        up = rng.random() < 0.5
        u = utt_relative_light(dev, area, rng, up)
        cur = e["attrs"].get("brightness", 0) if e["state"] == "on" else 0
        if not up and cur == 0:
            return conv(home, u.text, [], f"{e['name']} مطفي من گبيل، ما نگدر ننگّصو."), u
        new = max(5, min(100, cur + u.meta["step"]))
        if new == cur:
            return conv(home, u.text, [], f"{e['name']} ف {cur}% ذ هو الحدّ."), u
        args = {"name": e["name"], "brightness": new}
        e["state"], e["attrs"]["brightness"] = "on", new
        rep = f"{'زدت' if up else 'نگّصت'} {e['name']} من {cur} ل {new} فالمية."
    return conv(home, u.text, [("HassLightSet", args, tool_result([e]))], rep), u


def sft_climate(home, rng):
    acs = find(home, domain="climate")
    mode = pick(["set", "get", "get", "hot", "cold"], rng)
    if not acs:
        if mode in ("hot",) and find(home, domain="fan"):
            fan = pick(find(home, domain="fan"), rng)
            u = utt_implicit("hot", None, rng)
            fan["state"] = "on"
            return conv(home, u.text, [("HassTurnOn", {"name": fan["name"]}, tool_result([fan]))],
                        f"ما عندك كليم، ولاكن شعّلت {fan['name']}. الله يبرّد عليك."), u
        return None
    e = pick(acs, rng)
    area = e["area"]
    use_area = len(acs) > 1 or rng.random() < 0.5
    if mode == "set":
        u = utt_climate_set(area if use_area else None, rng)
        t = u.slots["temperature"]
        e["state"], e["attrs"]["temperature"] = "cool", t
        return conv(home, u.text, [("HassClimateSetTemperature", {"name": e["name"], "temperature": t}, tool_result([e]))],
                    pick([f"اهيه، {e['name']} ظرك على {t} درجة.", f"اعدلت {e['name']} ف {t} درجة."], rng)), u
    if mode == "get":
        u = utt_climate_get(area if use_area else None, rng)
        cur = e["attrs"]["current_temperature"]
        res = json.dumps({"response_type": "query_answer", "data": {"success": [{"name": e["name"], "id": e["entity_id"]}],
                          "current_temperature": cur, "target_temperature": e["attrs"]["temperature"], "state": e["state"]}},
                         ensure_ascii=False)
        extra = " الكليم مطفي، تبغي نگديه؟" if e["state"] == "off" and cur >= 30 else ""
        return conv(home, u.text, [("HassClimateGetTemperature", {"area": area["ar"][0]} if use_area else {"name": e["name"]}, res)],
                    f"السخانة {('ف ' + area['ar'][0]) if use_area else 'ف الدار'} ظرك {cur} درجة.{extra}"), u
    if mode == "hot":
        u = utt_implicit("hot", area if rng.random() < 0.5 else None, rng)
        if e["state"] == "off":
            e["state"] = "cool"
            return conv(home, u.text, [("HassTurnOn", {"name": e["name"]}, tool_result([e]))],
                        pick([f"شعّلت {e['name']} على {e['attrs']['temperature']} درجة، شوي وتبرد الدار.",
                              f"اهيه، {e['name']} گادي ظرك. السخانة ف {area['ar'][0]} {e['attrs']['current_temperature']} درجة."], rng)), u
        new = max(16, e["attrs"]["temperature"] - 2)
        old, e["attrs"]["temperature"] = e["attrs"]["temperature"], new
        return conv(home, u.text, [("HassClimateSetTemperature", {"name": e["name"], "temperature": new}, tool_result([e]))],
                    f"{e['name']} گادي من گبيل، نگّصتو من {old} ل {new} درجة."), u
    u = utt_implicit("cold", area if rng.random() < 0.5 else None, rng)
    if e["state"] == "off":
        return conv(home, u.text, [], f"{e['name']} مطفي من گبيل. السخانة ظرك {e['attrs']['current_temperature']} درجة. تبغي نسكّر الطاگة؟"), u
    e["state"] = "off"
    return conv(home, u.text, [("HassTurnOff", {"name": e["name"]}, tool_result([e]))], f"طفّيت {e['name']}، باش تسخن الدار شوي."), u


def sft_state(home, rng):
    if not home["entities"]:
        return None
    e = pick(home["entities"], rng)
    dev = Dev(e["key"], e["domain"], e["item"])
    u = utt_get_state(dev, e["area"] if rng.random() < 0.6 else None, rng)
    st = u.meta["state"]
    if u.template == "state.which":
        on = [x for x in find(home, key=e["key"]) if is_state(x, "on")]
        res = json.dumps({"response_type": "query_answer", "data": {"success": [{"name": x["name"]} for x in on]}}, ensure_ascii=False)
        rep = (f"الّي {pp('on', e['g'])} ظرك: {names_join(on)}." if on else f"حتى {strip_al(e['item']['ar'][0])} ما {pp('on', e['g'])} ظرك.")
        return conv(home, u.text, [("HassGetState", {"domain": e["domain"], "state": "on"}, res)], rep), u
    yes = is_state(e, st)
    res = json.dumps({"response_type": "query_answer", "data": {"success": [{"name": e["name"], "state": e["state"]}] if yes else [],
                      "failed": [] if yes else [{"name": e["name"], "state": e["state"]}]}}, ensure_ascii=False)
    if yes:
        rep = pick([f"ايّه، {e['name']} {pp(st, e['g'])}.", f"ايّه، راهو {pp(st, e['g'])}."], rng)
    else:
        opp = {"on": "off", "open": "closed", "closed": "open", "locked": "unlocked", "unlocked": "locked"}[st]
        offer = {"on": "تبغي نگديه؟", "open": "تبغي نحلّو؟", "closed": "تبغي نسدّو؟", "locked": "تبغي نغلگو؟", "unlocked": "تبغي نحلّو؟"}[st]
        rep = f"لا، {e['name']} {pp(opp, e['g'])}. {offer}"
    return conv(home, u.text, [("HassGetState", {"name": e["name"], "state": st}, res)], rep), u


COND = {"sunny": "شمس صافية", "clear-night": "الليل صافي", "cloudy": "سحاب", "partlycloudy": "شوي سحاب",
        "rainy": "مطر", "windy": "ريح", "exceptional": "عجاج"}


def sft_info(home, rng):
    u = utt_time_date_weather(rng)
    if u.intent == "HassGetCurrentTime":
        h, m = home["time"]
        return conv(home, u.text, [], pick([f"الساعة ظرك {say_time(h, m)}.", f"{say_time(h, m)}."], rng)), u
    if u.intent == "HassGetCurrentDate":
        d, mo, wd = home["date"]
        return conv(home, u.text, [], f"اليوم {WEEKDAYS[wd]}، {d} {MONTHS[mo - 1]}."), u
    cond = pick(list(COND), rng)
    t = rng.randint(18, 46)
    wind = rng.randint(5, 50)
    res = json.dumps({"condition": cond, "temperature": t, "humidity": rng.randint(8, 80), "wind_speed": wind,
                      "temperature_unit": "°C", "wind_speed_unit": "km/h"}, ensure_ascii=False)
    advice = ""
    if t >= 40:
        advice = " السخانة ياسر، اشرب الما وگابل الظلّ."
    elif cond == "exceptional" or wind > 40:
        advice = " فم ريح وعجاج، اگفل الطاگات."
    elif cond == "rainy":
        advice = " فم الشتا، الله يجعلها خير."
    return conv(home, u.text, [("HassGetWeather", {}, res)], f"الجو {COND[cond]}، والسخانة {t} درجة.{advice}"), u


def sft_timer(home, rng):
    r = rng.random()
    if r < 0.65:
        u = utt_timer(rng)
        (unit, n), = u.slots.items()
        label = count_with_unit(n, unit[:-1], random.Random(0))
        res = json.dumps({"response_type": "action_done", "data": {"success": [], "failed": []}}, ensure_ascii=False)
        return conv(home, u.text, [("HassStartTimer", dict(u.slots), res)],
                    pick([f"اهيه، منبّه {label} بدا.", f"تمّ، لاهي نگول لك بعد {label}.", f"بدا العدّاد: {label}."], rng)), u
    u = utt_timer_other(rng)
    if u.intent == "HassCancelAllTimers":
        return conv(home, u.text, [("HassCancelAllTimers", {}, json.dumps({"response_type": "action_done"}))], "اهيه، الغيت المنبّهات گاع."), u
    if u.intent == "HassCancelTimer":
        return conv(home, u.text, [("HassCancelTimer", {}, json.dumps({"response_type": "action_done"}))], "اهيه، الغيت المنبّه."), u
    rem = rng.randint(1, 25)
    res = json.dumps({"timers": [{"minutes_left": rem, "seconds_left": rng.randint(0, 59)}]})
    return conv(home, u.text, [("HassTimerStatus", {}, res)], f"باگي {count_with_unit(rem, 'minute', rng)} تقريبا."), u


def sft_media(home, rng):
    mps = find(home, domain="media_player")
    if not mps:
        return None
    e = pick(mps, rng)
    dev = Dev("media_player", "media_player", e["item"])
    if rng.random() < 0.3:
        up = rng.random() < 0.5
        u = utt_volume_relative(rng, up)
        cur = e["attrs"]["volume_level"]
        new = max(0, min(100, cur + u.meta["step"]))
        e["attrs"]["volume_level"] = new
        return conv(home, u.text, [("HassSetVolume", {"name": e["name"], "volume_level": new}, tool_result([e]))],
                    f"{'زدت' if up else 'نگّصت'} الصوت من {cur} ل {new}."), u
    u = utt_media(rng, dev)
    if u.intent == "HassMediaPause" and e["state"] != "playing":
        return conv(home, u.text, [], f"{e['name']} ما يخدم شي ظرك، ما فم شي نحبّس."), u
    args = {"name": e["name"]}
    args.update(u.slots)
    rep = {"HassMediaPlayerMute": "اهيه، سكّتّو.", "HassMediaPlayerUnmute": "اهيه، ردّيت الصوت.", "HassMediaPause": "اهيه، حبّستو.", "HassMediaUnpause": "اهيه، رجع يخدم.", "HassMediaNext": "هاك الّي بعدو.",
           "HassMediaPrevious": "رجّعت للّي گبل.", "HassSetVolume": f"الصوت ظرك ف {u.slots.get('volume_level')}."}[u.intent]
    return conv(home, u.text, [(u.intent, args, tool_result([e]))], rep), u


def sft_misc(home, rng):
    r = rng.random()
    if r < 0.3:
        u = utt_shopping(rng)
        return conv(home, u.text, [("HassShoppingListAddItem", u.slots, json.dumps({"response_type": "action_done"}))],
                    pick([f"زدت {u.slots['item']} ل ليستة الشرا.", f"تمّ، {u.slots['item']} مكتوب ف الليستة."], rng)), u
    if r < 0.45:
        vac = find(home, domain="vacuum")
        u = utt_vacuum(rng)
        if not vac:
            return conv(home, u.text, [], "ما عندك حتى روبو تنظيف مربوط ب الدار."), u
        return conv(home, u.text, [(u.intent, {"name": vac[0]["name"]}, tool_result(vac))],
                    "اهيه، الروبو بدا ينظّف." if u.intent == "HassVacuumStart" else "اهيه، الروبو راجع لبلاصتو."), u
    if r < 0.6:
        u = utt_social(rng)
        return conv(home, u.text, [], u.meta["reply"]), u
    if r < 0.72:
        u = utt_oos(rng)
        return conv(home, u.text, [], u.meta["reply"]), u
    if r < 0.82:
        u = utt_nevermind(rng)
        return conv(home, u.text, [], pick(["ما عليه.", "اهيه.", "خلاص، انا هون إلا احتجت شي."], rng)), u
    if r < 0.9:
        txt = pick(UNCLEAR, rng)
        u = Utt(txt, "Clarify", {}, kind="clarify", fast_path=False, template="clarify")
        return conv(home, u.text, [], pick(["ما فهمتك زين، عاود الله يخليك.", "ش تبغي نعدل بالظبط؟", "ش من آلة تبغي؟"], rng)), u
    covers = [e for e in find(home, domain="cover") if e["key"] == "cover" and e["item"]["ar"][0] == "الريدو"]
    if not covers:
        return None
    e = pick(covers, rng)
    u = utt_cover_position(e["area"], rng)
    return conv(home, u.text, [("HassSetPosition", {"name": e["name"], "position": u.slots["position"]}, tool_result([e]))],
                f"اهيه، {e['name']} ظرك مفتوح {u.slots['position']} فالمية."), u


def sft_routine(home, rng):
    kind = pick(["sleep", "leave", "arrive", "dark", "shower", "water"], rng)
    lights = find(home, domain="light")
    if kind in ("sleep", "leave"):
        u = utt_implicit(kind, None, rng)
        calls, done = [], []
        on_lights = [e for e in lights if e["state"] == "on"]
        if on_lights:
            calls.append(("HassTurnOff", {"domain": "light"}, tool_result(on_lights)))
            done.append("طفّيت الضو گاع")
            apply_state(on_lights, "off")
        if kind == "leave":
            on_ac = [e for e in find(home, domain="climate") if e["state"] != "off"]
            if on_ac:
                calls.append(("HassTurnOff", {"domain": "climate"}, tool_result(on_ac)))
                done.append("طفّيت الكليم")
                apply_state(on_ac, "off")
        else:
            tv = [e for e in find(home, domain="media_player") if e["state"] in ("playing", "paused", "idle")]
            if tv:
                calls.append(("HassTurnOff", {"domain": "media_player"}, tool_result(tv)))
                done.append("طفّيت التلفزيون")
        locks = [e for e in find(home, domain="lock") if e["state"] == "unlocked"]
        if locks:
            calls.append(("HassTurnOn", {"name": locks[0]["name"]}, tool_result(locks)))
            done.append(f"غلگت {locks[0]['name']}")
        if not calls:
            return conv(home, u.text, [], "كلّ شي مطفي ومگفول من گبيل. " + ("تصبح على خير." if kind == "sleep" else "طريق السلامة.")), u
        bye = "تصبح على خير." if kind == "sleep" else "طريق السلامة، الله يحفظك."
        return conv(home, u.text, calls, "، و".join(done) + ". " + bye), u
    if kind in ("arrive", "dark"):
        area = pick(home["areas"], rng) if kind == "dark" else AREA_BY_ID.get("salon")
        u = utt_implicit(kind, area if kind == "dark" and rng.random() < 0.7 else None, rng)
        tgt_area = area if u.slots.get("area") else None
        tl = [e for e in lights if (tgt_area is None and e["area"]["id"] in ("salon", "entrance")) or
              (tgt_area and e["area"]["id"] == tgt_area["id"])] or lights[:1]
        if not tl:
            return None
        apply_state(tl, "on")
        greet = "مرحبا بيك ف دارك. " if kind == "arrive" else ""
        return conv(home, u.text, [("HassTurnOn", target_args(tl, tgt_area, "light"), tool_result(tl))],
                    f"{greet}شعّلت {names_join(tl)}."), u
    if kind == "shower":
        heater = [e for e in find(home, key="switch") if "الشوفو" in e["item"]["ar"]]
        u = utt_implicit(kind, None, rng)
        if not heater:
            return conv(home, u.text, [], "ما عندك شوفو مربوط ب الدار، ما نگدر نسخّن الما."), u
        apply_state(heater, "on")
        return conv(home, u.text, [("HassTurnOn", {"name": heater[0]["name"]}, tool_result(heater))],
                    f"شعّلت {heater[0]['name']}، اصبر شي اشوي لين يسخن الما."), u
    pump = [e for e in find(home, key="switch") if "البومبة" in e["item"]["ar"]]
    u = utt_implicit(kind, None, rng)
    if not pump:
        return conv(home, u.text, [], "ما عندك بومبة مربوطة ب الدار. شوف روبيني الشارع."), u
    if pump[0]["state"] == "on":
        return conv(home, u.text, [], f"{pump[0]['name']} گادية من گبيل. يمكن الما مگطوع من البرّا."), u
    apply_state(pump, "on")
    return conv(home, u.text, [("HassTurnOn", {"name": pump[0]["name"]}, tool_result(pump))], f"شعّلت {pump[0]['name']}، الما لاهي يطلع."), u


def sft_multi(home, rng):
    lights = find(home, key="light")
    others = [e for e in home["entities"] if e["domain"] in ("climate", "fan", "media_player", "cover")]
    if not lights or not others:
        return None
    a, b = pick(lights, rng), pick(others, rng)
    act_a = "off" if a["state"] == "on" else "on"
    act_b = {"cover": "close" if b["state"] == "open" else "open"}.get(b["domain"], "off" if b["state"] not in ("off",) else "on")
    ua = utt_onoff(act_a, Dev("light", "light", a["item"]), a["area"], rng)
    ub = utt_onoff(act_b, Dev(b["key"], b["domain"], b["item"]), b["area"], rng, plain=True)
    joiner = pick([" و", "، و", "، ومن بعد ", " وزيد "], rng)
    text = ua.text.rstrip("؟") + joiner + ub.text
    na = "HassTurnOn" if act_a == "on" else "HassTurnOff"
    nb = "HassTurnOn" if act_b in ("on", "open") else "HassTurnOff"
    apply_state([a], act_a)
    apply_state([b], act_b)
    calls = [(na, {"name": a["name"]}, tool_result([a])), (nb, {"name": b["name"]}, tool_result([b]))]
    rep = f"اهيه: {a['name']} {pp(STATE_OF[act_a], a['g'])}، و{b['name']} {pp(STATE_OF[act_b], b['g'])}."
    return conv(home, text, calls, rep), Utt(text, "Multi", {}, kind="multi", fast_path=False, template="multi")


SFT_BUILDERS = [(sft_onoff, 34), (sft_light_set, 9), (sft_climate, 11), (sft_state, 8), (sft_info, 9),
                (sft_timer, 6), (sft_media, 6), (sft_misc, 9), (sft_routine, 6), (sft_multi, 4)]


# =============================================================== split + write
def split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "test" if h < 5 else "dev" if h < 10 else "train"


def nlu_record(u: Utt, rng, idx: int, aug: str) -> dict:
    text = u.text
    if aug == "asr":
        text = asr_noise(text, rng)
    elif aug == "arabizi":
        text = to_arabizi(text)
    return {"id": f"nlu-{idx:06d}", "text": text, "text_norm": normalize(text) if aug != "arabizi" else text.lower(),
            "intent": u.intent, "slots": u.slots, "kind": u.kind, "fast_path": u.fast_path and aug == "clean",
            "template": u.template, "augment": aug}


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlu", type=int, default=30000)
    ap.add_argument("--sft", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--asr-prompts", type=int, default=2000)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---------------- NLU
    seen, splits = set(), {"train": [], "dev": [], "test": []}
    tries, idx = 0, 0
    while idx < args.nlu and tries < args.nlu * 20:
        tries += 1
        u = sample_nlu(rng)
        aug = rng.choices(["clean", "asr", "arabizi"], [70, 22, 8])[0]
        rec = nlu_record(u, rng, idx, aug)
        k = rec["text_norm"]
        if k in seen:
            continue
        seen.add(k)
        # split on the noise-free normalized text so augmented twins never leak across splits
        splits[split_of(normalize(strip_tashkeel(u.text)))].append(rec)
        idx += 1
    for s, rows in splits.items():
        write_jsonl(OUT / "nlu" / f"{s}.jsonl", rows)

    # ---------------- SFT
    sft = {"train": [], "dev": [], "test": []}
    builders, weights = zip(*SFT_BUILDERS)
    n, tries, sft_seen = 0, 0, set()
    intent_counter = Counter()
    while n < args.sft and tries < args.sft * 30:
        tries += 1
        home = make_home(rng)
        home["_ctx"] = home_context(home)
        b = rng.choices(builders, weights)[0]
        out = b(home, rng)
        if not out:
            continue
        msgs, u = out
        # light ASR noise on the user turn(s) for robustness
        if rng.random() < 0.25:
            for m in msgs:
                if m["role"] == "user":
                    m["content"] = asr_noise(m["content"], rng, p=0.25)
        key = msgs[1]["content"] + "|" + msgs[-1]["content"]
        if key in sft_seen:
            continue
        sft_seen.add(key)
        rec = {"id": f"sft-{n:06d}", "intent": u.intent, "kind": u.kind, "messages": msgs}  # tools: data/schema/ha_tools.json
        sft[split_of(normalize(strip_tashkeel(u.text)))].append(rec)
        intent_counter[u.intent] += 1
        n += 1
    for s, rows in sft.items():
        write_jsonl(OUT / "sft" / f"{s}.jsonl", rows)

    # ---------------- ASR recording prompts (clean, unique, diverse)
    prompts, pseen = [], set()
    prng = random.Random(args.seed + 1)
    while len(prompts) < args.asr_prompts:
        u = sample_nlu(prng)
        t = strip_tashkeel(u.text)
        if normalize(t) in pseen or len(t) < 4:
            continue
        pseen.add(normalize(t))
        prompts.append({"prompt_id": f"hsn-{len(prompts):05d}", "text": t, "intent": u.intent})
    (OUT / "asr").mkdir(parents=True, exist_ok=True)
    with open(OUT / "asr" / "recording_prompts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "text", "intent"])
        w.writeheader()
        w.writerows(prompts)

    stats = {
        "nlu": {s: len(r) for s, r in splits.items()},
        "nlu_intents": dict(Counter(r["intent"] for rows in splits.values() for r in rows).most_common()),
        "nlu_augment": dict(Counter(r["augment"] for rows in splits.values() for r in rows)),
        "sft": {s: len(r) for s, r in sft.items()},
        "sft_intents": dict(intent_counter.most_common()),
        "sft_with_tool_calls": sum(any("tool_calls" in m for m in r["messages"]) for rows in sft.values() for r in rows),
        "sft_multi_turn": sum(sum(m["role"] == "user" for m in r["messages"]) > 1 for rows in sft.values() for r in rows),
        "asr_prompts": len(prompts),
        "seed": args.seed,
    }
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
