#!/usr/bin/env python3
"""Build Home Assistant custom sentences (hassil format) for Hassaniya from the lexicon.

Output: homeassistant/custom_sentences/ar/hassaniya.yaml
Copy it to <HA config>/custom_sentences/ar/ and restart HA. Set the Assist pipeline
language to Arabic (ar). These give the *fast, fully local, zero-LLM* path; anything
they don't match falls back to the fine-tuned local LLM (see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hassaniya import COUNT_SHORT, ROOT, load_lexicon, obj_suffix, strip_al, strip_tashkeel  # noqa: E402
import generate as G  # noqa: E402

LEX = load_lexicon()
OUT = ROOT / "homeassistant" / "custom_sentences" / "ar" / "hassaniya.yaml"


def alt(words) -> str:
    """hassil alternative group, adding tashkeel-free twins and deduping."""
    seen, out = set(), []
    for w in words:
        for v in (w, strip_tashkeel(w)):
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
    return "(" + "|".join(out) + ")"


def verb_rule(key: str) -> str:
    v = LEX["verbs"][key]
    want = alt(LEX["want"]["ar"])
    return alt(v["imp_m"] + v["imp_f"] + v["imp_pl"] + v["req"] + v.get("accept", []))[:-1] + f"|{want} {alt(v['req'])})"


def verb_obj_rule(key: str, g: str) -> str:
    v = LEX["verbs"][key]
    forms = [w + obj_suffix(g) for w in v["imp_m"] + v["imp_f"] + v.get("accept", []) if " " not in w] + v["imp_pl"]
    return alt(forms)


def dev_words(key: str, only_al=False):
    return [w for it in LEX["devices"][key]["items"] for w in it["ar"] if (w.startswith("ال") or not only_al)]


def number_values(lo=0, hi=100):
    vals = []
    for n in range(lo, hi + 1):
        words = set(LEX["numbers"].get(n, []))
        tens, unit = divmod(n, 10)
        if n not in LEX["numbers"] and unit and tens * 10 in LEX["numbers"]:
            for u in LEX["numbers"][unit]:
                for t in LEX["numbers"][tens * 10]:
                    words.add(f"{u} و{t}")
        if n in LEX.get("numbers_fr", {}):
            words.add(LEX["numbers_fr"][n])
        for w in words:
            for v in {w, strip_tashkeel(w)}:
                vals.append({"in": v, "out": n})
    return vals


def build() -> dict:
    R = {  # expansion rules
        "pre": "[" + alt([f + "" for f in LEX["fillers"]["ar"]]) + "] [" + alt(LEX["wake_words"]["ar"]) + "] [" + alt(LEX["politeness"]["pre"]) + "] [<soft>]",
        "post": "[<soft>] [" + alt(LEX["now"]["ar"]) + "] [" + alt(LEX["politeness"]["post"]) + "] [<soft>]",
        "soft": alt(LEX["softener"]["ar"]),
        "in": alt(LEX["in"]["ar"]),
        "loc": "<in> {hsn_area:area}",
        "all": alt(LEX["all"]["ar"]),
        "want": alt(LEX["want"]["ar"]),
        "pct_unit": alt(LEX["units"]["percent"][:3]),
        "deg": alt(LEX["units"]["degree"]),
        "how_much": alt(LEX["q"]["how_much"]["ar"]),
        "is_it": alt(LEX["q"]["is_it"]["ar"]),
        "timer_word": alt(["منبّه", "تايمر", "عدّاد", "مينوتري"]),
        "timer_word_def": alt(["المنبّه", "التايمر", "العدّاد"]),
        "purpose": alt(["ل أتاي", "ل الروز", "ل الفرن", "ل الصلاة"]),
        "list_word": alt(["ليستة الشرا", "لاليست", "ليستة التسوّگ", "ليستة لكورس", "الليستة"]),
        "media": alt(dev_words("media_player")),
        "curtain": alt(LEX["devices"]["cover"]["items"][0]["ar"]),
    }
    for vk in LEX["verbs"]:
        R[f"v_{vk}"] = verb_rule(vk)
    for k in LEX["devices"]:
        R[f"d_{k}"] = alt(dev_words(k))
        R[f"dc_{k}"] = alt([strip_al(w) for w in dev_words(k, only_al=True)])

    def num(slot):
        return f"({{hsn_number:{slot}}}|{{hsn_digits:{slot}}})"

    intents: dict = {}

    def add(intent, sentences, slots=None, response=None):
        d = {"sentences": [f"<pre> {s} <post>" for s in sentences]}
        if slots:
            d["slots"] = slots
        if response:
            d["response"] = response
        intents.setdefault(intent, {"data": []})["data"].append(d)

    # ---- on / off
    for key in ["light", "fan", "climate", "media_player", "switch"]:
        dom = LEX["devices"][key]["domain"]
        on_v = "turn_on" if key == "light" else "turn_on_device"
        for intent, vk in (("HassTurnOn", on_v), ("HassTurnOff", "turn_off")):
            objs = "|".join(verb_obj_rule(vk, g) for g in ("m", "f"))
            add(intent, [
                f"<v_{vk}> <d_{key}> [<loc>]",
                f"<v_{vk}> <dc_{key}> {{hsn_area:area}}",
                f"<d_{key}> [<loc>] ({objs})",
                f"<v_{vk}> <d_{key}> <all> [<loc>]",
                f"<v_{vk}> <all> <d_{key}> [<loc>]",
            ], {"domain": dom}, response="hsn_on" if intent == "HassTurnOn" else "hsn_off")
    # covers (open/close) and locks
    cover_al = alt([w for w in dev_words("cover", only_al=True)])
    cover_fixed = alt([w for w in dev_words("cover") if not w.startswith("ال")])
    for intent, vk in (("HassTurnOn", "open"), ("HassTurnOff", "close")):
        objs = "|".join(verb_obj_rule(vk, g) for g in ("m", "f"))
        add(intent, [f"<v_{vk}> {cover_al} [<loc>]", f"<v_{vk}> {cover_fixed}",
                     f"{cover_al} [<loc>] ({objs})", f"{cover_fixed} ({objs})",
                     f"<v_{vk}> <dc_cover> {{hsn_area:area}}"],
            {"domain": "cover"}, response="hsn_on" if intent == "HassTurnOn" else "hsn_off")
    add("HassTurnOn", ["<v_close> <d_lock> [(بالمفتاح|بالساروت)]"], {"domain": "lock"}, response="hsn_lock")
    add("HassTurnOff", ["<v_open> <d_lock>", "<d_lock> (" + "|".join(verb_obj_rule("open", g) for g in ("m", "f")) + ")"],
        {"domain": "lock"}, response="hsn_unlock")

    # ---- light set
    add("HassLightSet", [f"<v_set> <d_light> [<loc>] [(ف|على|ل)] {num('brightness')} <pct_unit>",
                         f"<d_light> [<loc>] {num('brightness')} <pct_unit>"], response="hsn_set")
    add("HassLightSet", ["<v_set> <d_light> [<loc>] [بلون] {hsn_color:color}"], response="hsn_set")
    # ---- climate
    add("HassClimateSetTemperature", [
        f"<v_set> <d_climate> [<loc>] [(ف|على)] {num('temperature')} [<deg>]",
        f"<want> السخانة [<loc>] {num('temperature')} [<deg>]",
        f"<d_climate> [<loc>] {num('temperature')} [<deg>]"], response="hsn_set")
    add("HassClimateGetTemperature", [
        "<how_much> (السخانة|درجة الحرارة) [<loc>]", "السخانة [<loc>] <how_much>",
        "ش السخانة [<loc>]", "<v_tell> السخانة [<loc>]"])
    # ---- state queries
    for key in ["light", "fan", "climate", "media_player", "switch"]:
        dom = LEX["devices"][key]["domain"]
        add("HassGetState", [f"[<is_it>] <d_{key}> [<loc>] (گادي|گادية|كادي|شاعل|شاعلة)",
                             f"<v_check> <is_it> <d_{key}> [<loc>] (گادي|گادية|كادي|شاعل|شاعلة)",
                             f"ش من <dc_{key}> (گادي|گادية|كادي|شاعل|شاعلة) [<loc>]"], {"domain": dom, "state": "on"})
    for st, words in (("open", "(مفتوح|مفتوحة|محلول|محلولة)"), ("closed", "(مگفول|مگفولة|مسدود|مسدودة)")):
        add("HassGetState", [f"[<is_it>] <d_cover> [<loc>] {words}", f"<v_check> <is_it> <d_cover> [<loc>] {words}"],
            {"domain": "cover", "state": st})
    for st, words in (("locked", "(مگفول|مگفولة|مغلوگ)"), ("unlocked", "(مفتوح|مفتوحة|محلول)")):
        add("HassGetState", [f"[<is_it>] <d_lock> {words}", f"<v_check> <is_it> <d_lock> {words}"],
            {"domain": "lock", "state": st})
    # ---- cover position
    add("HassSetPosition", ["<v_open> <curtain> [<loc>] (النصّ|انصّو|نصّ|النص|انصو|نص)"], {"position": 50})
    add("HassSetPosition", [f"<v_set> <curtain> [<loc>] (ف|على) {num('position')} <pct_unit>"])
    # ---- media
    add("HassMediaPause", ["(حبّس|وگّف) <media> " + alt(LEX["a_little"]["ar"]),
                           "حبّس (<media>|الموسيقى|الغنا|المديح)", "اعدل pause"])
    add("HassMediaUnpause", ["كمّل (<media>|الموسيقى|المديح)", "ردّ <media> يخدم", "خلّيه يكمّل",
                             "(اگدي|شعّل|حلّ) الموسيقى من جديد"])
    add("HassMediaNext", ["الّي بعدو", "دوّر للّي بعدو", "الأغنية الجاية", "جيب الّي بعدها", "next"])
    add("HassMediaPrevious", ["ردّ للّي گبل", "الأغنية الّي فاتت", "رجّع الّي گبلها", "ردّ اللّي فات"])
    add("HassSetVolume", [f"<v_set> الصوت (ف|على) {num('volume_level')}",
                          f"الصوت {num('volume_level')} " + alt(LEX["units"]["percent"][:2]),
                          f"<v_set> صوت <dc_media_player> ف {num('volume_level')}"])
    add("HassMediaPlayerMute", ["سكّت (<media>|الصوت)", "(گطع|اگطع) صوت <dc_media_player>", "<media> سكّتها"])
    add("HassMediaPlayerUnmute", ["ردّ صوت <dc_media_player>", "ردّ الصوت", "خلّي <media> يتكلم"])
    add("HassCancelAllTimers", ["<v_cancel> (المنبّهات|التايمرات) <all>", "<v_cancel> <all> (المنبّهات|التايمرات)"])
    add("HassSetVolumeRelative", ["<v_increase> (الصوت|صوت <dc_media_player>) [" + alt(LEX["a_little"]["ar"] + LEX["very"]["ar"]) + "]"],
        {"volume_step": "up"})
    add("HassSetVolumeRelative", ["<v_decrease> (الصوت|صوت <dc_media_player>) [" + alt(LEX["a_little"]["ar"] + LEX["very"]["ar"]) + "]"],
        {"volume_step": "down"})
    add("HassFanSetSpeed", [f"<v_set> <d_fan> [<loc>] (ف|على) {num('percentage')} <pct_unit>",
                            f"<d_fan> [<loc>] {num('percentage')} <pct_unit>"], response="hsn_set")
    # ---- timers
    durations = [
        ("{hsn_count:minutes} دگايگ", {}), ("{hsn_number:minutes} دقيقة", {}), ("{hsn_digits:minutes} دقيقة", {}),
        ("{hsn_digits:minutes} دگايگ", {}), ("دقيقة", {"minutes": 1}), ("دقيقتين", {"minutes": 2}),
        ("(نص ساعة|انصّ ساعة|انص ساعة)", {"minutes": 30}), ("ربع ساعة", {"minutes": 15}),
        ("ساعة", {"hours": 1}), ("ساعتين", {"hours": 2}), ("{hsn_count:hours} ساعات", {}),
        ("{hsn_count:seconds} ثواني", {}), ("{hsn_digits:seconds} ثانية", {}),
    ]
    for dur, fixed in durations:
        add("HassStartTimer", [f"<v_start_timer> <timer_word> [ل] {dur} [<purpose>]",
                               f"(نبّهني|نبهني|گول لي) بعد {dur} [<purpose>]",
                               f"{dur} [<purpose>] <v_start_timer> <timer_word>"], fixed or None, response="hsn_timer")
    add("HassCancelTimer", ["<v_cancel> <timer_word_def>"])
    add("HassTimerStatus", ["<how_much> باگي ف <timer_word_def>", "<timer_word_def> <how_much> باگي لو",
                            "<how_much> باگي من الوگت"])
    # ---- info
    add("HassGetCurrentTime", G.TIME_Q)
    add("HassGetCurrentDate", G.DATE_Q)
    add("HassGetWeather", G.WEATHER_Q)
    # ---- shopping / vacuum / nevermind
    add_obj = alt([w + "ه" for w in LEX["verbs"]["add"]["imp_m"] + LEX["verbs"]["add"]["imp_f"] if " " not in w])
    add("HassShoppingListAddItem", ["<v_add> {hsn_item:item} (ل|ف) <list_word>",
                                    f"[ما ننساو] {{hsn_item:item}} {add_obj} ف <list_word>"], response="hsn_item")
    add("HassGetState", ["[<is_it>] <d_vacuum> (گادي|گادية|كادي|شاعل|شاعلة|يخدم)"], {"domain": "vacuum", "state": "on"})
    add("HassVacuumStart", ["<v_clean> (الدار|{hsn_area:area})", "خلّي الروبو (ينظّف|يشطّب)", "(اگدي|حلّ|شعّل) الروبو"])
    add("HassVacuumReturnToBase", ["<v_return> الروبو (لبلاصتو|لامكانو|للشارجور)", "<v_return> المكنسة للشارجور",
                                   "خلّي الروبو يرجع لبلاصتو"])
    add("HassNevermind", G.NEVERMIND_Q)

    lists = {
        # every room also without its article ("ظو صالو", "ظو كوزينة")
        "hsn_area": {"values": [{"in": v, "out": a["ar"][0]} for a in LEX["areas"]
                                for v in dict.fromkeys(a["ar"] + [strip_tashkeel(x) for x in a["ar"]]
                                                       + [strip_al(strip_tashkeel(x)) for x in a["ar"] if x.startswith("ال")])]},
        "hsn_color": {"values": [{"in": v, "out": c["en"]} for c in LEX["colors"] for v in c["ar"]]},
        "hsn_number": {"values": number_values(0, 100)},
        "hsn_digits": {"range": {"from": 0, "to": 100}},
        "hsn_count": {"values": [{"in": v, "out": n} for n, w in COUNT_SHORT.items() for v in {w, strip_tashkeel(w)}]
                      + [{"in": str(n), "out": n} for n in range(3, 11)]},
        "hsn_item": {"wildcard": True},
    }
    responses = {"intents": {}}
    resp = {
        "hsn_on": "ايوه، گدّيتو.", "hsn_off": "ايوه، طفّيتو.", "hsn_lock": "ايوه، گفلت الباب.",
        "hsn_unlock": "ايوه، فتحت الباب.", "hsn_set": "ايوه، عدلتو.", "hsn_timer": "ايوه، المنبّه بدا.",
        "hsn_item": "زدت {{ slots.item }} ل ليستة الشرا.",
    }
    for intent, body in intents.items():
        used = {d.get("response") for d in body["data"]} - {None}
        if used:
            responses["intents"][intent] = {r: resp[r] for r in used}
    return {"language": "ar", "intents": intents, "lists": lists, "expansion_rules": R, "responses": responses}


def main():
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = ("# AUTO-GENERATED by scripts/build_ha_sentences.py from data/lexicon/lexicon.yaml - edit the lexicon, not this.\n"
              "# Hassaniya (Mauritanian Arabic) custom sentences for Home Assistant Assist (language: ar).\n"
              "# Area names: name your HA areas with the `out` values of the hsn_area list (or edit them here).\n")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=10_000)
    n = sum(len(d["sentences"]) for b in doc["intents"].values() for d in b["data"])
    print(f"wrote {OUT.relative_to(ROOT)}: {len(doc['intents'])} intents, {n} templates")


if __name__ == "__main__":
    main()
