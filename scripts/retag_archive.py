#!/usr/bin/env python3

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import tomli
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "content" / "article"

THEMATIC_TAGS = [
    "австрийская школа и праксеология",
    "капитализм, рынки и предпринимательство",
    "деньги и банковская система",
    "инфляция, дефляция и денежная политика",
    "биткоин, криптовалюты и CBDC",
    "государство, бюрократия и регуляторы",
    "налоги, долг и госрасходы",
    "социализм, марксизм и антикапитализм",
    "частная собственность и права",
    "частное право, безопасность и частное управление",
    "демократия, конституции и политические институты",
    "сецессия, децентрализация и федерализм",
    "война, империя и внешняя политика",
    "история свободы и экономическая история",
    "COVID-19, локдауны и вакцинная политика",
    "здравоохранение, фарма и медицинское регулирование",
    "климат, экология и энергетика",
    "наука, экспертиза и данные",
    "технологии, ИИ и цифровой контроль",
    "культура, идеология, образование и свобода слова",
]


@dataclass(frozen=True)
class Rule:
    tag: str
    strong: tuple[str, ...]
    medium: tuple[str, ...] = ()
    weak: tuple[str, ...] = ()


RULES = [
    Rule(
        "австрийская школа и праксеология",
        strong=(
            "австрийск",
            "праксеолог",
            "праксиолог",
            "человеческ действие",
            "human action",
            "людвиг фон мизес",
            "мюррей ротбард",
            "фридрих хайек",
            "карл менгер",
            "кирцнер",
        ),
        medium=("мизес", "ротбард", "хайек", "менгер", "субъектив", "методолог"),
        weak=("каталлактик", "каталактик", "предпочтени", "маржинал"),
    ),
    Rule(
        "капитализм, рынки и предпринимательство",
        strong=(
            "капитализм",
            "свободн рынок",
            "предприним",
            "конкуренц",
            "монопол",
            "прибыл",
            "убыт",
            "ценообраз",
            "торговл",
            "бизнес",
            "потребител",
        ),
        medium=("рынок", "рынк", "коммерц", "обмен", "фирм", "цена"),
        weak=("координац", "предприят", "торгов"),
    ),
    Rule(
        "деньги и банковская система",
        strong=(
            "денег",
            "деньг",
            "банков",
            "банк ",
            "банка ",
            "банки ",
            "кредит",
            "центральн банк",
            "центробанк",
            "золотой стандарт",
            "серебрян",
            "золот",
            "фиат",
            "валют",
            "депозит",
            "free banking",
        ),
        medium=("резерв", "фрс", "монетарн", "монетизац"),
    ),
    Rule(
        "инфляция, дефляция и денежная политика",
        strong=(
            "инфляц",
            "дефляц",
            "денежн политика",
            "процентн став",
            "таргетирован",
            "денежн масс",
            "cpi",
            "рост цен",
            "количественн смягч",
            "qe",
        ),
        medium=("фед", "ставк", "облигаци", "цб", "потребительск цен"),
        weak=("монетар",),
    ),
    Rule(
        "биткоин, криптовалюты и CBDC",
        strong=(
            "биткоин",
            "bitcoiners",
            "bitcoin",
            "криптов",
            "крипто",
            "блокчейн",
            "lightning",
            "cbdc",
            "стейблкоин",
            "stablecoin",
            "сатош",
            "ethereum",
            "эфириум",
        ),
    ),
    Rule(
        "государство, бюрократия и регуляторы",
        strong=(
            "государств",
            "бюрократ",
            "регулятор",
            "регулирован",
            "правительств",
            "чинов",
            "агентств",
            "лиценз",
            "министер",
            "госаппарат",
            "административн",
        ),
        medium=("ведомств", "аппарат", "власть", "госслужб"),
    ),
    Rule(
        "налоги, долг и госрасходы",
        strong=(
            "налог",
            "ндс",
            "акциз",
            "госдолг",
            "государственн долг",
            "дефицит",
            "бюджет",
            "госрасход",
            "расходы государства",
            "субсид",
            "трансферт",
        ),
        medium=("долг", "расход", "спендинг", "spending"),
    ),
    Rule(
        "социализм, марксизм и антикапитализм",
        strong=(
            "социализм",
            "социалист",
            "маркс",
            "марксизм",
            "коммуниз",
            "антикапитал",
            "плановая экономик",
            "планировани",
            "коллективизм",
        ),
        medium=("перераспредел", "социал", "коммунист"),
    ),
    Rule(
        "частная собственность и права",
        strong=(
            "частн собствен",
            "право собственности",
            "собственност",
            "естественн прав",
            "самопринадлеж",
            "экспропри",
        ),
        medium=("контракт", "договор", "имущественн прав"),
        weak=("присвоени",),
    ),
    Rule(
        "частное право, безопасность и частное управление",
        strong=(
            "анарх",
            "частн право",
            "free private cities",
            "private city",
            "частн управлен",
            "private governance",
            "без государства",
            "общественн благ",
            "public goods",
            "арбитраж",
            "lex mercatoria",
        ),
        medium=("частн полиц", "частн охран", "дорог", "спонтанн поряд"),
    ),
    Rule(
        "демократия, конституции и политические институты",
        strong=(
            "демократ",
            "конституц",
            "выбор",
            "голосован",
            "парламент",
            "республик",
            "президент",
            "электорат",
            "легитим",
            "политическ институт",
            "верховный суд",
        ),
        medium=("сенат", "судебн", "конгресс"),
    ),
    Rule(
        "сецессия, децентрализация и федерализм",
        strong=(
            "сецесс",
            "децентрализ",
            "федерализ",
            "отделен",
            "автоном",
            "локализм",
            "nations by consent",
        ),
        medium=("конфедерац", "нуллификац", "локал"),
    ),
    Rule(
        "война, империя и внешняя политика",
        strong=(
            "войн",
            "военн",
            "импер",
            "империал",
            "внешн полит",
            "геополит",
            "нато",
            "вторжен",
            "санкц",
            "оруж",
            "армия",
            "украин",
            "геноцид",
        ),
        medium=("росси", "путин", "дипломат", "союзник"),
    ),
    Rule(
        "история свободы и экономическая история",
        strong=(
            "история борьбы за свободу",
            "the fall of tyranny the rise of liberty",
            "экономическ истори",
            "история",
            "историческ",
            "древн",
            "средневек",
            "колониальн",
            "цивилизац",
            "революц",
            "рабств",
        ),
        medium=("археолог", "античн", "викториан", "биография", "цивилизация инда"),
    ),
    Rule(
        "COVID-19, локдауны и вакцинная политика",
        strong=(
            "covid",
            "ковид",
            "коронав",
            "sars-cov",
            "пандем",
            "локдаун",
            "карантин",
            "вакцин",
            "маск",
            "бустер",
            "избыточн смерт",
        ),
        medium=("вирус", "социальн дистанц", "эпидем"),
    ),
    Rule(
        "здравоохранение, фарма и медицинское регулирование",
        strong=(
            "здравоохран",
            "медицин",
            "медицин",
            "врач",
            "больниц",
            "фарма",
            "фарм",
            "fda",
            "medicare",
            "medicaid",
            "пациент",
            "лекарств",
            "hospital",
            "health care",
        ),
        medium=("страхован", "страхов", "терап", "диагноз"),
    ),
    Rule(
        "климат, экология и энергетика",
        strong=(
            "климат",
            "эколог",
            "энерг",
            "углерод",
            "carbon",
            "выброс",
            "нефт",
            "газ",
            "солнечн",
            "ветрян",
            "атомн",
            "esg",
            "потеплен",
        ),
        medium=("ископаем", "топлив", "электроэнерг"),
    ),
    Rule(
        "наука, экспертиза и данные",
        strong=(
            "наук",
            "учен",
            "эксперт",
            "статист",
            "данн",
            "модел",
            "исследован",
            "peer review",
            "консенсус",
            "доказательств",
            "журнал",
            "метаанализ",
        ),
        medium=("экспертиз", "прогноз", "сциентиз", "метрик"),
    ),
    Rule(
        "технологии, ИИ и цифровой контроль",
        strong=(
            "искусственн интеллект",
            "artificial intelligence",
            " ии ",
            " ai ",
            "технолог",
            "цифров",
            "интернет",
            "соцсет",
            "алгоритм",
            "надзор",
            "surveillance",
            "слежк",
            "шпиони",
            "наблюдени",
            "big tech",
            "робот",
            "автоматизац",
            "privacy",
            "chatgpt",
        ),
        medium=("конфиденциаль", "платформ", "диджитал", "цифровой контроль"),
    ),
    Rule(
        "культура, идеология, образование и свобода слова",
        strong=(
            "идеолог",
            "культур",
            "образован",
            "университет",
            "школ",
            "цензур",
            "свобода слова",
            "free speech",
            "woke",
            "феминиз",
            "расизм",
            "дискриминац",
            "пропаганд",
            "журналист",
            "медиа",
        ),
        medium=("студент", "кампус", "культурн войн", "политкоррект"),
    ),
]


STRONG_ONLY = {
    "государство, бюрократия и регуляторы",
    "наука, экспертиза и данные",
    "культура, идеология, образование и свобода слова",
}

COMPATIBLE_GROUPS = {
    frozenset(
        {
            "деньги и банковская система",
            "инфляция, дефляция и денежная политика",
        }
    ),
    frozenset(
        {
            "COVID-19, локдауны и вакцинная политика",
            "здравоохранение, фарма и медицинское регулирование",
            "наука, экспертиза и данные",
        }
    ),
    frozenset(
        {
            "частная собственность и права",
            "частное право, безопасность и частное управление",
        }
    ),
    frozenset(
        {
            "война, империя и внешняя политика",
            "история свободы и экономическая история",
        }
    ),
    frozenset(
        {
            "государство, бюрократия и регуляторы",
            "демократия, конституции и политические институты",
        }
    ),
    frozenset(
        {
            "технологии, ИИ и цифровой контроль",
            "культура, идеология, образование и свобода слова",
        }
    ),
}

SORT_PRIORITY = {
    "COVID-19, локдауны и вакцинная политика": 1,
    "биткоин, криптовалюты и CBDC": 2,
    "частное право, безопасность и частное управление": 3,
    "технологии, ИИ и цифровой контроль": 4,
    "климат, экология и энергетика": 5,
    "здравоохранение, фарма и медицинское регулирование": 6,
    "австрийская школа и праксеология": 7,
    "деньги и банковская система": 8,
    "инфляция, дефляция и денежная политика": 9,
    "налоги, долг и госрасходы": 10,
    "частная собственность и права": 11,
    "социализм, марксизм и антикапитализм": 12,
    "сецессия, децентрализация и федерализм": 13,
    "демократия, конституции и политические институты": 14,
    "война, империя и внешняя политика": 15,
    "история свободы и экономическая история": 16,
    "наука, экспертиза и данные": 17,
    "культура, идеология, образование и свобода слова": 18,
    "государство, бюрократия и регуляторы": 19,
    "капитализм, рынки и предпринимательство": 20,
}


def normalize(text: str) -> str:
    replacements = {
        "ё": "е",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        "\xa0": " ",
    }
    out = text.lower()
    for source, target in replacements.items():
        out = out.replace(source, target)
    out = html.unescape(out)
    out = re.sub(r"https?://\S+", " ", out)
    out = re.sub(r"<[^>]+>", " ", out)
    out = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", out)
    out = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", out)
    out = re.sub(r"[`*_>#~]", " ", out)
    out = re.sub(r"\s+", " ", out)
    return f" {out.strip()} "


def to_string(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def extract_front_matter(path: Path) -> tuple[str, dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"^(---|\+\+\+)\n(.*?)\n\1\n", raw, re.S)
    if not match:
        raise ValueError(f"Front matter not found in {path}")
    delim, front_matter = match.group(1), match.group(2)
    body = raw[match.end() :]
    if delim == "---":
        data = yaml.safe_load(front_matter) or {}
    else:
        data = tomli.loads(front_matter)
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported front matter in {path}")
    return delim, data, body


def serialize_yaml_value(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return json.dumps([to_string(item) for item in value], ensure_ascii=False)
    if key == "date":
        return to_string(value)
    return json.dumps(to_string(value), ensure_ascii=False)


def serialize_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return json.dumps([to_string(item) for item in value], ensure_ascii=False)
    return json.dumps(to_string(value), ensure_ascii=False)


def dump_front_matter(delim: str, data: dict[str, Any]) -> str:
    preferred = ["title", "author", "authors", "date", "tags", "summary", "draft"]
    ordered_keys: list[str] = []
    for key in preferred:
        if key in data:
            ordered_keys.append(key)
    for key in data:
        if key not in ordered_keys:
            ordered_keys.append(key)

    lines = [delim]
    if delim == "---":
        for key in ordered_keys:
            lines.append(f"{key}: {serialize_yaml_value(key, data[key])}")
    else:
        for key in ordered_keys:
            lines.append(f"{key} = {serialize_toml_value(data[key])}")
    lines.append(delim)
    return "\n".join(lines) + "\n"


def normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", normalize(text)).strip()


def looks_like_credential(segment: str) -> bool:
    token = normalize_for_compare(segment)
    exact_markers = {
        "доктор медицины",
        "доктор философии",
        "доктор юридических наук",
        "доктор наук",
        "магистр права",
        "кандидат наук",
        "профессор",
        "phd",
        "md",
        "m.d.",
        "doctor of medicine",
        "doctor of philosophy",
        "doctor of law",
    }
    if token in exact_markers:
        return True
    generic_markers = (
        "доктор ",
        "магистр ",
        "кандидат ",
        "doctor of ",
    )
    return any(token.startswith(marker) for marker in generic_markers)


def clean_author_name(name: str) -> str:
    cleaned = name.strip()
    if "," in cleaned:
        first, rest = cleaned.split(",", 1)
        if looks_like_credential(rest.strip()):
            cleaned = first.strip()
    return cleaned


def split_author_segment(segment: str) -> list[str]:
    segment = segment.strip()
    if not segment:
        return []

    shared_surname = re.match(r"^([^\s,]+)\s+и\s+([^\s,]+)\s+(.+)$", segment)
    if shared_surname and " " not in shared_surname.group(3).strip():
        surname = shared_surname.group(3).strip()
        return [
            f"{shared_surname.group(1).strip()} {surname}",
            f"{shared_surname.group(2).strip()} {surname}",
        ]

    if re.search(r"\s(?:and|&|и|та)\s", segment):
        parts = [part.strip() for part in re.split(r"\s(?:and|&|и|та)\s", segment) if part.strip()]
        if all(len(part.split()) >= 2 or part.lower().startswith("д-р ") for part in parts):
            return parts

    return [segment]


def parse_authors(author_value: Any) -> list[str]:
    if isinstance(author_value, list):
        raw_items = [to_string(item) for item in author_value]
    else:
        raw = to_string(author_value).strip()
        if not raw:
            return []
        raw_items = []
        for chunk in raw.split(";"):
            pieces = []
            for token in [part.strip() for part in chunk.split(",")]:
                if not token:
                    continue
                if pieces and looks_like_credential(token):
                    pieces[-1] = f"{pieces[-1]}, {token}"
                else:
                    pieces.append(token)
            raw_items.extend(pieces)

    authors: list[str] = []
    for item in raw_items:
        for author in split_author_segment(item):
            author = clean_author_name(author)
            if author and author not in authors:
                authors.append(author)
    return authors


def clean_existing_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        tags = [to_string(item).strip() for item in value]
    elif value is None:
        tags = []
    else:
        tags = [to_string(value).strip()]
    cleaned: list[str] = []
    for tag in tags:
        if tag and tag not in cleaned:
            cleaned.append(tag)
    return cleaned


def tag_matches_author(tag: str, author_raw: str, authors: list[str]) -> bool:
    tag_norm = normalize_for_compare(tag)
    if not tag_norm:
        return True
    candidates = [author_raw, clean_author_name(author_raw), *authors]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_norm = normalize_for_compare(candidate)
        if tag_norm == candidate_norm:
            return True
        if tag_norm.split() and candidate_norm.split() and SequenceMatcher(None, tag_norm, candidate_norm).ratio() >= 0.96:
            return True
    return False


def score_rule(rule: Rule, title: str, summary: str, tags_text: str, body: str) -> int:
    title_score = 0
    tags_score = 0
    summary_score = 0
    body_score = 0

    for needle in rule.strong:
        if needle in title:
            title_score += 8
        elif needle in tags_text:
            tags_score += 6
        elif needle in summary:
            summary_score += 4
        elif needle in body:
            body_score += 1
    for needle in rule.medium:
        if needle in title:
            title_score += 5
        elif needle in tags_text:
            tags_score += 4
        elif needle in summary:
            summary_score += 3
        elif needle in body:
            body_score += 1
    for needle in rule.weak:
        if needle in title:
            title_score += 3
        elif needle in tags_text or needle in summary:
            tags_score += 2
        elif needle in body:
            body_score += 1

    return min(title_score, 24) + min(tags_score, 12) + min(summary_score, 8) + min(body_score, 4)


def are_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    pair = frozenset({left, right})
    return pair in COMPATIBLE_GROUPS


def classify_thematic_tags(title: str, summary: str, preserved_tags: list[str], body: str) -> list[str]:
    normalized_title = normalize(title)
    normalized_summary = normalize(summary)
    normalized_tags = normalize(" ".join(preserved_tags))
    normalized_body = normalize(body[:24000])

    scores: dict[str, int] = {}
    for rule in RULES:
        score = score_rule(rule, normalized_title, normalized_summary, normalized_tags, normalized_body)
        if score:
            scores[rule.tag] = score

    if "деньги и банковская система" in scores and "инфляция, дефляция и денежная политика" in scores:
        scores["инфляция, дефляция и денежная политика"] += 2

    if "COVID-19, локдауны и вакцинная политика" in scores and "наука, экспертиза и данные" in scores:
        scores["наука, экспертиза и данные"] += 1

    if (
        "история свободы и экономическая история" in scores
        and "война, империя и внешняя политика" in scores
    ):
        scores["история свободы и экономическая история"] += 1

    ranked = sorted(scores.items(), key=lambda item: (-item[1], SORT_PRIORITY[item[0]]))
    if not ranked:
        fallback_text = f"{normalized_title} {normalized_summary} {normalized_tags}"
        if "истори" in fallback_text or "цивилизац" in fallback_text:
            return ["история свободы и экономическая история"]
        if "государств" in fallback_text or "правительств" in fallback_text:
            return ["государство, бюрократия и регуляторы"]
        if "рын" in fallback_text or "эконом" in fallback_text:
            return ["капитализм, рынки и предпринимательство"]
        return ["государство, бюрократия и регуляторы"]

    chosen = [ranked[0][0]]
    for tag, score in ranked[1:]:
        if len(chosen) >= 3:
            break
        threshold = 6 if tag in STRONG_ONLY else 4
        if score < threshold:
            continue
        if score >= 8:
            chosen.append(tag)
            continue
        if ranked[0][1] < 6 and not any(are_compatible(tag, current) for current in chosen):
            continue
        if score + 3 < ranked[0][1] and not any(are_compatible(tag, current) for current in chosen):
            continue
        if any(are_compatible(tag, current) for current in chosen) or score >= ranked[0][1] - 2:
            chosen.append(tag)

    return chosen


def build_updated_front_matter(data: dict[str, Any], thematic_tags: list[str], preserved_tags: list[str], authors: list[str]) -> dict[str, Any]:
    updated = dict(data)
    updated["tags"] = thematic_tags + [tag for tag in preserved_tags if tag not in thematic_tags]
    if authors:
        updated["authors"] = authors
    elif "authors" in updated:
        updated.pop("authors")
    return updated


def process_file(path: Path, apply_changes: bool) -> dict[str, Any] | None:
    delim, data, body = extract_front_matter(path)
    title = to_string(data.get("title", "")).strip()
    if not title:
        return None

    author_raw = to_string(data.get("author", "")).strip()
    authors = parse_authors(data.get("author", ""))
    existing_tags = clean_existing_tags(data.get("tags", []))
    preserved_tags = [
        tag for tag in existing_tags if not tag_matches_author(tag, author_raw=author_raw, authors=authors)
    ]
    summary = to_string(data.get("summary", "")).strip()
    thematic_tags = classify_thematic_tags(title=title, summary=summary, preserved_tags=preserved_tags, body=body)
    updated = build_updated_front_matter(data=data, thematic_tags=thematic_tags, preserved_tags=preserved_tags, authors=authors)

    if apply_changes:
        rendered = dump_front_matter(delim, updated) + body
        path.write_text(rendered, encoding="utf-8")

    return {
        "path": path,
        "authors": authors,
        "thematic_tags": thematic_tags,
        "preserved_tags": preserved_tags,
        "updated_tags": updated["tags"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retag the Hugo article archive with thematic tags and authors taxonomy.")
    parser.add_argument("--apply", action="store_true", help="Write updated front matter back to content files.")
    args = parser.parse_args()

    thematic_counts: Counter[str] = Counter()
    preserved_count = 0
    authors_count = 0
    article_count = 0
    skipped: list[str] = []
    multi_tag_counts: Counter[int] = Counter()

    for path in sorted(CONTENT_DIR.glob("*.md")):
        try:
            result = process_file(path, apply_changes=args.apply)
        except ValueError:
            skipped.append(path.name)
            continue
        if result is None:
            skipped.append(path.name)
            continue
        article_count += 1
        for tag in result["thematic_tags"]:
            thematic_counts[tag] += 1
        preserved_count += len(result["preserved_tags"])
        if result["authors"]:
            authors_count += 1
        multi_tag_counts[len(result["thematic_tags"])] += 1

    action = "updated" if args.apply else "analyzed"
    print(f"{action} {article_count} articles")
    print(f"articles with authors taxonomy: {authors_count}")
    print(f"preserved existing non-author tags: {preserved_count}")
    if skipped:
        print("skipped:", ", ".join(skipped))
    print("thematic tag counts:")
    for tag in THEMATIC_TAGS:
        print(f"  {thematic_counts[tag]:>4}  {tag}")
    print("articles by number of thematic tags:")
    for size in sorted(multi_tag_counts):
        print(f"  {size}: {multi_tag_counts[size]}")


if __name__ == "__main__":
    main()
