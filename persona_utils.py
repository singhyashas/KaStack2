def _top_claims(items, limit=5):
    sorted_items = sorted(items, key=lambda item: item.get("count", 0), reverse=True)
    return [
        {
            "claim": item.get("claim", ""),
            "count": item.get("count", 0),
            "confidence": item.get("confidence", ""),
        }
        for item in sorted_items[:limit]
    ]


def speaker_names(persona):
    return list(persona.get("speakers", {}).keys())


def get_speaker_profile(persona, speaker=None):
    speakers = persona.get("speakers", {})
    if not speakers:
        return {}
    selected = speaker if speaker in speakers else next(iter(speakers))
    return speakers[selected]


def summarize_round1_persona(persona, speaker=None):
    selected = speaker if speaker in persona.get("speakers", {}) else next(iter(persona.get("speakers", {"-": {}})))
    profile = get_speaker_profile(persona, selected)
    style = profile.get("communication_style", {})

    return {
        "schema_version": persona.get("schema_version", "-"),
        "source": persona.get("source", "-"),
        "speaker": selected,
        "message_count": style.get("message_count", 0),
        "average_content_words": style.get("average_content_words", 0),
        "question_rate": style.get("question_rate", 0),
        "exclamation_rate": style.get("exclamation_rate", 0),
        "short_message_rate": style.get("short_message_rate", 0),
        "style_notes": ", ".join(style.get("style_notes", [])),
        "top_terms": ", ".join(style.get("top_terms", [])[:8]),
        "top_traits": _top_claims(profile.get("personality_traits", []), 6),
        "top_facts": _top_claims(profile.get("personal_facts", []), 6),
        "top_preferences": _top_claims(profile.get("preferences", []), 6),
    }


def infer_baseline_tones(persona, speaker=None):
    profile = get_speaker_profile(persona, speaker)
    style = profile.get("communication_style", {})
    traits = {
        item.get("claim", "").lower()
        for item in profile.get("personality_traits", [])
    }

    tones = []
    if "curious" in traits or style.get("question_rate", 0) >= 0.18:
        tones.append("curious")
    if "supportive" in traits or "empathetic" in traits:
        tones.append("supportive")
    if "enthusiastic" in traits or style.get("exclamation_rate", 0) >= 0.35:
        tones.append("enthusiastic")
    if style.get("short_message_rate", 0) >= 0.35:
        tones.append("casual")
    if "family-oriented" in traits:
        tones.append("family-oriented")

    return tones[:4] or ["neutral"]
