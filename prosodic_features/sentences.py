"""Sentences of a recording, rebuilt from the words of its step-02 transcript.

All sentence wrangling happens here, after WhisperX (step 02 only transcribes,
aligns the words, assigns speakers and fits the word / chunk edges), so the
rules can change without transcribing again. Step 02's own grouping of the
words into segments is ignored. The rules, in order:

  1. Replacements (config.ABBREVIATIONS), word by word, before anything else:
     an ellipsis ("..." or "…") becomes a comma; an abbreviation is written
     out (Mr. -> Mister, U.S. -> United States); "No." before a number is
     "Number"; other dotted acronyms lose their dots (D.C. -> DC); a single
     initial loses its period (Martin L. King -> Martin L King), except "I."
     and "X." (Malcolm X.), which are words. The word keeps its timing, so
     "United States" is one timed word.
  2. A word ending in . ? ! (also before a closing quote or bracket) ends
     the sentence. A title (config.ABBREVIATION_TITLES: Mr., Dr., St., ...)
     never ends one; another abbreviation or acronym (U.S., D.C., Jr., ...)
     ends it if the next word starts with a capital letter (its period is
     then kept: "... in Washington, DC." / "We ...").
  3. A sentence that ends is never joined with the next one.
  4. A change of speaker always ends the sentence, so every sentence has one
     speaker; a sentence cut off this way (an interruption) without . ? !
     gets a "." (a trailing , ; : is replaced). So does the recording's
     last sentence.
  5. No pause rule: a pause alone never ends a sentence.
  6. Chunks of the edge words: a sentence's first word that is in no
     chunk (it fell in a gap) belongs to the next chunk, its last word to
     the previous chunk; other words without a chunk stay without one.
  7. Chunks (fit_sentences_to_chunks): a sentence can span chunks, but a
     piece shorter than SENTENCE_FRAGMENT_MAX_S in a chunk shared with the
     neighbouring sentence is an alignment fragment and the sentence leaves
     that chunk to its neighbour, unless the earlier of the two sentences
     ends in . ? ! (then nothing moves across that boundary); per chunk, the first sentence that takes
     it starts at the chunk's start and the last one ends at its end (or at
     its first / last word, if that lies outside the chunk).
The chunk table
of step 05 uses the same words and chunks (segments.load_chunks)."""
from __future__ import annotations

import re

from prosodic_features import config
from prosodic_features.utils import safe_float

_LEAD = re.compile(r'^["\'(\[]+')
_TRAIL = re.compile(r'[,;:"\')\]]+$')
_END = re.compile(r'[.?!]["\')\]]*$')
_ACRONYM = re.compile(r'(?:[A-Za-z]\.){2,}')
_INITIAL = re.compile(r'[A-Z]\.')
_NOT_INITIALS = {"I.", "X."}


def _split_token(text: str) -> tuple[str, str, str]:
    """(leading quotes/brackets, core, trailing , ; : quotes/brackets)."""
    lead = _LEAD.match(text)
    lead = lead.group(0) if lead else ""
    rest = text[len(lead):]
    trail = _TRAIL.search(rest)
    trail = trail.group(0) if trail else ""
    return lead, rest[:len(rest) - len(trail)], trail


def expand_word(text: str, next_text: str = "") -> tuple[str, str]:
    """Rule 1 for one word. Returns (new text, kind): kind is "title" (never
    ends a sentence), "abbrev" (ends one only before a capitalised word) or
    "" (an ordinary word: ends one if it ends in . ? !)."""
    for old, new in config.ABBREVIATIONS.items():          # punctuation entries: anywhere
        if not old[:1].isalnum() and old in text:
            text = re.sub(r"(?:\s*" + re.escape(old) + r")+", new, text)
    text = re.sub(r",{2,}", ",", text)
    lead, core, trail = _split_token(text)
    if core in config.ABBREVIATIONS:
        kind = "title" if core in config.ABBREVIATION_TITLES else "abbrev"
        return lead + config.ABBREVIATIONS[core] + trail, kind
    if core == "No." and next_text[:1].isdigit():
        return lead + "Number" + trail, "title"
    if _ACRONYM.fullmatch(core):
        return lead + core.replace(".", "") + trail, "abbrev"
    if _INITIAL.fullmatch(core) and core not in _NOT_INITIALS:
        return lead + core[0] + trail, "title"
    return text, ""


def close_with_period(text: str) -> str:
    """text ending in . ? ! (before closing quotes / brackets) as it is;
    otherwise a trailing , ; : is dropped and a "." added (before any closing
    quotes / brackets)."""
    if _END.search(text):
        return text
    m = re.search(r'["\')\]]*$', text)
    body, close = text[:m.start()], m.group(0)
    return body.rstrip(",;: ") + "." + close


def _words(transcript: dict, chunks: list[tuple[float, float]]) -> list[dict]:
    """All words with valid times, by start, rule 1 applied; each with its
    chunk (from step 02, else the chunk containing its midpoint)."""
    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            ws, we = safe_float(w.get("start")), safe_float(w.get("end"))
            txt = (w.get("word") or "").strip()
            if ws is None or we is None or not txt:
                continue
            k = w.get("chunk")
            if k is None or not (0 <= k < len(chunks)):
                mid = 0.5 * (ws + we)
                k = next((i for i, (cs, ce) in enumerate(chunks) if cs <= mid <= ce), None)
            words.append({"start": ws, "end": we, "word": txt, "chunk": k,
                          "speaker": w.get("speaker") or config.DEFAULT_SPEAKER})
    words.sort(key=lambda w: w["start"])
    for i, w in enumerate(words):
        nxt = words[i + 1]["word"] if i + 1 < len(words) else ""
        w["word"], w["kind"] = expand_word(w["word"], nxt)
    for i, w in enumerate(words):                           # rule 2 for abbreviations
        nxt = words[i + 1]["word"] if i + 1 < len(words) else ""
        lead, core, trail = _split_token(w["word"])
        w["ends"] = (bool(_END.search(w["word"])) if w["kind"] == "" else
                     w["kind"] == "abbrev" and (not nxt or _split_token(nxt)[1][:1].isupper()))
        if w["ends"] and w["kind"] == "abbrev" and not _END.search(w["word"]):
            w["word"] = lead + core + "." + trail
    return words


def fit_sentences_to_chunks(sentences: list[dict], chunks: list[tuple[float, float]],
                            max_fragment: float = config.SENTENCE_FRAGMENT_MAX_S) -> int:
    """Rule 7, IN PLACE: each sentence's chunks, start and end. Word times
    are not changed. A fragment moves only across a boundary whose earlier
    sentence does not end in . ? !. Returns how many fragments were left to
    a neighbour."""
    def ended(s):
        return bool(_END.search(s["words"][-1]["word"]))

    def piece(s, k):
        ws = [w for w in s["words"] if w["chunk"] == k]
        return ws[-1]["end"] - ws[0]["start"] if ws else 0.0

    own = [sorted({w["chunk"] for w in s["words"] if w["chunk"] is not None}) for s in sentences]
    n_fragments = 0
    for i, s in enumerate(sentences):
        ks = own[i]
        if len(ks) < 2:
            continue
        if i > 0 and not ended(sentences[i - 1]) and ks[0] in own[i - 1] \
                and piece(s, ks[0]) < max_fragment:
            ks = ks[1:]
            n_fragments += 1
        if i + 1 < len(sentences) and not ended(s) and ks[-1] in own[i + 1] and len(ks) > 1 \
                and piece(s, ks[-1]) < max_fragment:
            ks = ks[:-1]
            n_fragments += 1
        own[i] = ks
    takers: dict[int, list[int]] = {}
    for i, ks in enumerate(own):
        for k in ks:
            takers.setdefault(k, []).append(i)
    for i, s in enumerate(sentences):
        ks = own[i]
        s["chunks"] = ks
        in_own = [w for w in s["words"] if w["chunk"] in ks] or s["words"]
        s["start"], s["end"] = in_own[0]["start"], in_own[-1]["end"]
        if ks:                     # snap to the chunk edges, never cutting off its own words
            if takers[ks[0]][0] == i:
                s["start"] = min(chunks[ks[0]][0], s["words"][0]["start"])
            if takers[ks[-1]][-1] == i:
                s["end"] = max(chunks[ks[-1]][1], s["words"][-1]["end"])
    return n_fragments


def build_sentences(transcript: dict, chunks: list[tuple[float, float]]) -> list[dict]:
    """The recording's sentences by rules 1-7, by start, as
    [{start, end, text, words, chunks, speaker, speaker_pure}] (the shape of
    step 02's segments); words are [{start, end, word, chunk, speaker}]."""
    sentences, cur = [], []
    for w in _words(transcript, chunks):
        if cur and w["speaker"] != cur[-1]["speaker"]:        # rule 4
            sentences.append(cur)
            cur = []
        cur.append(w)
        if w["ends"]:                                         # rules 2-3
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    for ws in sentences:                                      # rule 4: interrupted -> "."
        ws[-1]["word"] = close_with_period(ws[-1]["word"])
    for ws in sentences:                                      # rule 6
        if ws[0]["chunk"] is None:
            ws[0]["chunk"] = next((k for k, (cs, _) in enumerate(chunks) if cs >= ws[0]["start"]), None)
        if ws[-1]["chunk"] is None:
            ws[-1]["chunk"] = next((k for k in range(len(chunks) - 1, -1, -1)
                                    if chunks[k][1] <= ws[-1]["end"]), None)
    out = [{"words": [{k: w[k] for k in ("start", "end", "word", "chunk", "speaker")} for w in ws],
            "text": " ".join(w["word"] for w in ws),
            "speaker": ws[0]["speaker"], "speaker_pure": True} for ws in sentences]
    fit_sentences_to_chunks(out, chunks)
    return out
