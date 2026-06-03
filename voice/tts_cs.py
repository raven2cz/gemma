#!/usr/bin/env python3
"""Chatterbox-TTS-Czech: python tts_cs.py "text" out.wav [device] [ref.wav]

Rozdělí text na věty, každou TTS samostatně, spojí s krátkou mezerou.
Normalizuje číslice a zkratky, aby model nehláskoval písmena.
"""
import re
import sys
import threading
from pathlib import Path
import numpy as np
from num2words import num2words
import soundfile as sf
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
    MinPLogitsWarper,
)

import chatterbox
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES, punc_norm
from chatterbox.models.t3 import T3
from chatterbox.models.t3.t3 import _ensure_BOT_EOT
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from chatterbox.models.s3tokenizer import drop_invalid_tokens

# Monkey-patch na no-CFG cestu (batch=1 místo batch=2) — reálné ~2× zrychlení
# pro multilingual variantu, která to upstream nemá (anglická tts.py už má).
# Když `pip install -U chatterbox-tts` změní verzi, import sem spadne s hlášením
# → review patche proti nové verzi mtl_tts.py / t3.py.
# Legacy globální cancel event — používá ho /api/tts a legacy _tts_synth_blocking.
# Nové streaming /api/turn má per-turn threading.Event (viz `set_cancel_event`
# níž), aby cancel turnu A neshazoval souběžný turn B.
CANCEL_EVENT = threading.Event()

# Thread-local držák pro aktuální cancel event. Monkey-patchnutá inference
# smyčka ho přečte a checkne `is_set()` každý token. Caller (synth wrapper)
# nastaví event před model.generate a resetne v `finally`.
_cancel_local = threading.local()


def set_cancel_event(ev) -> None:
    """Naváže `ev` (threading.Event | None) na tento thread. Inference smyčka
    pak bude checkovat `ev.is_set()`. `None` odpojí."""
    _cancel_local.event = ev


class TTSCanceled(RuntimeError):
    """Vyhodí se z inference smyčky, když je setnut cancel event."""


_PINNED_CHATTERBOX = "0.1.7"
assert chatterbox.__version__ == _PINNED_CHATTERBOX, (
    f"chatterbox-tts {chatterbox.__version__} != pinned {_PINNED_CHATTERBOX}. "
    f"Review monkey-patch v tts_cs.py proti nové verzi mtl_tts.py / models/t3/t3.py "
    f"(hlavně ChatterboxMultilingualTTS.generate a T3.inference — hledej 'torch.cat([text_tokens, text_tokens]' "
    f"a 'batch_size=2 for CFG'). Pokud upstream opravil no-CFG cestu, patch zahoď."
)


@torch.inference_mode()
def _t3_inference_nocfg(
    self,
    *,
    t3_cond,
    text_tokens,
    initial_speech_tokens=None,
    prepend_prompt_speech_tokens=None,
    num_return_sequences=1,
    max_new_tokens=None,
    stop_on_eos=True,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    min_p=0.05,
    length_penalty=1.0,
    repetition_penalty=1.2,
    cfg_weight=0.5,
):
    """Patched T3.inference. Když cfg_weight==0, jede na batch=1 (2× rychleji)."""
    assert prepend_prompt_speech_tokens is None, "not implemented"
    _ensure_BOT_EOT(text_tokens, self.hp)
    text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)
    no_cfg = (float(cfg_weight) == 0.0)

    if initial_speech_tokens is None:
        initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

    embeds, len_cond = self.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=initial_speech_tokens,
        cfg_weight=cfg_weight,
    )

    self.compiled = False
    if not self.compiled:
        alignment_stream_analyzer = None
        if self.hp.is_multilingual:
            alignment_stream_analyzer = AlignmentStreamAnalyzer(
                self.tfmr, None,
                text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                alignment_layer_idx=9,
                eos_idx=self.hp.stop_speech_token,
            )
        patched_model = T3HuggingfaceBackend(
            config=self.cfg,
            llama=self.tfmr,
            speech_enc=self.speech_emb,
            speech_head=self.speech_head,
            alignment_stream_analyzer=alignment_stream_analyzer,
        )
        self.patched_model = patched_model
        self.compiled = True

    device = embeds.device
    bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
    bos_embed = self.speech_emb(bos_token)
    bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)
    if not no_cfg:
        bos_embed = torch.cat([bos_embed, bos_embed])
    inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

    generated_ids = bos_token.clone()
    predicted = []
    top_p_warper = TopPLogitsWarper(top_p=top_p)
    min_p_warper = MinPLogitsWarper(min_p=min_p)
    repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

    output = self.patched_model(
        inputs_embeds=inputs_embeds,
        past_key_values=None,
        use_cache=True,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = output.past_key_values

    for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
        _ev = getattr(_cancel_local, "event", None)
        if _ev is not None and _ev.is_set():
            raise TTSCanceled(f"canceled at token {i}")
        logits_step = output.logits[:, -1, :]
        if no_cfg:
            logits = logits_step[0:1, :]
        else:
            cond = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)

        if self.patched_model.alignment_stream_analyzer is not None:
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            last_token = generated_ids[0, -1].item() if len(generated_ids[0]) > 0 else None
            logits = self.patched_model.alignment_stream_analyzer.step(logits, next_token=last_token)

        ids_for_proc = generated_ids[:1, ...]
        logits = repetition_penalty_processor(ids_for_proc, logits)
        if temperature != 1.0:
            logits = logits / temperature
        logits = min_p_warper(ids_for_proc, logits)
        logits = top_p_warper(ids_for_proc, logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        predicted.append(next_token)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.view(-1) == self.hp.stop_speech_token:
            break

        next_token_embed = self.speech_emb(next_token)
        next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)
        if not no_cfg:
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

        output = self.patched_model(
            inputs_embeds=next_token_embed,
            past_key_values=past,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values

    predicted_tokens = torch.cat(predicted, dim=1)
    return predicted_tokens


def _mtl_generate_nocfg(
    self,
    text,
    language_id,
    audio_prompt_path=None,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    repetition_penalty=2.0,
    min_p=0.05,
    top_p=1.0,
):
    """Patched ChatterboxMultilingualTTS.generate. cfg_weight==0 → batch=1 cesta."""
    if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
        supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise ValueError(f"Unsupported language_id '{language_id}'. Supported: {supported_langs}")

    if audio_prompt_path:
        self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
    else:
        assert self.conds is not None, "Please prepare_conditionals first"

    if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
        _cond = self.conds.t3
        self.conds.t3 = T3Cond(
            speaker_emb=_cond.speaker_emb,
            cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)

    text = punc_norm(text)
    text_tokens = self.tokenizer.text_to_tokens(
        text, language_id=language_id.lower() if language_id else None
    ).to(self.device)

    if cfg_weight > 0.0:
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

    sot = self.t3.hp.start_text_token
    eot = self.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    with torch.inference_mode():
        speech_tokens = self.t3.inference(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=1000,
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
        speech_tokens = speech_tokens[0]
        speech_tokens = drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens.to(self.device)

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
    return torch.from_numpy(watermarked_wav).unsqueeze(0)


T3.inference = _t3_inference_nocfg
ChatterboxMultilingualTTS.generate = _mtl_generate_nocfg

HERE = Path(__file__).parent
CKPT = HERE / "chatterbox-cs" / "t3_cs.safetensors"
REF = HERE / "ref_female.wav"

# Digit-by-digit fallback (jen pro extrémně dlouhé řetězce číslic - IDs, hashe).
DIGITS_CS = {"0":"nula","1":"jedna","2":"dva","3":"tři","4":"čtyři","5":"pět","6":"šest","7":"sedm","8":"osm","9":"devět"}
ABBREV = {
    "atd.":"a tak dále", "apod.":"a podobně", "např.":"například", "tj.":"to jest",
    "tzv.":"takzvaný", "resp.":"respektive", "cca":"cirka", "cca.":"cirka",
    "kg":"kilogramů", "MB":"megabajtů", "GB":"gigabajtů", "km":"kilometrů",
    "Hz":"hertzů", "kHz":"kilohertzů", "VRAM":"vé ramka", "RAM":"ramka",
    "GPU":"gépéúčko", "CPU":"sépéúčko", "CS":"česky", "EN":"anglicky",
    "LLM":"el el em", "TTS":"té té es", "STT":"es té té",
}

# Symboly → CS slova (před číselnou normalizací, ať "70%" → "70 procent").
SYMBOLS_CS = {
    "%": " procent", "‰": " promile", "°C": " stupňů celsia", "°": " stupňů",
    "€": " eur", "$": " dolarů", "£": " liber", "&": " a ", "+": " plus ",
    "×": " krát ", "÷": " děleno ", "=": " rovná se ",
}

# Anglické termíny → fonetický přepis do češtiny. Chatterbox jede přes
# polský checkpoint (pl trik), takže anglická slova čte polsko/česky =
# šíleně. Tady nejčastější tech termíny. Rozšiřitelné. Case-insensitive
# match jako celá slova. Best-effort - nepokryje vše, ale ty nejčastější ano.
EN_PHONETIC_CS = {
    "email": "ímejl", "e-mail": "ímejl", "online": "onlajn", "offline": "oflajn",
    "software": "softvér", "hardware": "hardvér", "default": "defolt",
    "browser": "brauzr", "feature": "fíčr", "file": "fajl", "files": "fajly",
    "commit": "komit", "deploy": "deploj", "release": "rilís", "update": "apdejt",
    "upgrade": "apgrejd", "download": "daunloud", "upload": "aplout",
    "frontend": "frontend", "backend": "bekend", "framework": "frejmvork",
    "open source": "oupn sors", "open-source": "oupn sors", "issue": "išjú",
    "pull request": "pul rikvest", "merge": "mardž", "branch": "brenč",
    "username": "júzrnejm", "password": "pasvord", "login": "logyn",
    "cache": "keš", "queue": "kjú", "timeout": "tajmaut", "thread": "tred",
    "token": "token", "endpoint": "endpojnt", "request": "rikvest",
    "response": "respons", "debug": "dýbag", "build": "bild",
}

MAX_CHARS = 180  # Chatterbox 1000 kroků ~ 8 s audia ~ 150-200 znaků CS
MIN_CHARS = 12   # kratší chunky Chatterbox halucinuje (GitHub #97) → merge
PAUSE_MS = 120


def _int_to_cs(num_str: str) -> str:
    """Celé číslo → česká slova přes num2words. Velmi dlouhé řetězce (IDs,
    hashe, telefon) → po číslicích (číst 16místné číslo jako jeden číselný
    výraz je nesmysl). num2words selhání → digit fallback (nikdy nespadne)."""
    if len(num_str) > 12:
        return " ".join(DIGITS_CS[c] for c in num_str)
    try:
        return num2words(int(num_str), lang="cs")
    except Exception:
        return " ".join(DIGITS_CS.get(c, c) for c in num_str)


def _num_to_cs(m: "re.Match") -> str:
    """Match handler: desetinné (3.14 / 3,14) i celé. Desetinné jen bez mezer
    kolem oddělovače (chrání před '3, 4 a 5' = výčet, ne 3.4)."""
    s = m.group(0)
    if re.fullmatch(r"\d+[.,]\d+", s):
        try:
            return num2words(float(s.replace(",", ".")), lang="cs")
        except Exception:
            return s
    return _int_to_cs(s)


def _version_to_cs(m: "re.Match") -> str:
    """Verze / IP (0.1.7, 192.168.1.1) → každé číslo kardinálně, tečky jako
    'tečka'. Jinak by decimal regex rozbil '0.1.7' na 'nula celá jedna.sedm'."""
    parts = m.group(0).split(".")
    return " tečka ".join(_int_to_cs(p) for p in parts)


def _version_to_en(m: "re.Match") -> str:
    parts = m.group(0).split(".")
    spoken = []
    for p in parts:
        try:
            spoken.append(num2words(int(p), lang="en"))
        except Exception:
            spoken.append(p)
    return " dot ".join(spoken)


def _int_to_en(m: "re.Match") -> str:
    s = m.group(0)
    if re.fullmatch(r"\d+[.,]\d+", s):
        try:
            return num2words(float(s.replace(",", ".")), lang="en")
        except Exception:
            return s
    if len(s) > 12:
        return " ".join(s)
    try:
        return num2words(int(s), lang="en")
    except Exception:
        return s

ANSI_ESC = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

# Detekce jazyka — heuristika CZ/EN. Silný signál = CZ diakritika; fallback = stopwords
# hlasování. Krátký text → `prev` (hysterese proti oscilaci).
CS_DIACRITICS = set("ěščřžýáíéůúňďťóĚŠČŘŽÝÁÍÉŮÚŇĎŤÓ")
CS_STOPWORDS = {
    # POZOR: žádné slovo které je běžné i v EN (např. "to" = CS zájmeno + EN
    # předložka). Jinak detect_lang na čistě anglických větách falešně trefí CS.
    "je", "se", "na", "si", "ve", "ze", "nebo", "ale", "takze", "prosim",
    "dekuji", "jak", "co", "kdy", "kde", "jsem", "jsi", "jsme", "jste", "delas",
    "muzes", "ktery", "ktera", "ktere", "proc", "ano", "neni", "byl", "byla",
    "bylo", "jeste", "uz", "taky", "tak", "takhle", "vzdy", "nikdy", "treba",
    "asi", "mozna", "prosimte", "kdyz", "tam", "tady", "dobre",
}
EN_STOPWORDS = {
    # Rozšířený set — malá krátká slova ("i", "a", "to", "of"…) jsou klíčová,
    # protože EN překlady často nemají dlouhá stopwords jako "the" v každé větě,
    # ale VŽDY mají tyhle bare particles.
    "the", "is", "are", "and", "or", "but", "you", "this", "that", "with",
    "how", "what", "when", "where", "have", "has", "was", "were", "will",
    "would", "can", "could", "should", "about", "from", "they", "them",
    "there", "here", "your", "my", "me", "we", "our", "their", "been",
    "doing", "does", "didn", "don", "isn", "aren", "wasn", "weren",
    "i", "a", "an", "to", "of", "in", "on", "for", "as", "at", "by",
    "it", "not", "go", "do", "did", "if", "so", "no", "up", "out",
    "she", "he", "his", "her", "him", "all", "any", "some", "one", "two",
}


def detect_lang(text: str, prev: str = "cs") -> str:
    """Vrací 'cs' nebo 'en'. `prev` je fallback pro krátké nebo nejednoznačné vstupy."""
    t = (text or "").strip()
    if len(t) < 10 or len(t.split()) < 3:
        return prev
    if any(c in CS_DIACRITICS for c in t):
        return "cs"
    words = {w.strip(".,!?;:\"'()[]") for w in t.lower().split()}
    cs_hits = len(words & CS_STOPWORDS)
    en_hits = len(words & EN_STOPWORDS)
    if cs_hits == en_hits:
        return prev
    return "cs" if cs_hits > en_hits else "en"


def _strip_control(text: str) -> str:
    text = ANSI_ESC.sub("", text)
    text = THINK_BLOCK.sub("", text)
    return text


def normalize_cs(text: str) -> str:
    text = _strip_control(text)
    # Zkratky jako celá slova (case-sensitive pro VRAM apod.)
    for k, v in sorted(ABBREV.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", v, text)
    # Anglické termíny → fonetický CS přepis (case-insensitive, celá slova).
    # Delší fráze ("open source") před kratšími slovy.
    for k, v in sorted(EN_PHONETIC_CS.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", v, text, flags=re.IGNORECASE)
    # Symboly → slova (před čísly: "70%" → "70 procent" → "sedmdesát procent").
    for k, v in sorted(SYMBOLS_CS.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(k, v)
    # Spojené tisíce s mezerou: "1 000" / "10 000" → "1000" ať num2words trefí.
    text = re.sub(r"(?<=\d) (?=\d{3}(?:\D|$))", "", text)
    # Verze / IP (X.Y.Z, 2+ tečky) PŘED desetinnými (jinak rozbije 0.1.7).
    text = re.sub(r"\b\d+(?:\.\d+){2,}\b", _version_to_cs, text)
    # Čísla → slova přes num2words (desetinné i celé, libovolná délka).
    # Desetinné MUSÍ být před celými (jinak "3.14" rozbije na "3" "." "14").
    text = re.sub(r"\d+[.,]\d+", _num_to_cs, text)
    text = re.sub(r"\d+", _num_to_cs, text)
    # Akronymy 2+ velkými → hláskovat (po ABBREV/EN, ať NASA → "N A S A").
    def spell_caps(m):
        return " ".join(m.group(0))
    text = re.sub(r"\b[A-Z]{2,}\b", spell_caps, text)
    text = re.sub(r"[*_`#~]+", " ", text)
    text = re.sub(r"[^\w\sáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ.,;:!?\-–—()\"']+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_en(text: str) -> str:
    """Minimální normalizace pro anglický TTS. Nepoužívá CZ ABBREV (GPU → "gépéúčko"
    je špatně) ani spell_caps (v EN se „NASA" čte jako slovo, ne hláskování).
    Ponechává apostrofy ("don't", "it's") a běžnou interpunkci.
    Čísla expanduje num2words (lang=en) - Chatterbox je sám nerozvíjí."""
    text = _strip_control(text)
    text = re.sub(r"(?<=\d) (?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"\b\d+(?:\.\d+){2,}\b", _version_to_en, text)
    text = re.sub(r"\d+[.,]\d+", _int_to_en, text)
    text = re.sub(r"\d+", _int_to_en, text)
    text = re.sub(r"[*_`#~]+", " ", text)
    text = re.sub(r"[^\w\s.,;:!?\-–—()\"']+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize(text: str, lang: str = "cs") -> str:
    """Zpětně kompatibilní alias: lang='cs' volá normalize_cs, 'en' volá normalize_en."""
    return normalize_en(text) if lang == "en" else normalize_cs(text)

def chunk(text: str) -> list[str]:
    # Rozdělit po větách (., !, ?) a pokud moc dlouhé, po čárkách
    sents = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(s) <= MAX_CHARS:
            out.append(s)
            continue
        # dál rozdělit po čárkách
        parts = re.split(r",\s*", s)
        buf = ""
        for p in parts:
            if not buf:
                buf = p
            elif len(buf) + len(p) + 2 <= MAX_CHARS:
                buf = buf + ", " + p
            else:
                out.append(buf)
                buf = p
        if buf:
            out.append(buf)
    return _merge_short(out)


def _merge_short(chunks: list[str]) -> list[str]:
    """Slij příliš krátké chunky (< MIN_CHARS) se sousedem. Chatterbox na
    krátkých segmentech (jednotlivé slovo/číslo) halucinuje šum nebo jiné slovo
    (GitHub #97). Sloučení dá modelu víc kontextu. Zachová pořadí; pokud by
    sloučení překročilo MAX_CHARS, neslévá (radši krátký než hallucinující dlouhý)."""
    if not chunks:
        return chunks
    merged: list[str] = []
    for c in chunks:
        if merged and (len(c) < MIN_CHARS or len(merged[-1]) < MIN_CHARS):
            joined = merged[-1] + " " + c
            if len(joined) <= MAX_CHARS:
                merged[-1] = joined
                continue
        merged.append(c)
    return merged

def main():
    if len(sys.argv) < 3:
        print("usage: tts_cs.py <text> <out.wav> [device] [ref.wav] [--fast]", file=sys.stderr)
        sys.exit(1)
    fast = "--fast" in sys.argv
    pos = [a for a in sys.argv if a != "--fast"]
    text_raw, out = pos[1], pos[2]
    device = pos[3] if len(pos) > 3 else "cuda"
    ref = pos[4] if len(pos) > 4 and pos[4] else (str(REF) if REF.exists() else None)

    text = normalize(text_raw)
    chunks = chunk(text)
    if not chunks:
        print("empty text after normalize", file=sys.stderr)
        sys.exit(1)
    print(f"chunks: {len(chunks)}")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] ({len(c)}) {c}")

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    model.t3.load_state_dict(load_safetensors(str(CKPT), device="cpu"))
    model.t3.to(device).eval()

    kwargs = {"language_id": "pl"}
    if ref:
        kwargs["audio_prompt_path"] = ref
    if fast:
        kwargs["cfg_weight"] = 0.0  # batch=1 cesta přes monkey-patch, ~2× rychleji
        kwargs["exaggeration"] = 0.3
        print("fast mode: cfg_weight=0 (no-CFG batch=1 path), exaggeration=0.3")

    sr = model.sr
    pause = np.zeros(int(sr * PAUSE_MS / 1000), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for c in chunks:
        wav = model.generate(c, **kwargs)
        a = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        pieces.append(a)
        pieces.append(pause)
    if pieces:
        pieces.pop()  # odstraň poslední pause
    audio = np.concatenate(pieces)
    sf.write(out, audio, sr)
    print(f"saved {out} ({len(audio) / sr:.2f}s @ {sr}Hz, {len(chunks)} chunks)")

if __name__ == "__main__":
    main()
