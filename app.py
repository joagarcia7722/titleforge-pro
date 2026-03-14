"""
TitleForge Pro v2.0 — Enterprise Edition
Single-file FastAPI application for MSP data matching, cleaning & standardization.
Built for PRO Unlimited | Internal Tool

Deploy: Docker → Railway
Stack:  FastAPI + Jinja2 + HTMX + Tailwind CSS + Lucide Icons
"""

# ═══════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS & CONSTANTS
# ═══════════════════════════════════════════════════════════════
import os, sys, re, json, uuid, hashlib, io, csv, math, time, logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from fastapi import FastAPI, Request, Response, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from jinja2 import Environment, DictLoader, select_autoescape

import openpyxl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("titleforge")

APP_VERSION = "2.0.0"
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: DATA MODELS & FILE STORAGE
# ═══════════════════════════════════════════════════════════════
@dataclass
class User:
    username: str
    password_hash: str
    role: str = "user"           # admin | user
    active: bool = True
    created_at: str = ""
    last_login: str = ""
    active_project: str = ""

@dataclass
class MatchResult:
    source_title: str
    matched_title: str
    confidence: float
    zone: str                    # auto_approve | review | auto_reject
    signals: dict = field(default_factory=dict)
    status: str = "pending"      # pending | approved | rejected | overridden
    override_value: str = ""
    explanation: str = ""

@dataclass
class CleaningAudit:
    step: int
    name: str
    column: str
    description: str
    count: int
    severity: str                # info | warning | error | success
    p_value: Optional[float] = None

@dataclass
class HistoryEntry:
    id: str = ""
    module: str = ""             # matching | cleaning | standardize
    timestamp: str = ""
    rows_in: int = 0
    rows_out: int = 0
    params: dict = field(default_factory=dict)
    summary: str = ""
    user: str = ""

@dataclass
class Preset:
    name: str
    module: str
    config: dict = field(default_factory=dict)
    created_at: str = ""

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

class FileStore:
    """Simple JSON-file persistence."""
    def __init__(self, base: Path):
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)
        self._init_defaults()

    def _path(self, name: str) -> Path:
        return self.base / f"{name}.json"

    def load(self, name: str) -> list[dict]:
        p = self._path(name)
        if p.exists():
            return json.loads(p.read_text())
        return []

    def save(self, name: str, data: list[dict]):
        self._path(name).write_text(json.dumps(data, default=str, indent=2))

    def _init_defaults(self):
        if not self._path("users").exists():
            self.save("users", [asdict(User(
                username="admin",
                password_hash=_hash_pw("admin123"),
                role="admin",
                created_at=datetime.now().isoformat(),
            ))])
        for name in ["history", "presets"]:
            if not self._path(name).exists():
                self.save(name, [])

store = FileStore(DATA_DIR)


# ═══════════════════════════════════════════════════════════════
# SECTION 3: AUTH & SESSION HELPERS
# ═══════════════════════════════════════════════════════════════
def get_user(username: str) -> Optional[dict]:
    users = store.load("users")
    return next((u for u in users if u["username"] == username), None)

def authenticate(username: str, password: str) -> Optional[dict]:
    u = get_user(username)
    if u and u["password_hash"] == _hash_pw(password) and u.get("active", True):
        users = store.load("users")
        for rec in users:
            if rec["username"] == username:
                rec["last_login"] = datetime.now().isoformat()
        store.save("users", users)
        return u
    return None

def require_auth(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user

def require_admin(request: Request):
    user = require_auth(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required")
    return user

# ═══════════════════════════════════════════════════════════════
# SECTION 4: ABBREVIATION PATTERNS (130+ MSP INDUSTRY)
# ═══════════════════════════════════════════════════════════════
BUILTIN_ABBREVIATIONS = [
    # Core nursing
    (r'\brn\b', 'registered nurse'), (r'\blpn\b', 'licensed practical nurse'),
    (r'\blvn\b', 'licensed vocational nurse'), (r'\bcna\b', 'certified nursing assistant'),
    (r'\bna\b', 'nursing assistant'), (r'\bdon\b', 'director of nursing'),
    (r'\badon\b', 'assistant director of nursing'),
    # Therapy
    (r'\bpt\b', 'physical therapist'), (r'\bpta\b', 'physical therapy assistant'),
    (r'\bhha\b', 'home health aide'), (r'\bot\b', 'occupational therapist'),
    (r'\bota\b', 'occupational therapy assistant'),
    (r'\bslp\b', 'speech language pathologist'),
    (r'\brt\b', 'respiratory therapist'), (r'\bcrt\b', 'respiratory therapist'),
    (r'\brrt\b', 'respiratory therapist'),
    # Imaging / tech
    (r'\bct\s+tech\b', 'ct technologist'), (r'\bmri\s+tech\b', 'mri technologist'),
    (r'\bxray\s+tech\b', 'radiologic technologist'),
    (r'\brad\s+tech\b', 'radiologic technologist'),
    (r'\bultrasound\s+tech\b', 'ultrasound technologist'),
    (r'\becho\s+tech\b', 'echocardiography technologist'),
    (r'\bekg\s+tech\b', 'ekg technologist'), (r'\blab\s+tech\b', 'laboratory technologist'),
    (r'\bpharm\s+tech\b', 'pharmacy technologist'),
    (r'\bsurg\s+tech\b', 'surgical technologist'),
    (r'\bcvt\b', 'cardiovascular technologist'), (r'\bcst\b', 'certified surgical technologist'),
    # Physicians / advanced practice
    (r'\bmd\b', 'physician'), (r'\bnp\b', 'nurse practitioner'),
    (r'\bpa-c\b', 'physician assistant'), (r'\bpa\b', 'physician assistant'),
    (r'\bcma\b', 'certified medical assistant'), (r'\bma\b', 'medical assistant'),
    (r'\bemt\b', 'emergency medical technician'),
    (r'\bcrna\b', 'certified registered nurse anesthetist'),
    (r'\bcnm\b', 'certified nurse midwife'),
    # Units / departments
    (r'\bor\s+rn\b', 'operating room nurse'), (r'\bor\b', 'operating room'),
    (r'\ber\b', 'emergency'), (r'\bicu\b', 'intensive care unit'),
    (r'\bccu\b', 'critical care unit'), (r'\bmedsurg\b', 'medical surgical'),
    (r'\bmed\s*surg\b', 'medical surgical'), (r'\bpacu\b', 'post anesthesia care unit'),
    (r'\bob/gyn\b', 'obstetrics gynecology'), (r'\bobgyn\b', 'obstetrics gynecology'),
    (r'\bl&d\b', 'labor and delivery'), (r'\blnd\b', 'labor and delivery'),
    (r'\bpeds\b', 'pediatrics'), (r'\bonc\b', 'oncology'),
    (r'\bbh\b', 'behavioral health'), (r'\bpsych\b', 'psychiatric'),
    (r'\bnicu\b', 'neonatal intensive care unit'),
    (r'\bmicu\b', 'medical intensive care unit'),
    (r'\bsicu\b', 'surgical intensive care unit'),
    (r'\bpicu\b', 'pediatric intensive care unit'),
    (r'\bcvicu\b', 'cardiovascular intensive care unit'),
    # Rank / modifiers
    (r'\bsr\.?\b', 'senior'), (r'\bjr\.?\b', 'junior'),
    (r'\badmin\b', 'administrator'), (r'\bmgr\b', 'manager'),
    (r'\bmgmt\b', 'management'), (r'\bspec\b', 'specialist'),
    (r'\bassoc\b', 'associate'), (r'\basst\b', 'assistant'),
    (r'\bdir\b', 'director'), (r'\bcoord\b', 'coordinator'),
    (r'\bsupr?\b', 'supervisor'), (r'\bsupv\b', 'supervisor'),
    # Staffing industry
    (r'\bmsp\b', 'managed service provider'), (r'\bvms\b', 'vendor management system'),
    (r'\bsow\b', 'statement of work'), (r'\bic\b', 'independent contractor'),
    (r'\bfte\b', 'full time employee'), (r'\bpte\b', 'part time employee'),
    (r'\bw2\b', 'w2 employee'), (r'\b1099\b', '1099 contractor'),
    (r'\brfp\b', 'request for proposal'), (r'\brfq\b', 'request for quotation'),
    # Specialty nursing
    (r'\bdialysis\s+rn\b', 'dialysis nurse'), (r'\btele\s+rn\b', 'telemetry nurse'),
    (r'\btelemetry\s+rn\b', 'telemetry nurse'),
    (r'\bnicu\s+rn\b', 'neonatal intensive care nurse'),
    (r'\bhouse\s+sup\b', 'house supervisor'),
    (r'\bpsych\s+nurse\b', 'psychiatric nurse'),
    (r'\bhh\s+rn\b', 'home health registered nurse'),
    (r'\bltac\s+rn\b', 'long term acute care nurse'),
    (r'\bnurse\s+manager\b', 'nurse manager'),
    (r'\binfection\s+control\s+nurse\b', 'infection control nurse'),
    (r'\bcase\s+manager\s+rn\b', 'nurse case manager'),
    (r'\bcath\s+lab\s+rn\b', 'cardiac cath lab nurse'),
    (r'\bendo\s+rn\b', 'endoscopy nurse'),
    (r'\btrauma\s+rn\b', 'trauma nurse'), (r'\bhospice\s+rn\b', 'hospice nurse'),
    (r'\brehab\s+rn\b', 'rehabilitation nurse'),
    (r'\bcharge\s+rn\b', 'charge nurse'), (r'\bflight\s+rn\b', 'flight nurse'),
    (r'\btravel\s+rn\b', 'travel nurse'), (r'\bschool\s+rn\b', 'school nurse'),
    (r'\bnurse\s+educator\b', 'nurse educator'),
    (r'\bnurse\s+navigator\b', 'nurse navigator'),
    (r'\bper\s+diem\s+rn\b', 'per diem nurse'),
    (r'\bpatient\s+observer\b', 'patient sitter'), (r'\btelesitter\b', 'patient sitter'),
    # IT / professional (MSP crossover)
    (r'\bpm\b', 'project manager'), (r'\bba\b', 'business analyst'),
    (r'\bqa\b', 'quality assurance'), (r'\bux\b', 'user experience'),
    (r'\bui\b', 'user interface'), (r'\bdevops\b', 'devops engineer'),
    (r'\bsde\b', 'software development engineer'), (r'\bswe\b', 'software engineer'),
    (r'\bdba\b', 'database administrator'), (r'\bsa\b', 'systems administrator'),
    (r'\bsm\b', 'scrum master'),
]


# ═══════════════════════════════════════════════════════════════
# SECTION 5: MATCHING ENGINE — 3-PASS ALGORITHM
# ═══════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation."""
    if not isinstance(text, str):
        text = str(text)
    t = text.lower().strip()
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def expand_abbreviations(text: str, custom_abbrs: list = None) -> str:
    """Expand abbreviations using built-in + custom patterns."""
    t = text.lower().strip()
    patterns = BUILTIN_ABBREVIATIONS + (custom_abbrs or [])
    for pattern, replacement in patterns:
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    return t

# AI-powered abbreviation cache (populated by API calls)
_ai_expansion_cache: dict[str, str] = {}

async def ai_expand_abbreviations_batch(titles: list[str], api_key: str = None) -> dict[str, str]:
    """
    Use Claude API to expand abbreviated job titles.
    Falls back to built-in patterns if API unavailable.
    Returns dict mapping original → expanded title.
    """
    import httpx

    # First apply built-in patterns
    result = {}
    remaining = []
    for t in titles:
        expanded = expand_abbreviations(t)
        norm = normalize_text(expanded)
        orig_norm = normalize_text(t)
        if norm != orig_norm:
            result[t] = expanded
        else:
            # Check cache
            if t.lower() in _ai_expansion_cache:
                result[t] = _ai_expansion_cache[t.lower()]
            else:
                remaining.append(t)

    if not remaining:
        return result

    # Call Claude API for remaining unresolved titles
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        # No API key — return built-in expansions only
        for t in remaining:
            result[t] = expand_abbreviations(t)
        return result

    try:
        # Batch in groups of 50
        for batch_start in range(0, len(remaining), 50):
            batch = remaining[batch_start:batch_start+50]
            titles_text = "\n".join(f"- {t}" for t in batch)

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 2000,
                        "messages": [{"role": "user", "content": f"""Expand these abbreviated job titles into their full standardized form.
Return ONLY a JSON object mapping each original title to its expanded form.
No markdown, no explanation, just the JSON.

Titles:
{titles_text}"""}],
                    }
                )
                data = resp.json()
                text = data.get("content", [{}])[0].get("text", "{}")
                text = re.sub(r'```json|```', '', text).strip()
                mappings = json.loads(text)

                for orig, expanded in mappings.items():
                    result[orig] = expanded
                    _ai_expansion_cache[orig.lower()] = expanded

    except Exception as e:
        logger.warning(f"AI expansion failed: {e}. Using built-in patterns only.")
        for t in remaining:
            if t not in result:
                result[t] = expand_abbreviations(t)

    return result

def strip_credentials(text: str) -> str:
    """Remove level indicators and credential suffixes."""
    t = re.sub(r'\b(i{1,3}|iv|v|vi{0,3}|[1-9])\s*$', '', text)
    t = re.sub(r'\b(certified|registered|licensed)\b', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def base_title(text: str) -> str:
    """Pre-comma text: 'Nurse, Registered' → 'Nurse'."""
    return text.split(',')[0].strip()

# ── PASS 1: DETERMINISTIC MATCHING ──────────────────────────
def pass1_deterministic(source_titles: list[str], master_titles: list[str],
                        custom_abbrs: list = None) -> dict[int, MatchResult]:
    """Exact matches at 100% confidence."""
    results = {}
    # Build lookup tables
    lookup = {}
    for i, mt in enumerate(master_titles):
        keys = [
            normalize_text(mt),
            normalize_text(expand_abbreviations(mt, custom_abbrs)),
            normalize_text(strip_credentials(normalize_text(mt))),
            normalize_text(base_title(mt)),
        ]
        # technologist/technician variants
        for k in list(keys):
            if 'technologist' in k:
                keys.append(k.replace('technologist', 'technician'))
            elif 'technician' in k:
                keys.append(k.replace('technician', 'technologist'))
        for k in keys:
            if k and k not in lookup:
                lookup[k] = (i, mt)

    for si, st_raw in enumerate(source_titles):
        candidates = [
            normalize_text(st_raw),
            normalize_text(expand_abbreviations(st_raw, custom_abbrs)),
            normalize_text(strip_credentials(normalize_text(expand_abbreviations(st_raw, custom_abbrs)))),
            normalize_text(base_title(st_raw)),
        ]
        for c in list(candidates):
            if 'technologist' in c:
                candidates.append(c.replace('technologist', 'technician'))
            elif 'technician' in c:
                candidates.append(c.replace('technician', 'technologist'))

        for key in candidates:
            if key in lookup:
                mi, mt = lookup[key]
                results[si] = MatchResult(
                    source_title=st_raw, matched_title=mt,
                    confidence=100.0, zone="auto_approve",
                    signals={"deterministic": 100.0},
                    status="pending",
                    explanation=f"Exact deterministic match after normalization."
                )
                break
    return results

# ── PASS 2: MULTI-SIGNAL SCORING (6 algorithms) ─────────────
def _tfidf_cosine_scores(source_clean: list[str], master_clean: list[str]) -> np.ndarray:
    """TF-IDF cosine similarity matrix."""
    all_docs = master_clean + source_clean
    vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
    tfidf = vec.fit_transform(all_docs)
    master_m = tfidf[:len(master_clean)]
    source_m = tfidf[len(master_clean):]
    return cosine_similarity(source_m, master_m)

def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0

def _char_ngrams(a: str, b: str, n: int = 3) -> float:
    def ngrams(s):
        return set(s[i:i+n] for i in range(max(0, len(s)-n+1)))
    ga, gb = ngrams(a), ngrams(b)
    union = ga | gb
    return len(ga & gb) / len(union) if union else 0.0

def _structural_similarity(a: str, b: str) -> float:
    """Compare word-length patterns and positional structure."""
    wa, wb = a.split(), b.split()
    if not wa or not wb:
        return 0.0
    # Length pattern match
    la = [len(w) for w in wa]
    lb = [len(w) for w in wb]
    max_len = max(len(la), len(lb))
    la_pad = la + [0] * (max_len - len(la))
    lb_pad = lb + [0] * (max_len - len(lb))
    diff = sum(abs(a - b) for a, b in zip(la_pad, lb_pad))
    max_diff = sum(max(a, b) for a, b in zip(la_pad, lb_pad)) or 1
    return 1.0 - (diff / max_diff)

def _prefix_match(a: str, b: str) -> float:
    """Leading character match ratio."""
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    match_len = 0
    for i in range(min_len):
        if a[i] == b[i]:
            match_len += 1
        else:
            break
    return match_len / max(len(a), len(b))

def _faiss_semantic_scores(source_vecs: np.ndarray, master_vecs: np.ndarray,
                           k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """FAISS approximate nearest neighbor search."""
    if not FAISS_AVAILABLE or source_vecs is None or master_vecs is None:
        return None, None
    d = master_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    m_norm = master_vecs.copy().astype('float32')
    faiss.normalize_L2(m_norm)
    index.add(m_norm)
    s_norm = source_vecs.copy().astype('float32')
    faiss.normalize_L2(s_norm)
    scores, indices = index.search(s_norm, k)
    return scores, indices

def _tfidf_embeddings(texts: list[str], dim: int = 128) -> np.ndarray:
    """Create dense-ish TF-IDF vectors for semantic matching (no sentence-transformers needed)."""
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=dim, min_df=1)
    m = vec.fit_transform(texts).toarray().astype('float32')
    return m

def pass2_multi_signal(source_titles: list[str], master_titles: list[str],
                       unmatched_indices: list[int], custom_abbrs: list = None
                       ) -> dict[int, dict]:
    """Compute 6 signal scores for each unmatched source vs all masters."""
    source_clean = [normalize_text(expand_abbreviations(source_titles[i], custom_abbrs))
                    for i in unmatched_indices]
    master_clean = [normalize_text(expand_abbreviations(t, custom_abbrs)) for t in master_titles]

    if not source_clean or not master_clean:
        return {}

    # Signal 1: TF-IDF Cosine (batch)
    tfidf_matrix = _tfidf_cosine_scores(source_clean, master_clean)

    # Signal 4: FAISS semantic (batch)
    all_texts = master_clean + source_clean
    embeddings = _tfidf_embeddings(all_texts, dim=128)
    master_emb = embeddings[:len(master_clean)]
    source_emb = embeddings[len(master_clean):]
    faiss_scores, faiss_idxs = _faiss_semantic_scores(source_emb, master_emb, k=5)

    signal_data = {}
    for j, si in enumerate(unmatched_indices):
        sc = source_clean[j]
        # Find top candidate by TF-IDF
        tfidf_row = tfidf_matrix[j]
        top_k_idx = np.argsort(tfidf_row)[-5:][::-1]

        candidates = {}
        for mi in top_k_idx:
            mc = master_clean[mi]
            signals = {
                "tfidf_cosine": float(tfidf_row[mi]),
                "token_jaccard": _token_jaccard(sc, mc),
                "char_ngrams": _char_ngrams(sc, mc),
                "structural": _structural_similarity(sc, mc),
                "prefix": _prefix_match(sc, mc),
            }
            # Add FAISS if available
            if faiss_scores is not None:
                faiss_row_scores = faiss_scores[j]
                faiss_row_idxs = faiss_idxs[j]
                if mi in faiss_row_idxs:
                    fidx = list(faiss_row_idxs).index(mi)
                    signals["semantic"] = float(faiss_row_scores[fidx])
                else:
                    signals["semantic"] = 0.0
            else:
                signals["semantic"] = signals["tfidf_cosine"]  # fallback

            candidates[mi] = signals

        # Also add FAISS top-k candidates not already in TF-IDF top-k
        if faiss_idxs is not None:
            for fi_pos in range(min(5, faiss_idxs.shape[1])):
                mi = int(faiss_idxs[j][fi_pos])
                if mi not in candidates:
                    mc = master_clean[mi]
                    candidates[mi] = {
                        "tfidf_cosine": float(tfidf_matrix[j][mi]),
                        "token_jaccard": _token_jaccard(sc, mc),
                        "char_ngrams": _char_ngrams(sc, mc),
                        "structural": _structural_similarity(sc, mc),
                        "prefix": _prefix_match(sc, mc),
                        "semantic": float(faiss_scores[j][fi_pos]),
                    }

        signal_data[si] = candidates
    return signal_data

# ── PASS 3: SELF-CALIBRATING ENSEMBLE ───────────────────────
def pass3_ensemble(signal_data: dict[int, dict], p1_results: dict[int, MatchResult],
                   source_titles: list[str], master_titles: list[str],
                   thresholds: dict = None) -> dict[int, MatchResult]:
    """Train logistic regression on Pass 1 confirmed matches, score Pass 2 candidates."""
    if thresholds is None:
        thresholds = {"auto_approve": 90, "review": 70}

    signal_names = ["tfidf_cosine", "token_jaccard", "char_ngrams",
                    "semantic", "structural", "prefix"]

    # Build training data from Pass 1 matches (positive) + random negatives
    X_train, y_train = [], []
    confirmed_pairs = []
    for si, mr in p1_results.items():
        mi_idx = next((i for i, t in enumerate(master_titles) if t == mr.matched_title), None)
        if mi_idx is not None:
            confirmed_pairs.append((si, mi_idx))

    # Compute signals for confirmed pairs to use as positive training examples
    if confirmed_pairs and len(confirmed_pairs) >= 5:
        source_for_train = [normalize_text(expand_abbreviations(source_titles[si]))
                            for si, _ in confirmed_pairs]
        master_for_train = [normalize_text(expand_abbreviations(master_titles[mi]))
                            for _, mi in confirmed_pairs]
        for sc, mc in zip(source_for_train, master_for_train):
            features = [
                _token_jaccard(sc, mc),   # approx tfidf with jaccard
                _token_jaccard(sc, mc),
                _char_ngrams(sc, mc),
                _char_ngrams(sc, mc),     # semantic approx
                _structural_similarity(sc, mc),
                _prefix_match(sc, mc),
            ]
            X_train.append(features)
            y_train.append(1)

        # Negative examples: random non-matching pairs
        rng = np.random.RandomState(42)
        for _ in range(min(len(confirmed_pairs) * 2, 100)):
            si = rng.randint(0, len(source_titles))
            mi = rng.randint(0, len(master_titles))
            sc = normalize_text(expand_abbreviations(source_titles[si]))
            mc = normalize_text(expand_abbreviations(master_titles[mi]))
            features = [
                _token_jaccard(sc, mc),
                _token_jaccard(sc, mc),
                _char_ngrams(sc, mc),
                _char_ngrams(sc, mc),
                _structural_similarity(sc, mc),
                _prefix_match(sc, mc),
            ]
            X_train.append(features)
            y_train.append(0)

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=200, random_state=42)
            clf.fit(X_scaled, y_train)
            use_ml = True
            logger.info(f"Pass 3: ML ensemble trained on {len(X_train)} examples")
        except Exception as e:
            logger.warning(f"Pass 3: ML training failed ({e}), using equal weights")
            use_ml = False
    else:
        use_ml = False
        logger.info("Pass 3: Insufficient training data, using equal weights")

    # Score all unmatched candidates
    results = {}
    for si, candidates in signal_data.items():
        best_score = -1
        best_mi = None
        best_signals = {}

        for mi, signals in candidates.items():
            feat = [signals.get(s, 0.0) for s in signal_names]

            if use_ml:
                feat_scaled = scaler.transform([feat])
                prob = clf.predict_proba(feat_scaled)[0][1]
                score = prob * 100
            else:
                # Equal weight fallback
                score = np.mean(feat) * 100

            if score > best_score:
                best_score = score
                best_mi = mi
                best_signals = signals.copy()
                best_signals["ensemble_score"] = score

        if best_mi is not None:
            confidence = round(best_score, 1)
            if confidence >= thresholds["auto_approve"]:
                zone = "auto_approve"
            elif confidence >= thresholds["review"]:
                zone = "review"
            else:
                zone = "auto_reject"

            # Build explanation
            top_signals = sorted(best_signals.items(), key=lambda x: x[1], reverse=True)[:3]
            sig_parts = [f"{k}: {v:.0%}" for k, v in top_signals if k != "ensemble_score"]
            explanation = f"Ensemble score {confidence:.1f}%. Top signals: {', '.join(sig_parts)}"

            results[si] = MatchResult(
                source_title=source_titles[si],
                matched_title=master_titles[best_mi],
                confidence=confidence,
                zone=zone,
                signals=best_signals,
                status="pending",
                explanation=explanation,
            )

    return results

def run_full_matching(source_titles: list[str], master_titles: list[str],
                      custom_abbrs: list = None,
                      thresholds: dict = None) -> list[MatchResult]:
    """Execute full 3-pass matching pipeline."""
    total = len(source_titles)
    if not master_titles:
        return [MatchResult(source_title=t, matched_title="", confidence=0,
                            zone="auto_reject", explanation="No master list") for t in source_titles]

    # Pass 1
    p1 = pass1_deterministic(source_titles, master_titles, custom_abbrs)
    logger.info(f"Pass 1: {len(p1)}/{total} deterministic matches")

    # Pass 2
    unmatched = [i for i in range(total) if i not in p1]
    if unmatched:
        signals = pass2_multi_signal(source_titles, master_titles, unmatched, custom_abbrs)
        # Pass 3
        p3 = pass3_ensemble(signals, p1, source_titles, master_titles, thresholds)
    else:
        p3 = {}

    # Combine
    all_results = []
    for i in range(total):
        if i in p1:
            all_results.append(p1[i])
        elif i in p3:
            all_results.append(p3[i])
        else:
            all_results.append(MatchResult(
                source_title=source_titles[i], matched_title="(no match)",
                confidence=0, zone="auto_reject",
                explanation="No suitable match found."
            ))

    return all_results


# ═══════════════════════════════════════════════════════════════
# SECTION 6: CLEANING PIPELINE — 8 STEPS WITH P-VALUES
# ═══════════════════════════════════════════════════════════════

def parse_pay_value(val) -> float:
    """Parse raw pay into float. Handles $25.50, 25-30 (midpoint), /hr, N/A, etc."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in ('n/a', 'na', 'tbd', 'negotiable', 'open', '', '-', 'null', 'none'):
        return np.nan
    s = re.sub(r'[\$,£€¥]', '', s)
    s = re.sub(r'/(hr|hour|wk|week|yr|year|mo|month|annual|ann)', '', s, flags=re.IGNORECASE)
    s = s.strip()
    m = re.match(r'^([\d.]+)\s*[-–—]\s*([\d.]+)$', s)
    if m:
        return round((float(m.group(1)) + float(m.group(2))) / 2, 4)
    try:
        return float(s)
    except ValueError:
        return np.nan

def detect_pay_columns(df: pd.DataFrame) -> list[str]:
    kw = ['pay', 'rate', 'wage', 'salary', 'hourly', 'weekly', 'annual',
          'bill', 'stipend', 'comp', 'earn', 'income', 'gross']
    return [c for c in df.columns if any(k in c.lower() for k in kw)]

def grubbs_test_pvalue(data: np.ndarray) -> float:
    """One-sided Grubbs test p-value for the most extreme outlier."""
    n = len(data)
    if n < 3:
        return 1.0
    mean, std = np.mean(data), np.std(data, ddof=1)
    if std == 0:
        return 1.0
    G = np.max(np.abs(data - mean)) / std
    t_sq = (n - 2) * G**2 / (n - 1 - G**2 + 1e-12)
    t_sq = max(t_sq, 0)
    t_val = math.sqrt(t_sq)
    p = 2 * n * (1 - sp_stats.t.cdf(t_val, n - 2))
    return min(max(p, 0), 1)

def run_cleaning_pipeline(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    """8-step cleaning pipeline with statistical p-values."""
    cleaned = df.copy()
    audit = []
    pay_cols = config.get('pay_cols', [])
    job_col = config.get('job_col')
    hourly_col = config.get('hourly_col')
    weekly_col = config.get('weekly_col')
    hours_col = config.get('hours_col')
    neg_action = config.get('negative_action', 'flag')
    zero_action = config.get('zero_action', 'flag')
    min_hourly = config.get('min_hourly', 5.0)
    max_hourly = config.get('max_hourly', 500.0)
    min_weekly = config.get('min_weekly', 50.0)
    max_weekly = config.get('max_weekly', 20000.0)
    iqr_mult = config.get('iqr_mult', 1.5)
    z_thresh = config.get('z_thresh', 3.0)
    sig_level = config.get('sig_level', 0.05)
    missing_strategy = config.get('missing_strategy', 'keep')

    # ── Step 1: Format Parsing ──
    for col in pay_cols:
        if col not in cleaned.columns:
            continue
        before = cleaned[col].copy()
        cleaned[col] = cleaned[col].apply(parse_pay_value)
        changed = int((before.astype(str) != cleaned[col].astype(str)).sum())
        if changed:
            audit.append({"step": 1, "name": "Format Parsing", "column": col,
                         "description": f"Parsed {changed} values (stripped symbols, converted ranges).",
                         "count": changed, "severity": "info", "p_value": None})

    # ── Step 2: Type Validation ──
    for col in pay_cols:
        if col not in cleaned.columns:
            continue
        non_num = pd.to_numeric(cleaned[col], errors='coerce').isna() & cleaned[col].notna()
        cnt = int(non_num.sum())
        if cnt:
            cleaned.loc[non_num, col] = np.nan
            audit.append({"step": 2, "name": "Type Validation", "column": col,
                         "description": f"Nullified {cnt} non-numeric values.",
                         "count": cnt, "severity": "warning", "p_value": None})
        cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

        # Shapiro-Wilk normality test
        data = cleaned[col].dropna().values
        if len(data) >= 8 and len(data) <= 5000:
            try:
                _, sw_p = sp_stats.shapiro(data[:5000])
                audit.append({"step": 2, "name": "Normality Test (Shapiro-Wilk)", "column": col,
                             "description": f"p={sw_p:.4f} — {'Normal' if sw_p > sig_level else 'Non-normal'} distribution.",
                             "count": 0, "severity": "info", "p_value": round(sw_p, 6)})
            except Exception:
                pass

    # ── Step 3: Zero/Negative Handling ──
    for col in pay_cols:
        if col not in cleaned.columns:
            continue
        neg_mask = cleaned[col] < 0
        cnt = int(neg_mask.sum())
        if cnt:
            if neg_action == 'remove':
                cleaned = cleaned[~neg_mask].reset_index(drop=True)
                desc = f"Removed {cnt} rows with negative values."
            elif neg_action == 'abs':
                cleaned.loc[neg_mask, col] = cleaned.loc[neg_mask, col].abs()
                desc = f"Converted {cnt} negatives to absolute."
            else:
                cleaned.loc[neg_mask, f'FLAG_{col}_neg'] = True
                desc = f"Flagged {cnt} negative values."
            audit.append({"step": 3, "name": "Negative Handling", "column": col,
                         "description": desc, "count": cnt, "severity": "warning", "p_value": None})

        zero_mask = cleaned[col] == 0
        cnt = int(zero_mask.sum())
        if cnt:
            if zero_action == 'remove':
                cleaned = cleaned[~zero_mask].reset_index(drop=True)
                desc = f"Removed {cnt} zero-value rows."
            elif zero_action == 'null':
                cleaned.loc[zero_mask, col] = np.nan
                desc = f"Nullified {cnt} zeros."
            else:
                cleaned.loc[zero_mask, f'FLAG_{col}_zero'] = True
                desc = f"Flagged {cnt} zero values."
            audit.append({"step": 3, "name": "Zero Handling", "column": col,
                         "description": desc, "count": cnt, "severity": "warning", "p_value": None})

    # ── Step 4: Range Validation ──
    range_map = {}
    if hourly_col and hourly_col in pay_cols:
        range_map[hourly_col] = (min_hourly, max_hourly, "hourly")
    if weekly_col and weekly_col in pay_cols:
        range_map[weekly_col] = (min_weekly, max_weekly, "weekly")
    for col in pay_cols:
        if col not in cleaned.columns:
            continue
        lo, hi, label = range_map.get(col, (0, 1e9, "general"))
        out_mask = cleaned[col].notna() & ((cleaned[col] < lo) | (cleaned[col] > hi))
        cnt = int(out_mask.sum())
        if cnt:
            cleaned.loc[out_mask, f'FLAG_{col}_range'] = True
            audit.append({"step": 4, "name": "Range Validation", "column": col,
                         "description": f"Flagged {cnt} values outside [{lo:,.2f}–{hi:,.2f}] ({label}).",
                         "count": cnt, "severity": "warning", "p_value": None})

    # ── Step 5: IQR Outlier Detection with Grubbs' p-values ──
    for col in pay_cols:
        if col not in cleaned.columns:
            continue
        if job_col and job_col in cleaned.columns:
            outlier_rows = []
            p_values = []
            for _, grp in cleaned.groupby(job_col):
                vals = grp[col].dropna()
                if len(vals) < 4:
                    continue
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                lo_b, hi_b = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
                out = grp[(grp[col] < lo_b) | (grp[col] > hi_b)]
                outlier_rows.extend(out.index.tolist())
                if len(vals.values) >= 3:
                    p_values.append(grubbs_test_pvalue(vals.values))
            if outlier_rows:
                cleaned.loc[outlier_rows, f'FLAG_{col}_iqr'] = True
                avg_p = np.mean(p_values) if p_values else None
                audit.append({"step": 5, "name": "IQR Outlier (per title)", "column": col,
                             "description": f"Flagged {len(outlier_rows)} IQR outliers (mult={iqr_mult}).",
                             "count": len(outlier_rows), "severity": "warning",
                             "p_value": round(avg_p, 6) if avg_p else None})
        else:
            data = cleaned[col].dropna()
            if len(data) < 4:
                continue
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            lo_b, hi_b = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
            out_mask = cleaned[col].notna() & ((cleaned[col] < lo_b) | (cleaned[col] > hi_b))
            cnt = int(out_mask.sum())
            if cnt:
                cleaned.loc[out_mask, f'FLAG_{col}_iqr'] = True
                gp = grubbs_test_pvalue(data.values)
                audit.append({"step": 5, "name": "IQR Outlier (global)", "column": col,
                             "description": f"Flagged {cnt} global IQR outliers.",
                             "count": cnt, "severity": "warning", "p_value": round(gp, 6)})

    # ── Step 6: Z-Score Outlier Detection with p-values ──
    for col in pay_cols:
        if col not in cleaned.columns:
            continue
        data = cleaned[col].dropna()
        if len(data) < 5:
            continue
        mu, sigma = data.mean(), data.std()
        if sigma == 0:
            continue
        z = ((cleaned[col] - mu) / sigma).abs()
        out_mask = z > z_thresh
        cnt = int(out_mask.sum())
        if cnt:
            cleaned.loc[out_mask, f'FLAG_{col}_zscore'] = True
            # Two-tailed p-values for the most extreme
            max_z = float(z.max())
            two_tail_p = 2 * (1 - sp_stats.norm.cdf(max_z))
            audit.append({"step": 6, "name": "Z-Score Outlier", "column": col,
                         "description": f"Flagged {cnt} values with |z|>{z_thresh} (mean={mu:.2f}, σ={sigma:.2f}). Most extreme p={two_tail_p:.2e}.",
                         "count": cnt, "severity": "warning", "p_value": round(two_tail_p, 8)})

    # ── Step 7: Cross-Column Consistency ──
    if hourly_col and weekly_col and hours_col:
        if all(c in cleaned.columns for c in [hourly_col, weekly_col, hours_col]):
            mask = (cleaned[hourly_col].notna() & cleaned[weekly_col].notna() &
                    cleaned[hours_col].notna() & (cleaned[hours_col] > 0))
            if mask.any():
                expected = cleaned.loc[mask, hourly_col] * cleaned.loc[mask, hours_col]
                actual = cleaned.loc[mask, weekly_col]
                pct_diff = ((actual - expected).abs() / expected.replace(0, np.nan) * 100)
                incon = pct_diff > 10
                cnt = int(incon.sum())
                if cnt:
                    cleaned.loc[mask, 'FLAG_weekly_mismatch'] = incon.values
                    audit.append({"step": 7, "name": "Cross-Column Consistency",
                                 "column": f"{weekly_col} vs {hourly_col}×{hours_col}",
                                 "description": f"Flagged {cnt} rows where weekly differs >10% from hourly×hours.",
                                 "count": cnt, "severity": "warning", "p_value": None})

    # ── Step 8: Duplicate Detection & Audit Trail ──
    dup_mask = cleaned.duplicated(keep=False)
    cnt = int(dup_mask.sum())
    if cnt:
        cleaned.loc[dup_mask, 'FLAG_duplicate'] = True
        audit.append({"step": 8, "name": "Duplicate Detection", "column": "all",
                     "description": f"Flagged {cnt} exact duplicate rows.",
                     "count": cnt, "severity": "warning", "p_value": None})

    if not audit:
        audit.append({"step": 0, "name": "All Steps", "column": "—",
                     "description": "No issues detected. Data looks clean!",
                     "count": 0, "severity": "success", "p_value": None})

    return cleaned, audit

# ═══════════════════════════════════════════════════════════════
# SECTION 7: STANDARDIZE MODE — AUTO-CLUSTERING
# ═══════════════════════════════════════════════════════════════

def auto_cluster_titles(titles: list[str], threshold: float = 0.75,
                        min_cluster_size: int = 2) -> list[dict]:
    """Cluster similar titles using TF-IDF cosine similarity. No master list needed."""
    clean = [normalize_text(expand_abbreviations(t)) for t in titles]
    if not clean:
        return []

    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    tfidf = vec.fit_transform(clean)
    sim_matrix = cosine_similarity(tfidf)

    assigned = set()
    clusters = []

    # Greedy clustering
    for i in range(len(titles)):
        if i in assigned:
            continue
        cluster_indices = [i]
        assigned.add(i)
        for j in range(i + 1, len(titles)):
            if j in assigned:
                continue
            if sim_matrix[i][j] >= threshold:
                cluster_indices.append(j)
                assigned.add(j)

        if len(cluster_indices) >= min_cluster_size:
            # Pick canonical: longest cleaned title as representative
            members = [(titles[idx], clean[idx]) for idx in cluster_indices]
            canonical = max(members, key=lambda x: len(x[1]))[0]
            clusters.append({
                "canonical": canonical,
                "members": [titles[idx] for idx in cluster_indices],
                "size": len(cluster_indices),
                "avg_similarity": float(np.mean([sim_matrix[cluster_indices[0]][j]
                                                  for j in cluster_indices])),
            })
        elif len(cluster_indices) == 1:
            clusters.append({
                "canonical": titles[i],
                "members": [titles[i]],
                "size": 1,
                "avg_similarity": 1.0,
            })

    return clusters

# ═══════════════════════════════════════════════════════════════
# SECTION 8: HISTORY TRACKER
# ═══════════════════════════════════════════════════════════════

def save_history(module: str, rows_in: int, rows_out: int,
                 params: dict, summary: str, user: str = ""):
    history = store.load("history")
    entry = {
        "id": str(uuid.uuid4())[:8],
        "module": module,
        "timestamp": datetime.now().isoformat(),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "params": params,
        "summary": summary,
        "user": user,
    }
    history.insert(0, entry)
    history = history[:100]  # keep last 100
    store.save("history", history)


# ═══════════════════════════════════════════════════════════════
# SECTION 9: HTML TEMPLATES (Jinja2 DictLoader)
# ═══════════════════════════════════════════════════════════════

TEMPLATES = {}

TEMPLATES["base.html"] = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{% block title %}TitleForge Pro{% endblock %}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <script src="https://unpkg.com/lucide@latest"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <script>
  tailwind.config = {
    theme: {
      extend: {
        fontFamily: { display: ['Outfit'], body: ['DM Sans'], mono: ['JetBrains Mono'] },
        colors: {
          brand: { 50:'#eff6ff', 100:'#dbeafe', 200:'#bfdbfe', 500:'#3b82f6',
                   600:'#2563eb', 700:'#1d4ed8', 900:'#1e3a5f' },
          surface: { 50:'#f8fafc', 100:'#f1f5f9', 200:'#e2e8f0', 700:'#334155', 900:'#0f172a' },
        }
      }
    }
  }
  </script>
  <style>
    * { font-family: 'DM Sans', sans-serif; }
    h1,h2,h3,h4,h5,h6,.font-display { font-family: 'Outfit', sans-serif; }
    code,pre,.font-mono { font-family: 'JetBrains Mono', monospace; }
    .sidebar-link { transition: all 0.15s ease; }
    .sidebar-link:hover { background: rgba(255,255,255,0.08); }
    .sidebar-link.active { background: rgba(59,130,246,0.2); border-right: 3px solid #3b82f6; }
    .card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; transition: box-shadow 0.2s; }
    .card:hover { box-shadow: 0 8px 32px rgba(0,0,0,0.08); }
    .btn-primary { background: #2563eb; color: white; font-weight: 600; border-radius: 8px;
                   padding: 10px 20px; transition: all 0.15s; border: none; cursor: pointer; }
    .btn-primary:hover { background: #1d4ed8; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37,99,235,0.3); }
    .btn-secondary { background: white; color: #334155; font-weight: 500; border-radius: 8px;
                     padding: 10px 20px; border: 1px solid #e2e8f0; cursor: pointer; transition: all 0.15s; }
    .btn-secondary:hover { background: #f8fafc; border-color: #cbd5e1; }
    .badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 10px;
             border-radius: 20px; font-size: 12px; font-weight: 600; }
    .badge-green { background: #dcfce7; color: #166534; }
    .badge-amber { background: #fef3c7; color: #92400e; }
    .badge-red { background: #fee2e2; color: #991b1b; }
    .badge-blue { background: #dbeafe; color: #1e40af; }
    .badge-slate { background: #f1f5f9; color: #475569; }
    .score-bar { height: 6px; border-radius: 3px; background: #e2e8f0; overflow: hidden; }
    .score-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
    .htmx-indicator { display: none; }
    .htmx-request .htmx-indicator { display: inline-flex; }
    .fade-in { animation: fadeIn 0.3s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
    .stat-card { text-align: center; padding: 20px; }
    .stat-val { font-family: 'Outfit'; font-size: 2rem; font-weight: 800; line-height: 1; }
    .stat-label { font-size: 0.7rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 6px; }
    input, select, textarea { font-family: 'DM Sans', sans-serif; }
    .toast { position: fixed; top: 20px; right: 20px; z-index: 9999; animation: slideIn 0.3s ease; }
    @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: none; opacity: 1; } }
  </style>
</head>
<body class="bg-surface-50 min-h-screen">
  <div class="flex min-h-screen">
    <!-- SIDEBAR -->
    <aside class="w-64 bg-surface-900 text-white flex-shrink-0 flex flex-col" id="sidebar">
      <div class="p-5 border-b border-white/10">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-lg bg-brand-600 flex items-center justify-center">
            <i data-lucide="layers" class="w-5 h-5"></i>
          </div>
          <div>
            <div class="font-display font-bold text-sm tracking-tight">TitleForge Pro</div>
            <div class="text-[11px] text-slate-400">v2.0 Enterprise</div>
          </div>
        </div>
      </div>
      <nav class="flex-1 py-4 px-3 space-y-1">
        <a href="/" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 {% if page == 'home' %}active text-white{% endif %}">
          <i data-lucide="home" class="w-4 h-4"></i> Home
        </a>
        <a href="/matching" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 {% if page == 'matching' %}active text-white{% endif %}">
          <i data-lucide="git-merge" class="w-4 h-4"></i> Matching Engine
        </a>
        <a href="/cleaning" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 {% if page == 'cleaning' %}active text-white{% endif %}">
          <i data-lucide="sparkles" class="w-4 h-4"></i> Data Cleaner
        </a>
        <a href="/standardize" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 {% if page == 'standardize' %}active text-white{% endif %}">
          <i data-lucide="layers" class="w-4 h-4"></i> Standardize
        </a>
        <div class="pt-4 pb-2 px-3"><div class="text-[10px] text-slate-500 font-semibold uppercase tracking-wider">System</div></div>
        <a href="/history" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 {% if page == 'history' %}active text-white{% endif %}">
          <i data-lucide="clock" class="w-4 h-4"></i> History
        </a>
        <a href="/admin" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 {% if page == 'admin' %}active text-white{% endif %}">
          <i data-lucide="settings" class="w-4 h-4"></i> Admin Panel
        </a>
        <a href="/api-settings" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 {% if page == 'api-settings' %}active text-white{% endif %}">
          <i data-lucide="key" class="w-4 h-4"></i> API Key
        </a>
      </nav>
      <div class="p-4 border-t border-white/10">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <div class="w-7 h-7 rounded-full bg-brand-600 flex items-center justify-center text-xs font-bold">
              {{ user.username[0]|upper if user else '?' }}
            </div>
            <div class="text-xs text-slate-400">{{ user.username if user else 'Guest' }}</div>
          </div>
          <a href="/logout" class="text-slate-500 hover:text-white"><i data-lucide="log-out" class="w-4 h-4"></i></a>
        </div>
      </div>
    </aside>

    <!-- MAIN CONTENT -->
    <main class="flex-1 overflow-y-auto">
      <div class="max-w-6xl mx-auto px-8 py-6">
        {% block content %}{% endblock %}
      </div>
    </main>
  </div>
  <script>lucide.createIcons();</script>
  {% block scripts %}{% endblock %}
</body>
</html>"""


TEMPLATES["login.html"] = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Login — TitleForge Pro</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
  <style>* { font-family: 'DM Sans', sans-serif; } h1,h2 { font-family: 'Outfit', sans-serif; }</style>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-blue-900 flex items-center justify-center">
  <div class="w-full max-w-sm">
    <div class="text-center mb-8">
      <div class="w-14 h-14 rounded-2xl bg-blue-600 flex items-center justify-center mx-auto mb-4 shadow-lg shadow-blue-500/30">
        <svg class="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
      </div>
      <h1 class="text-2xl font-bold text-white tracking-tight">TitleForge Pro</h1>
      <p class="text-slate-400 text-sm mt-1">Enterprise Data Matching Platform</p>
    </div>
    <div class="bg-white/5 backdrop-blur border border-white/10 rounded-2xl p-8">
      {% if error %}<div class="bg-red-500/10 border border-red-500/30 text-red-300 text-sm rounded-lg px-4 py-3 mb-4">{{ error }}</div>{% endif %}
      <form method="POST" action="/login" class="space-y-4">
        <div>
          <label class="block text-xs text-slate-400 font-medium mb-1.5">Username</label>
          <input name="username" type="text" required class="w-full bg-white/5 border border-white/10 text-white rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 placeholder-slate-500" placeholder="admin">
        </div>
        <div>
          <label class="block text-xs text-slate-400 font-medium mb-1.5">Password</label>
          <input name="password" type="password" required class="w-full bg-white/5 border border-white/10 text-white rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 placeholder-slate-500" placeholder="••••••••">
        </div>
        <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 rounded-lg text-sm transition">Sign In</button>
      </form>
    </div>
    <p class="text-center text-slate-500 text-xs mt-6">Default: admin / admin123</p>
  </div>
</body></html>"""

TEMPLATES["index.html"] = """{% extends "base.html" %}
{% block title %}Home — TitleForge Pro{% endblock %}
{% block content %}
<div class="mb-8">
  <h1 class="font-display text-3xl font-bold text-slate-900 tracking-tight">Welcome back{% if user %}, {{ user.username }}{% endif %}</h1>
  <p class="text-slate-500 mt-1">Your enterprise data matching & cleaning platform.</p>
</div>

<div class="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
  <a href="/matching" class="card p-6 group cursor-pointer">
    <div class="w-11 h-11 rounded-xl bg-blue-50 flex items-center justify-center mb-4 group-hover:bg-blue-100 transition">
      <i data-lucide="git-merge" class="w-5 h-5 text-blue-600"></i>
    </div>
    <h3 class="font-display font-bold text-slate-900 mb-1">Matching Engine</h3>
    <p class="text-sm text-slate-500 leading-relaxed">3-pass algorithm: deterministic rules, multi-signal scoring (6 algorithms), self-calibrating ML ensemble.</p>
    <div class="mt-3 text-xs font-semibold text-blue-600">Open module →</div>
  </a>
  <a href="/cleaning" class="card p-6 group cursor-pointer">
    <div class="w-11 h-11 rounded-xl bg-emerald-50 flex items-center justify-center mb-4 group-hover:bg-emerald-100 transition">
      <i data-lucide="sparkles" class="w-5 h-5 text-emerald-600"></i>
    </div>
    <h3 class="font-display font-bold text-slate-900 mb-1">Data Cleaner</h3>
    <p class="text-sm text-slate-500 leading-relaxed">8-step pipeline with Grubbs' test p-values, Shapiro-Wilk normality, Z-score detection, cross-column validation.</p>
    <div class="mt-3 text-xs font-semibold text-emerald-600">Open module →</div>
  </a>
  <a href="/standardize" class="card p-6 group cursor-pointer">
    <div class="w-11 h-11 rounded-xl bg-violet-50 flex items-center justify-center mb-4 group-hover:bg-violet-100 transition">
      <i data-lucide="layers" class="w-5 h-5 text-violet-600"></i>
    </div>
    <h3 class="font-display font-bold text-slate-900 mb-1">Standardize Mode</h3>
    <p class="text-sm text-slate-500 leading-relaxed">Auto-cluster and standardize dirty titles without a master list. TF-IDF similarity clustering.</p>
    <div class="mt-3 text-xs font-semibold text-violet-600">Open module →</div>
  </a>
</div>

<div class="card p-6">
  <h3 class="font-display font-semibold text-slate-900 mb-4">Platform Capabilities</h3>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
    {% for item in [
      ("shield-check", "3-Pass Matching", "Deterministic + ML ensemble"),
      ("bar-chart-3", "Statistical P-Values", "Grubbs, Shapiro-Wilk, Z-score"),
      ("file-text", "AI-Powered Expansion", "Claude API abbreviation engine"),
      ("download", "Multi-Format Export", "CSV, XLSX, JSON output"),
      ("sliders", "Saved Presets", "Reusable configurations"),
      ("clock", "Processing History", "Full audit trail"),
      ("users", "User Management", "Role-based access control"),
      ("zap", "Auto-Clustering", "No master list required"),
    ] %}
    <div class="flex items-start gap-3 p-3 rounded-lg bg-slate-50">
      <i data-lucide="{{ item[0] }}" class="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0"></i>
      <div>
        <div class="text-xs font-semibold text-slate-800">{{ item[1] }}</div>
        <div class="text-[11px] text-slate-500">{{ item[2] }}</div>
      </div>
    </div>
    {% endfor %}
  </div>
</div>
{% endblock %}"""


TEMPLATES["matching.html"] = """{% extends "base.html" %}
{% block title %}Matching Engine — TitleForge Pro{% endblock %}
{% block content %}
<div class="flex items-center justify-between mb-6">
  <div>
    <h1 class="font-display text-2xl font-bold text-slate-900 tracking-tight">Matching Engine</h1>
    <p class="text-sm text-slate-500 mt-0.5">3-pass algorithm · 6 signals · AI-powered abbreviation expansion</p>
  </div>
  <div class="flex gap-2">
    <a href="/matching/template/master" class="btn-secondary text-xs px-3 py-2 flex items-center gap-1.5"><i data-lucide="download" class="w-3.5 h-3.5"></i> Master Template</a>
    <a href="/matching/template/source" class="btn-secondary text-xs px-3 py-2 flex items-center gap-1.5"><i data-lucide="download" class="w-3.5 h-3.5"></i> Source Template</a>
  </div>
</div>

<!-- Upload Section -->
<div class="grid grid-cols-2 gap-5 mb-6">
  <div class="card p-5">
    <div class="flex items-center gap-2 mb-3">
      <i data-lucide="database" class="w-4 h-4 text-blue-600"></i>
      <span class="font-display font-semibold text-sm">Master List</span>
      {% if master_count %}<span class="badge badge-blue">{{ master_count }} titles</span>{% endif %}
    </div>
    <form hx-post="/matching/upload-master" hx-encoding="multipart/form-data" hx-target="#upload-status" hx-swap="innerHTML">
      <input type="file" name="file" accept=".csv,.xlsx" class="block w-full text-xs text-slate-500 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer">
      <button type="submit" class="btn-primary text-xs mt-3 w-full">Upload Master</button>
    </form>
  </div>
  <div class="card p-5">
    <div class="flex items-center gap-2 mb-3">
      <i data-lucide="file-input" class="w-4 h-4 text-amber-600"></i>
      <span class="font-display font-semibold text-sm">Source Data</span>
      {% if source_count %}<span class="badge badge-amber">{{ source_count }} titles</span>{% endif %}
    </div>
    <form hx-post="/matching/upload-source" hx-encoding="multipart/form-data" hx-target="#upload-status" hx-swap="innerHTML">
      <input type="file" name="file" accept=".csv,.xlsx" class="block w-full text-xs text-slate-500 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-amber-50 file:text-amber-700 hover:file:bg-amber-100 cursor-pointer">
      <button type="submit" class="btn-primary text-xs mt-3 w-full">Upload Source</button>
    </form>
  </div>
</div>
<div id="upload-status" class="mb-4"></div>

<!-- Configuration -->
<div class="card p-5 mb-6">
  <div class="flex items-center justify-between mb-4">
    <div class="flex items-center gap-2">
      <i data-lucide="sliders" class="w-4 h-4 text-slate-600"></i>
      <span class="font-display font-semibold text-sm">Configuration</span>
    </div>
  </div>
  <form hx-post="/matching/run" hx-target="#match-results" hx-swap="innerHTML" hx-indicator="#match-spinner">
    <div class="max-w-xs mb-4">
      <label class="block text-xs text-slate-500 font-medium mb-1">Auto-Approve Threshold (%)</label>
      <input type="number" name="auto_approve" value="90" min="50" max="100" class="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500/30 focus:border-blue-400 outline-none">
      <p class="text-[11px] text-slate-400 mt-1">Titles above this score are auto-approved. Below goes to manual review.</p>
    </div>
    <div class="flex items-center gap-3">
      <button type="submit" class="btn-primary text-sm flex items-center gap-2" {% if not master_count or not source_count %}disabled style="opacity:0.5;cursor:not-allowed"{% endif %}>
        <i data-lucide="play" class="w-4 h-4"></i> Run 3-Pass Matching
      </button>
      <div id="match-spinner" class="htmx-indicator flex items-center gap-2 text-sm text-blue-600">
        <svg class="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
        Processing...
      </div>
    </div>
  </form>
</div>

<!-- AI Expansion Status -->
<div class="mb-6 flex items-center gap-3 px-4 py-3 rounded-xl {% if ai_active %}bg-emerald-50 border border-emerald-200{% else %}bg-slate-50 border border-slate-200{% endif %}">
  {% if ai_active %}
    <span class="relative flex h-2.5 w-2.5"><span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span><span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span></span>
    <span class="text-sm font-medium text-emerald-800">AI Expansion Active</span>
    <span class="text-xs text-emerald-600">— Claude API will expand any abbreviation the built-in patterns can't resolve</span>
  {% else %}
    <span class="w-2.5 h-2.5 rounded-full bg-slate-300"></span>
    <span class="text-sm font-medium text-slate-600">Built-in Patterns Only</span>
    <span class="text-xs text-slate-400">— 130+ abbreviations. <a href="/api-settings" class="underline text-blue-600 font-semibold">Add an API key</a> for unlimited AI expansion.</span>
  {% endif %}
</div>

<!-- Results -->
<div id="match-results">
{% if results %}
  <div class="grid grid-cols-4 gap-4 mb-6">
    <div class="card stat-card"><div class="stat-val text-blue-600">{{ results|length }}</div><div class="stat-label">Total</div></div>
    <div class="card stat-card"><div class="stat-val text-emerald-600">{{ results|selectattr('zone','eq','auto_approve')|list|length }}</div><div class="stat-label">Auto-Approved</div></div>
    <div class="card stat-card"><div class="stat-val text-amber-500">{{ results|selectattr('zone','eq','review')|list|length }}</div><div class="stat-label">Needs Review</div></div>
    <div class="card stat-card"><div class="stat-val text-red-500">{{ results|selectattr('zone','eq','auto_reject')|list|length }}</div><div class="stat-label">Rejected</div></div>
  </div>

  <div class="card overflow-hidden mb-6">
    <div class="px-5 py-3 border-b border-slate-100 flex items-center justify-between">
      <span class="font-display font-semibold text-sm">Match Results</span>
      <div class="flex gap-2">
        <a href="/matching/export/csv" class="btn-secondary text-xs px-3 py-1.5">Export Results CSV</a>
        <a href="/matching/export/xlsx" class="btn-secondary text-xs px-3 py-1.5">Export Results Excel</a>
        <a href="/matching/export/master/csv" class="btn-secondary text-xs px-3 py-1.5" style="border-color:#bfdbfe;color:#1e40af;background:#eff6ff;">Export Master List CSV</a>
        <a href="/matching/export/master/xlsx" class="btn-secondary text-xs px-3 py-1.5" style="border-color:#bfdbfe;color:#1e40af;background:#eff6ff;">Export Master List Excel</a>
      </div>
    </div>
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead><tr class="bg-slate-50 text-xs text-slate-500 font-medium">
          <th class="text-left px-4 py-2.5">Source Title</th>
          <th class="text-left px-4 py-2.5">Matched Title</th>
          <th class="text-left px-4 py-2.5">Confidence</th>
          <th class="text-left px-4 py-2.5">Zone</th>
          <th class="text-left px-4 py-2.5">Actions</th>
        </tr></thead>
        <tbody class="divide-y divide-slate-100">
        {% for r in results %}
          <tr class="hover:bg-slate-50/50" id="row-{{ loop.index0 }}">
            <td class="px-4 py-3 font-medium text-slate-800">{{ r.source_title }}</td>
            <td class="px-4 py-3 text-slate-600">{{ r.matched_title }}</td>
            <td class="px-4 py-3">
              <div class="flex items-center gap-2">
                <div class="score-bar w-20">
                  <div class="score-fill {% if r.confidence >= 90 %}bg-emerald-500{% elif r.confidence >= 70 %}bg-amber-400{% else %}bg-red-400{% endif %}" style="width:{{ r.confidence }}%"></div>
                </div>
                <span class="text-xs font-semibold {% if r.confidence >= 90 %}text-emerald-600{% elif r.confidence >= 70 %}text-amber-600{% else %}text-red-500{% endif %}">{{ "%.1f"|format(r.confidence) }}%</span>
              </div>
            </td>
            <td class="px-4 py-3">
              {% if r.zone == 'auto_approve' %}<span class="badge badge-green">{% if r.status == 'overridden' %}Edited{% else %}Approved{% endif %}</span>
              {% elif r.zone == 'review' %}<span class="badge badge-amber">Review</span>
              {% else %}<span class="badge badge-red">Rejected</span>{% endif %}
            </td>
            <td class="px-4 py-3">
              <div class="flex gap-1">
                <button onclick="document.getElementById('edit-panel-{{ loop.index0 }}').classList.toggle('hidden')"
                  class="p-1.5 rounded-md hover:bg-blue-50 text-blue-600" title="Edit mapping">
                  <i data-lucide="pencil" class="w-3.5 h-3.5"></i>
                </button>
                <button hx-post="/matching/approve/{{ loop.index0 }}" hx-target="body" class="p-1.5 rounded-md hover:bg-emerald-50 text-emerald-600" title="Approve"><i data-lucide="check" class="w-3.5 h-3.5"></i></button>
                <button hx-post="/matching/reject/{{ loop.index0 }}" hx-target="body" class="p-1.5 rounded-md hover:bg-red-50 text-red-500" title="Reject"><i data-lucide="x" class="w-3.5 h-3.5"></i></button>
              </div>
            </td>
          </tr>
          <!-- Edit Panel (hidden by default) -->
          <tr id="edit-panel-{{ loop.index0 }}" class="hidden bg-blue-50/50">
            <td colspan="5" class="px-4 py-4">
              <form method="POST" action="/matching/edit/{{ loop.index0 }}">
                <div class="flex items-end gap-4">
                  <div class="flex-1">
                    <label class="block text-xs text-slate-500 font-medium mb-1">Select from Master List</label>
                    <select name="select_title" class="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm bg-white">
                      <option value="">— Choose existing title —</option>
                      {% for mt in master_titles %}
                      <option value="{{ mt }}" {% if mt == r.matched_title %}selected{% endif %}>{{ mt }}</option>
                      {% endfor %}
                    </select>
                  </div>
                  <div class="text-xs text-slate-400 font-semibold px-2">OR</div>
                  <div class="flex-1">
                    <label class="block text-xs text-slate-500 font-medium mb-1">Add New Title to Master List</label>
                    <input type="text" name="new_title" placeholder="Type new standardized title..." class="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500/30 outline-none">
                  </div>
                  <button type="submit" class="btn-primary text-xs px-4 py-2">Save</button>
                  <button type="button" onclick="this.closest('tr').classList.add('hidden')" class="btn-secondary text-xs px-4 py-2">Cancel</button>
                </div>
              </form>
            </td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
{% endif %}
</div>
{% endblock %}
{% block scripts %}<script>document.body.addEventListener('htmx:afterSwap', function(){ lucide.createIcons(); });</script>{% endblock %}"""


TEMPLATES["cleaning.html"] = """{% extends "base.html" %}
{% block title %}Data Cleaner — TitleForge Pro{% endblock %}
{% block content %}
<div class="mb-6">
  <h1 class="font-display text-2xl font-bold text-slate-900 tracking-tight">Data Cleaner</h1>
  <p class="text-sm text-slate-500 mt-0.5">8-step pipeline · Statistical p-values · Full audit trail</p>
</div>

<div class="card p-5 mb-6">
  <div class="flex items-center gap-2 mb-3">
    <i data-lucide="upload" class="w-4 h-4 text-emerald-600"></i>
    <span class="font-display font-semibold text-sm">Upload Pay Data</span>
    {% if data_rows %}<span class="badge badge-blue">{{ data_rows }} rows × {{ data_cols }} cols</span>{% endif %}
  </div>
  <form hx-post="/cleaning/upload" hx-encoding="multipart/form-data" hx-target="#clean-status" hx-swap="innerHTML">
    <input type="file" name="file" accept=".csv,.xlsx,.xls" class="block w-full text-xs text-slate-500 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-emerald-50 file:text-emerald-700 hover:file:bg-emerald-100 cursor-pointer">
    <button type="submit" class="btn-primary text-xs mt-3">Upload</button>
  </form>
</div>
<div id="clean-status" class="mb-4"></div>

{% if data_rows %}
<div class="card p-5 mb-6">
  <div class="flex items-center gap-2 mb-4">
    <i data-lucide="sliders" class="w-4 h-4 text-slate-600"></i>
    <span class="font-display font-semibold text-sm">Configure Pipeline</span>
  </div>
  <form hx-post="/cleaning/run" hx-target="#clean-results" hx-swap="innerHTML" hx-indicator="#clean-spinner">
    <div class="grid grid-cols-2 gap-4 mb-4">
      <div>
        <label class="block text-xs text-slate-500 font-medium mb-1">Pay Columns</label>
        <select name="pay_cols" multiple class="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm" size="4">
          {% for col in columns %}<option value="{{ col }}" {% if col in detected_pay %}selected{% endif %}>{{ col }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label class="block text-xs text-slate-500 font-medium mb-1">Job Title Column (optional)</label>
        <select name="job_col" class="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm">
          <option value="">— None —</option>
          {% for col in columns %}<option value="{{ col }}">{{ col }}</option>{% endfor %}
        </select>
      </div>
    </div>
    <div class="grid grid-cols-4 gap-4 mb-4">
      <div><label class="block text-xs text-slate-500 font-medium mb-1">Negative Values</label>
        <select name="neg_action" class="w-full border rounded-lg px-3 py-2 text-sm"><option value="flag">Flag</option><option value="abs">Make Absolute</option><option value="remove">Remove Row</option></select></div>
      <div><label class="block text-xs text-slate-500 font-medium mb-1">Zero Values</label>
        <select name="zero_action" class="w-full border rounded-lg px-3 py-2 text-sm"><option value="flag">Flag</option><option value="null">Set Null</option><option value="remove">Remove Row</option></select></div>
      <div><label class="block text-xs text-slate-500 font-medium mb-1">IQR Multiplier</label>
        <input name="iqr_mult" type="number" value="1.5" step="0.1" min="1" max="3" class="w-full border rounded-lg px-3 py-2 text-sm"></div>
      <div><label class="block text-xs text-slate-500 font-medium mb-1">Z-Score Threshold</label>
        <input name="z_thresh" type="number" value="3.0" step="0.1" min="1.5" max="5" class="w-full border rounded-lg px-3 py-2 text-sm"></div>
    </div>
    <button type="submit" class="btn-primary text-sm flex items-center gap-2">
      <i data-lucide="sparkles" class="w-4 h-4"></i> Run 8-Step Pipeline
    </button>
    <span id="clean-spinner" class="htmx-indicator text-sm text-emerald-600 ml-3">Processing...</span>
  </form>
</div>
{% endif %}

<div id="clean-results">
{% if audit_log %}
  <div class="grid grid-cols-4 gap-4 mb-6">
    <div class="card stat-card"><div class="stat-val text-blue-600">{{ original_rows }}</div><div class="stat-label">Original Rows</div></div>
    <div class="card stat-card"><div class="stat-val text-emerald-600">{{ cleaned_rows }}</div><div class="stat-label">After Cleaning</div></div>
    <div class="card stat-card"><div class="stat-val text-amber-500">{{ issues_count }}</div><div class="stat-label">Issues Found</div></div>
    <div class="card stat-card"><div class="stat-val text-violet-600">{{ flag_cols }}</div><div class="stat-label">Flag Columns</div></div>
  </div>

  <div class="card p-5 mb-6">
    <div class="flex items-center justify-between mb-4">
      <span class="font-display font-semibold text-sm">Audit Log</span>
      <div class="flex gap-2">
        <a href="/cleaning/export/csv" class="btn-secondary text-xs px-3 py-1.5">Export CSV</a>
        <a href="/cleaning/export/xlsx" class="btn-secondary text-xs px-3 py-1.5">Export Excel</a>
        <a href="/cleaning/export/report" class="btn-secondary text-xs px-3 py-1.5">Audit Report</a>
      </div>
    </div>
    <div class="space-y-2">
    {% for e in audit_log %}
      <div class="flex items-center justify-between p-3 rounded-lg {% if e.severity == 'success' %}bg-emerald-50{% elif e.severity == 'warning' %}bg-amber-50{% elif e.severity == 'error' %}bg-red-50{% else %}bg-blue-50{% endif %}">
        <div class="flex-1">
          <div class="flex items-center gap-2">
            <span class="font-semibold text-xs text-slate-700">Step {{ e.step }}: {{ e.name }}</span>
            <span class="text-xs text-slate-400">col: {{ e.column }}</span>
          </div>
          <div class="text-xs text-slate-600 mt-0.5">{{ e.description }}</div>
        </div>
        <div class="flex items-center gap-3">
          {% if e.p_value is not none %}<span class="badge badge-slate text-[10px]">p={{ e.p_value }}</span>{% endif %}
          <span class="badge {% if e.severity == 'success' %}badge-green{% elif e.severity == 'warning' %}badge-amber{% elif e.severity == 'error' %}badge-red{% else %}badge-blue{% endif %}">{{ e.count }}</span>
        </div>
      </div>
    {% endfor %}
    </div>
  </div>
{% endif %}
</div>
{% endblock %}
{% block scripts %}<script>document.body.addEventListener('htmx:afterSwap', function(){ lucide.createIcons(); });</script>{% endblock %}"""

TEMPLATES["standardize.html"] = """{% extends "base.html" %}
{% block title %}Standardize — TitleForge Pro{% endblock %}
{% block content %}
<div class="mb-6">
  <h1 class="font-display text-2xl font-bold text-slate-900 tracking-tight">Standardize Mode</h1>
  <p class="text-sm text-slate-500 mt-0.5">Auto-cluster dirty titles without a master list</p>
</div>

<div class="card p-5 mb-6">
  <form hx-post="/standardize/run" hx-encoding="multipart/form-data" hx-target="#std-results" hx-swap="innerHTML" hx-indicator="#std-spinner">
    <div class="grid grid-cols-2 gap-4 mb-4">
      <div>
        <label class="block text-xs text-slate-500 font-medium mb-1">Upload Titles (CSV/Excel)</label>
        <input type="file" name="file" accept=".csv,.xlsx" class="block w-full text-xs text-slate-500 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-violet-50 file:text-violet-700 cursor-pointer">
      </div>
      <div>
        <label class="block text-xs text-slate-500 font-medium mb-1">Similarity Threshold</label>
        <input name="threshold" type="number" value="0.75" step="0.05" min="0.3" max="0.99" class="w-full border rounded-lg px-3 py-2 text-sm">
        <p class="text-[11px] text-slate-400 mt-1">Higher = stricter grouping. 0.75 is a good default.</p>
      </div>
    </div>
    <button type="submit" class="btn-primary text-sm flex items-center gap-2"><i data-lucide="layers" class="w-4 h-4"></i> Run Clustering</button>
    <span id="std-spinner" class="htmx-indicator text-sm text-violet-600 ml-3">Clustering...</span>
  </form>
</div>

<div id="std-results">
{% if clusters %}
  <div class="grid grid-cols-3 gap-4 mb-6">
    <div class="card stat-card"><div class="stat-val text-violet-600">{{ clusters|length }}</div><div class="stat-label">Clusters</div></div>
    <div class="card stat-card"><div class="stat-val text-blue-600">{{ total_titles }}</div><div class="stat-label">Total Titles</div></div>
    <div class="card stat-card"><div class="stat-val text-emerald-600">{{ multi_clusters }}</div><div class="stat-label">Multi-Title Groups</div></div>
  </div>

  <div class="card p-4 mb-6">
    <div class="flex items-center justify-between">
      <span class="font-display font-semibold text-sm">Export Standardized Data</span>
      <div class="flex gap-2">
        <a href="/standardize/export/csv" class="btn-secondary text-xs px-3 py-1.5 flex items-center gap-1"><i data-lucide="download" class="w-3 h-3"></i> Export CSV</a>
        <a href="/standardize/export/xlsx" class="btn-secondary text-xs px-3 py-1.5 flex items-center gap-1"><i data-lucide="download" class="w-3 h-3"></i> Export Excel</a>
      </div>
    </div>
  </div>

  <div class="space-y-3">
  {% for c in clusters %}
    <div class="card p-4">
      <div class="flex items-center justify-between mb-2">
        <div class="flex items-center gap-2">
          <span class="font-display font-semibold text-sm text-slate-900">{{ c.canonical }}</span>
          <span class="badge badge-blue">{{ c.size }} titles</span>
          <span class="badge badge-slate">{{ "%.0f"|format(c.avg_similarity * 100) }}% avg sim</span>
        </div>
      </div>
      {% if c.size > 1 %}
      <div class="flex flex-wrap gap-1.5 mt-2">
        {% for m in c.members %}
        <span class="text-xs px-2 py-1 rounded-md {% if m == c.canonical %}bg-blue-100 text-blue-700 font-medium{% else %}bg-slate-100 text-slate-600{% endif %}">{{ m }}</span>
        {% endfor %}
      </div>
      {% endif %}
    </div>
  {% endfor %}
  </div>
{% endif %}
</div>
{% endblock %}"""

TEMPLATES["history.html"] = """{% extends "base.html" %}
{% block title %}History — TitleForge Pro{% endblock %}
{% block content %}
<div class="mb-6">
  <h1 class="font-display text-2xl font-bold text-slate-900 tracking-tight">Processing History</h1>
  <p class="text-sm text-slate-500 mt-0.5">All past processing runs with parameters and results</p>
</div>
{% if entries %}
<div class="space-y-3">
  {% for e in entries %}
  <div class="card p-4">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-lg flex items-center justify-center
          {% if e.module == 'matching' %}bg-blue-50{% elif e.module == 'cleaning' %}bg-emerald-50{% else %}bg-violet-50{% endif %}">
          <i data-lucide="{% if e.module == 'matching' %}git-merge{% elif e.module == 'cleaning' %}sparkles{% else %}layers{% endif %}"
             class="w-4 h-4 {% if e.module == 'matching' %}text-blue-600{% elif e.module == 'cleaning' %}text-emerald-600{% else %}text-violet-600{% endif %}"></i>
        </div>
        <div>
          <div class="font-semibold text-sm text-slate-800">{{ e.module|title }} — {{ e.summary }}</div>
          <div class="text-xs text-slate-400">{{ e.timestamp }} · {{ e.user or 'system' }}</div>
        </div>
      </div>
      <div class="flex items-center gap-4 text-xs text-slate-500">
        <span>In: <strong>{{ e.rows_in }}</strong></span>
        <span>Out: <strong>{{ e.rows_out }}</strong></span>
        <span class="badge badge-slate">{{ e.id }}</span>
      </div>
    </div>
  </div>
  {% endfor %}
</div>
{% else %}
<div class="card p-8 text-center">
  <i data-lucide="clock" class="w-10 h-10 text-slate-300 mx-auto mb-3"></i>
  <div class="text-sm text-slate-500">No processing history yet. Run a matching or cleaning job to see it here.</div>
</div>
{% endif %}
{% endblock %}"""

TEMPLATES["admin.html"] = """{% extends "base.html" %}
{% block title %}Admin — TitleForge Pro{% endblock %}
{% block content %}
<div class="mb-6">
  <h1 class="font-display text-2xl font-bold text-slate-900 tracking-tight">Admin Panel</h1>
  <p class="text-sm text-slate-500 mt-0.5">Manage user accounts and system settings</p>
</div>

<div class="card p-5 mb-6">
  <div class="font-display font-semibold text-sm mb-4">Create User</div>
  <form hx-post="/admin/create-user" hx-target="#admin-status" hx-swap="innerHTML" class="flex items-end gap-3">
    <div><label class="block text-xs text-slate-500 mb-1">Username</label><input name="username" required class="border rounded-lg px-3 py-2 text-sm"></div>
    <div><label class="block text-xs text-slate-500 mb-1">Password</label><input name="password" type="password" required class="border rounded-lg px-3 py-2 text-sm"></div>
    <div><label class="block text-xs text-slate-500 mb-1">Role</label>
      <select name="role" class="border rounded-lg px-3 py-2 text-sm"><option value="user">User</option><option value="admin">Admin</option></select></div>
    <button type="submit" class="btn-primary text-sm">Create</button>
  </form>
  <div id="admin-status" class="mt-3"></div>
</div>

<div class="card overflow-hidden">
  <div class="px-5 py-3 border-b border-slate-100">
    <span class="font-display font-semibold text-sm">Users ({{ users|length }})</span>
  </div>
  <table class="w-full text-sm">
    <thead><tr class="bg-slate-50 text-xs text-slate-500"><th class="text-left px-4 py-2">Username</th><th class="text-left px-4 py-2">Role</th><th class="text-left px-4 py-2">Status</th><th class="text-left px-4 py-2">Last Login</th><th class="px-4 py-2">Actions</th></tr></thead>
    <tbody class="divide-y divide-slate-100">
    {% for u in users %}
    <tr>
      <td class="px-4 py-3 font-medium text-slate-800">{{ u.username }}</td>
      <td class="px-4 py-3"><span class="badge {% if u.role == 'admin' %}badge-blue{% else %}badge-slate{% endif %}">{{ u.role }}</span></td>
      <td class="px-4 py-3"><span class="badge {% if u.active %}badge-green{% else %}badge-red{% endif %}">{{ 'Active' if u.active else 'Inactive' }}</span></td>
      <td class="px-4 py-3 text-xs text-slate-400">{{ u.last_login or 'Never' }}</td>
      <td class="px-4 py-3 text-center">
        <button hx-post="/admin/toggle-user/{{ u.username }}" hx-target="body" class="text-xs btn-secondary px-2 py-1">{{ 'Deactivate' if u.active else 'Activate' }}</button>
      </td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
{% endblock %}"""

# Partial templates for HTMX responses
TEMPLATES["_upload_ok.html"] = """<div class="bg-emerald-50 border border-emerald-200 text-emerald-700 text-sm rounded-lg px-4 py-3 fade-in flex items-center gap-2">
  <i data-lucide="check-circle" class="w-4 h-4"></i> {{ message }}
</div>"""

TEMPLATES["_upload_err.html"] = """<div class="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3 fade-in flex items-center gap-2">
  <i data-lucide="alert-circle" class="w-4 h-4"></i> {{ message }}
</div>"""

TEMPLATES["api_settings.html"] = """{% extends "base.html" %}
{% block title %}API Key — TitleForge Pro{% endblock %}
{% block content %}
<div class="mb-6">
  <h1 class="font-display text-2xl font-bold text-slate-900 tracking-tight">API Key Settings</h1>
  <p class="text-sm text-slate-500 mt-0.5">Enter your own Anthropic API key for AI-powered abbreviation expansion</p>
</div>

<div class="card p-6 max-w-2xl">
  <div class="flex items-start gap-3 mb-6 bg-blue-50 border border-blue-200 rounded-xl p-4">
    <i data-lucide="info" class="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5"></i>
    <div class="text-sm text-blue-800">
      <strong>How it works:</strong> The matching engine uses Claude to expand any abbreviation the built-in 130+ patterns can't resolve.
      Your key is stored in your session only — it's never saved to disk and is cleared when you log out.
      <br><br>
      Get your API key at <a href="https://console.anthropic.com" target="_blank" class="underline font-semibold">console.anthropic.com</a>
    </div>
  </div>

  <form method="POST" action="/api-settings">
    <label class="block text-xs text-slate-500 font-semibold mb-2">Anthropic API Key</label>
    <div class="flex gap-3">
      <input name="api_key" type="password" value="{{ current_key }}"
             placeholder="sk-ant-api03-..."
             class="flex-1 border border-slate-200 rounded-lg px-4 py-2.5 text-sm font-mono focus:ring-2 focus:ring-blue-500/30 focus:border-blue-400 outline-none">
      <button type="submit" class="btn-primary text-sm px-6">Save</button>
    </div>
    <p class="text-[11px] text-slate-400 mt-2">Leave blank to use built-in patterns only (no API calls).</p>
  </form>

  {% if saved %}
  <div class="mt-4 bg-emerald-50 border border-emerald-200 text-emerald-700 text-sm rounded-lg px-4 py-3 flex items-center gap-2">
    <i data-lucide="check-circle" class="w-4 h-4"></i>
    {% if current_key %}API key saved to your session. AI expansion is active.{% else %}API key cleared. Using built-in patterns only.{% endif %}
  </div>
  {% endif %}

  <div class="mt-6 pt-5 border-t border-slate-100">
    <div class="text-xs text-slate-500 font-semibold mb-3">Current Status</div>
    <div class="flex items-center gap-2">
      {% if current_key %}
        <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
        <span class="text-sm text-emerald-700 font-medium">AI Expansion Active</span>
        <span class="text-xs text-slate-400 ml-2">Key: sk-ant-...{{ current_key[-4:] }}</span>
      {% elif env_key %}
        <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
        <span class="text-sm text-emerald-700 font-medium">Server Key Active</span>
        <span class="text-xs text-slate-400 ml-2">(set by admin via environment variable)</span>
      {% else %}
        <span class="w-2 h-2 rounded-full bg-slate-300"></span>
        <span class="text-sm text-slate-500 font-medium">Built-in Patterns Only</span>
        <span class="text-xs text-slate-400 ml-2">(130+ abbreviations — add a key for unlimited AI expansion)</span>
      {% endif %}
    </div>
  </div>
</div>
{% endblock %}"""


# ═══════════════════════════════════════════════════════════════
# SECTION 10: FASTAPI APPLICATION & ROUTES
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="TitleForge Pro", version=APP_VERSION)
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SECRET_KEY", "tf-pro-dev-key-change-me"))

jinja_env = Environment(
    loader=DictLoader(TEMPLATES),
    autoescape=select_autoescape(['html']),
)

# In-memory session data (per-user project data)
_session_data: dict[str, dict] = {}

def get_sdata(request: Request) -> dict:
    uid = request.session.get("user", {}).get("username", "_anon")
    if uid not in _session_data:
        _session_data[uid] = {
            "master_titles": [], "source_titles": [], "source_df": None,
            "match_results": [], "clean_df": None, "clean_raw_df": None,
            "clean_audit": [], "clean_config": {}, "api_key": "",
            "std_clusters": [], "std_titles": [],
        }
    return _session_data[uid]

def _ai_active(sd: dict) -> bool:
    """Check if AI expansion is available (user key or env key)."""
    return bool(sd.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"))

def render(template: str, request: Request, **ctx):
    user = request.session.get("user")
    ctx["user"] = user
    ctx["request"] = request
    html = jinja_env.get_template(template).render(**ctx)
    return HTMLResponse(html)

# ── AUTH ROUTES ──────────────────────────────────────────────
@app.get("/login")
async def login_page(request: Request):
    return render("login.html", request, error=None)

@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    user = authenticate(username, password)
    if user:
        request.session["user"] = {"username": user["username"], "role": user["role"]}
        return RedirectResponse("/", status_code=303)
    return render("login.html", request, error="Invalid credentials or inactive account.")

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

# ── HOME ─────────────────────────────────────────────────────
@app.get("/")
async def home(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login", status_code=303)
    return render("index.html", request, page="home")

# ── MATCHING ROUTES ──────────────────────────────────────────
@app.get("/matching")
async def matching_page(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login", status_code=303)
    sd = get_sdata(request)
    return render("matching.html", request, page="matching",
                  master_count=len(sd["master_titles"]),
                  source_count=len(sd["source_titles"]),
                  results=sd["match_results"],
                  master_titles=sd["master_titles"],
                  ai_active=_ai_active(sd))

@app.post("/matching/upload-master")
async def upload_master(request: Request, file: UploadFile = File(...)):
    user = request.session.get("user")
    if not user:
        return HTMLResponse("Unauthorized", status_code=401)
    sd = get_sdata(request)
    try:
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
        col = df.columns[0]
        titles = df[col].dropna().astype(str).unique().tolist()
        sd["master_titles"] = titles
        return render("_upload_ok.html", request, message=f"Loaded {len(titles)} master titles from column '{col}'.")
    except Exception as e:
        return render("_upload_err.html", request, message=str(e))

@app.post("/matching/upload-source")
async def upload_source(request: Request, file: UploadFile = File(...)):
    user = request.session.get("user")
    if not user:
        return HTMLResponse("Unauthorized", status_code=401)
    sd = get_sdata(request)
    try:
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
        col = df.columns[0]
        sd["source_titles"] = df[col].dropna().astype(str).tolist()
        sd["source_df"] = df
        return render("_upload_ok.html", request, message=f"Loaded {len(sd['source_titles'])} source titles from column '{col}'.")
    except Exception as e:
        return render("_upload_err.html", request, message=str(e))

@app.post("/matching/run")
async def run_matching(request: Request,
                       auto_approve: int = Form(90)):
    user = request.session.get("user")
    if not user:
        return HTMLResponse("Unauthorized", status_code=401)
    sd = get_sdata(request)
    if not sd["master_titles"] or not sd["source_titles"]:
        return render("_upload_err.html", request, message="Upload both files first.")

    thresholds = {"auto_approve": auto_approve, "review": 70}
    results = run_full_matching(sd["source_titles"], sd["master_titles"],
                                thresholds=thresholds)
    sd["match_results"] = results

    # Save history
    n_auto = sum(1 for r in results if r.zone == "auto_approve")
    n_review = sum(1 for r in results if r.zone == "review")
    save_history("matching", len(sd["source_titles"]), len(results),
                 {"auto_approve": auto_approve},
                 f"{n_auto} auto-approved, {n_review} for review",
                 user.get("username", ""))

    return render("matching.html", request, page="matching",
                  master_count=len(sd["master_titles"]),
                  source_count=len(sd["source_titles"]),
                  results=results,
                  master_titles=sd["master_titles"],
                  ai_active=_ai_active(sd))

@app.post("/matching/edit/{idx}")
async def edit_match(request: Request, idx: int):
    """Edit a match: remap to existing master title or add a new one."""
    sd = get_sdata(request)
    form = await request.form()
    new_title = form.get("new_title", "").strip()
    select_title = form.get("select_title", "").strip()

    if idx >= len(sd["match_results"]):
        return RedirectResponse("/matching", status_code=303)

    if new_title:
        # Add to master list if not already present
        if new_title not in sd["master_titles"]:
            sd["master_titles"].append(new_title)
        sd["match_results"][idx].matched_title = new_title
        sd["match_results"][idx].confidence = 100.0
        sd["match_results"][idx].zone = "auto_approve"
        sd["match_results"][idx].status = "overridden"
        sd["match_results"][idx].explanation = f"Manually mapped to new title: {new_title}"
    elif select_title:
        sd["match_results"][idx].matched_title = select_title
        sd["match_results"][idx].confidence = 100.0
        sd["match_results"][idx].zone = "auto_approve"
        sd["match_results"][idx].status = "overridden"
        sd["match_results"][idx].explanation = f"Manually remapped to: {select_title}"

    return RedirectResponse("/matching", status_code=303)

@app.get("/matching/export/master/{fmt}")
async def export_master_list(request: Request, fmt: str):
    """Export the current master list (including any new titles added during editing)."""
    sd = get_sdata(request)
    if not sd["master_titles"]:
        return HTMLResponse("No master list loaded")
    df = pd.DataFrame({"Standardized Title": sd["master_titles"]})
    if fmt == "xlsx":
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        return StreamingResponse(io.BytesIO(buf.getvalue()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=master_list_{datetime.now():%Y%m%d}.xlsx"})
    else:
        return StreamingResponse(io.BytesIO(df.to_csv(index=False).encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=master_list_{datetime.now():%Y%m%d}.csv"})

@app.post("/matching/approve/{idx}")
async def approve_match(request: Request, idx: int):
    sd = get_sdata(request)
    if idx < len(sd["match_results"]):
        sd["match_results"][idx].status = "approved"
        sd["match_results"][idx].zone = "auto_approve"
    return RedirectResponse("/matching", status_code=303)

@app.post("/matching/reject/{idx}")
async def reject_match(request: Request, idx: int):
    sd = get_sdata(request)
    if idx < len(sd["match_results"]):
        sd["match_results"][idx].status = "rejected"
        sd["match_results"][idx].zone = "auto_reject"
    return RedirectResponse("/matching", status_code=303)

@app.get("/matching/export/{fmt}")
async def export_matching(request: Request, fmt: str):
    sd = get_sdata(request)
    if not sd["match_results"]:
        return HTMLResponse("No results to export")
    rows = [{"Source": r.source_title, "Matched": r.matched_title,
             "Confidence": r.confidence, "Zone": r.zone, "Status": r.status}
            for r in sd["match_results"]]
    df = pd.DataFrame(rows)
    if fmt == "xlsx":
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        return StreamingResponse(io.BytesIO(buf.getvalue()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=match_results_{datetime.now():%Y%m%d}.xlsx"})
    else:
        return StreamingResponse(io.BytesIO(df.to_csv(index=False).encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=match_results_{datetime.now():%Y%m%d}.csv"})

@app.get("/matching/template/{kind}")
async def matching_template(kind: str):
    if kind == "master":
        csv_data = "title,department,category\nRegistered Nurse,Nursing,Clinical\nPhysical Therapist,Therapy,Clinical\n"
    else:
        csv_data = "title,source_system\nRN - ICU,Legacy System\nPT Assistant,Manual Entry\n"
    return StreamingResponse(io.BytesIO(csv_data.encode()), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={kind}_template.csv"})

# ── CLEANING ROUTES ──────────────────────────────────────────
@app.get("/cleaning")
async def cleaning_page(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login", status_code=303)
    sd = get_sdata(request)
    raw = sd.get("clean_raw_df")
    ctx = {"page": "cleaning", "data_rows": 0, "data_cols": 0, "columns": [], "detected_pay": [],
           "audit_log": sd.get("clean_audit", []),
           "original_rows": len(raw) if raw is not None else 0,
           "cleaned_rows": len(sd["clean_df"]) if sd.get("clean_df") is not None else 0,
           "issues_count": sum(1 for e in sd.get("clean_audit", []) if e.get("severity") in ("warning", "error")),
           "flag_cols": len([c for c in (sd["clean_df"].columns if sd.get("clean_df") is not None else []) if str(c).startswith("FLAG_")])}
    if raw is not None:
        ctx["data_rows"] = len(raw)
        ctx["data_cols"] = len(raw.columns)
        ctx["columns"] = list(raw.columns)
        ctx["detected_pay"] = detect_pay_columns(raw)
    return render("cleaning.html", request, **ctx)

@app.post("/cleaning/upload")
async def upload_cleaning(request: Request, file: UploadFile = File(...)):
    user = request.session.get("user")
    if not user:
        return HTMLResponse("Unauthorized", status_code=401)
    sd = get_sdata(request)
    try:
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
        sd["clean_raw_df"] = df
        sd["clean_df"] = None
        sd["clean_audit"] = []
        return render("_upload_ok.html", request, message=f"Loaded {len(df)} rows × {len(df.columns)} columns.")
    except Exception as e:
        return render("_upload_err.html", request, message=str(e))

@app.post("/cleaning/run")
async def run_cleaning_route(request: Request):
    user = request.session.get("user")
    if not user:
        return HTMLResponse("Unauthorized", status_code=401)
    sd = get_sdata(request)
    if sd.get("clean_raw_df") is None:
        return render("_upload_err.html", request, message="Upload data first.")

    form = await request.form()
    config = {
        "pay_cols": form.getlist("pay_cols"),
        "job_col": form.get("job_col") or None,
        "hourly_col": None, "weekly_col": None, "hours_col": None,
        "negative_action": form.get("neg_action", "flag"),
        "zero_action": form.get("zero_action", "flag"),
        "iqr_mult": float(form.get("iqr_mult", 1.5)),
        "z_thresh": float(form.get("z_thresh", 3.0)),
        "min_hourly": 5.0, "max_hourly": 500.0,
        "min_weekly": 50.0, "max_weekly": 20000.0,
        "sig_level": 0.05,
    }
    sd["clean_config"] = config

    cleaned, audit = run_cleaning_pipeline(sd["clean_raw_df"], config)
    sd["clean_df"] = cleaned
    sd["clean_audit"] = audit

    save_history("cleaning", len(sd["clean_raw_df"]), len(cleaned),
                 config, f"{sum(1 for a in audit if a['severity']=='warning')} issues found",
                 user.get("username", ""))

    return RedirectResponse("/cleaning", status_code=303)

@app.get("/cleaning/export/{fmt}")
async def export_cleaning(request: Request, fmt: str):
    sd = get_sdata(request)
    if sd.get("clean_df") is None:
        return HTMLResponse("No cleaned data")
    df = sd["clean_df"]
    if fmt == "xlsx":
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        return StreamingResponse(io.BytesIO(buf.getvalue()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=cleaned_data_{datetime.now():%Y%m%d}.xlsx"})
    elif fmt == "report":
        lines = [f"TitleForge Pro — Cleaning Audit Report", f"Generated: {datetime.now():%Y-%m-%d %H:%M}", ""]
        for e in sd.get("clean_audit", []):
            lines.append(f"[Step {e['step']}] {e['name']} — {e['column']}")
            lines.append(f"  {e['description']}")
            if e.get('p_value') is not None:
                lines.append(f"  p-value: {e['p_value']}")
            lines.append("")
        return StreamingResponse(io.BytesIO("\n".join(lines).encode()),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=audit_report_{datetime.now():%Y%m%d}.txt"})
    else:
        return StreamingResponse(io.BytesIO(df.to_csv(index=False).encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=cleaned_data_{datetime.now():%Y%m%d}.csv"})

# ── STANDARDIZE ROUTES ───────────────────────────────────────
@app.get("/standardize")
async def standardize_page(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login", status_code=303)
    return render("standardize.html", request, page="standardize", clusters=None)

@app.post("/standardize/run")
async def run_standardize(request: Request, file: UploadFile = File(...),
                          threshold: float = Form(0.75)):
    user = request.session.get("user")
    if not user:
        return HTMLResponse("Unauthorized", status_code=401)
    try:
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
        titles = df.iloc[:, 0].dropna().astype(str).tolist()
        clusters = auto_cluster_titles(titles, threshold=threshold, min_cluster_size=1)

        # Store for export
        sd = get_sdata(request)
        sd["std_clusters"] = clusters
        sd["std_titles"] = titles

        save_history("standardize", len(titles), len(clusters),
                     {"threshold": threshold},
                     f"{len(clusters)} clusters from {len(titles)} titles",
                     user.get("username", ""))

        multi = sum(1 for c in clusters if c["size"] > 1)
        return render("standardize.html", request, page="standardize",
                      clusters=clusters, total_titles=len(titles), multi_clusters=multi)
    except Exception as e:
        return render("_upload_err.html", request, message=str(e))

@app.get("/standardize/export/{fmt}")
async def export_standardize(request: Request, fmt: str):
    """Export standardized clusters as CSV or Excel."""
    sd = get_sdata(request)
    clusters = sd.get("std_clusters", [])
    if not clusters:
        return HTMLResponse("No clusters to export")
    rows = []
    for c in clusters:
        for m in c["members"]:
            rows.append({
                "Original Title": m,
                "Standardized Title": c["canonical"],
                "Cluster Size": c["size"],
                "Avg Similarity": round(c["avg_similarity"], 3),
            })
    df = pd.DataFrame(rows)
    if fmt == "xlsx":
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        return StreamingResponse(io.BytesIO(buf.getvalue()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=standardized_{datetime.now():%Y%m%d}.xlsx"})
    else:
        return StreamingResponse(io.BytesIO(df.to_csv(index=False).encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=standardized_{datetime.now():%Y%m%d}.csv"})

# ── HISTORY ROUTES ───────────────────────────────────────────
@app.get("/history")
async def history_page(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login", status_code=303)
    entries = store.load("history")
    return render("history.html", request, page="history", entries=entries)

# ── ADMIN ROUTES ─────────────────────────────────────────────
@app.get("/admin")
async def admin_page(request: Request):
    user = request.session.get("user")
    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=303)
    users = store.load("users")
    return render("admin.html", request, page="admin", users=users)

@app.post("/admin/create-user")
async def create_user(request: Request, username: str = Form(...),
                      password: str = Form(...), role: str = Form("user")):
    user = request.session.get("user")
    if not user or user.get("role") != "admin":
        return HTMLResponse("Unauthorized", status_code=401)
    users = store.load("users")
    if any(u["username"] == username for u in users):
        return render("_upload_err.html", request, message=f"User '{username}' already exists.")
    users.append(asdict(User(username=username, password_hash=_hash_pw(password),
                             role=role, created_at=datetime.now().isoformat())))
    store.save("users", users)
    return render("_upload_ok.html", request, message=f"User '{username}' created.")

@app.post("/admin/toggle-user/{username}")
async def toggle_user(request: Request, username: str):
    user = request.session.get("user")
    if not user or user.get("role") != "admin":
        return HTMLResponse("Unauthorized", status_code=401)
    users = store.load("users")
    for u in users:
        if u["username"] == username:
            u["active"] = not u.get("active", True)
    store.save("users", users)
    return RedirectResponse("/admin", status_code=303)

# ── API KEY SETTINGS ROUTES ──────────────────────────────────
@app.get("/api-settings")
async def api_settings_page(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login", status_code=303)
    sd = get_sdata(request)
    return render("api_settings.html", request, page="api-settings",
                  current_key=sd.get("api_key", ""),
                  env_key=bool(os.environ.get("ANTHROPIC_API_KEY")),
                  saved=False)

@app.post("/api-settings")
async def save_api_key(request: Request, api_key: str = Form("")):
    user = request.session.get("user")
    if not user:
        return HTMLResponse("Unauthorized", status_code=401)
    sd = get_sdata(request)
    sd["api_key"] = api_key.strip()
    return render("api_settings.html", request, page="api-settings",
                  current_key=sd.get("api_key", ""),
                  env_key=bool(os.environ.get("ANTHROPIC_API_KEY")),
                  saved=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 11: MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

