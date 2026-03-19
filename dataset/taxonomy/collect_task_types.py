#!/usr/bin/env python3
"""
Collect and analyze instruction types across instruction-tuning datasets/benchmarks.

Key design choices:
  - UNIFIED taxonomy: tasks ("summarize this") and constraints ("use ≤200 words")
    are both "instructions" — they differ only in *level* (content_task, format,
    style, content_constraint, meta).
  - DYNAMIC enrichment: the taxonomy starts with seed entries but grows as each
    dataset is processed.  New instruction types discovered from explicit dataset
    categories or frequent imperative n-grams are added on-the-fly.
  - NO data saved to disk (streaming everywhere).
"""

import json
import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output" / "taxonomy"
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  UNIFIED INSTRUCTION TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════

# Every entry lives in exactly ONE level:
#   content_task       – what to produce  (summarize, translate, …)
#   format             – structural rules (word count, JSON, bullets, …)
#   style              – surface style    (uppercase, language, tone, …)
#   content_constraint – inclusion/exclusion of specific content
#   meta               – meta-instructions (repeat prompt, give 2 answers, …)

LEVELS = ("task_type", "format_constraint", "content_style_constraint", "process_directive")


@dataclass
class InstructionType:
    """One node in the taxonomy."""
    name: str
    level: str                          # one of LEVELS
    patterns: list[re.Pattern] = field(default_factory=list)
    discovered_from: set[str] = field(default_factory=set)  # dataset names
    examples: list[str] = field(default_factory=list)       # short example snippets
    description: str = ""

    def matches(self, text: str) -> bool:
        return any(p.search(text) for p in self.patterns)


class Taxonomy:
    """
    A living taxonomy that maps instruction names → InstructionType objects.
    It starts with seed entries and can be enriched at runtime.
    """

    def __init__(self) -> None:
        self._entries: dict[str, InstructionType] = {}
        self._init_seed()

    # ----- read helpers -----
    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> InstructionType:
        return self._entries[name]

    def __iter__(self):
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def by_level(self, level: str) -> list[InstructionType]:
        return [e for e in self._entries.values() if e.level == level]

    # ----- mutation -----
    def add(self, entry: InstructionType) -> None:
        self._entries[entry.name] = entry

    def get_or_create(self, name: str, level: str, **kwargs) -> InstructionType:
        if name not in self._entries:
            self._entries[name] = InstructionType(name=name, level=level, **kwargs)
        return self._entries[name]

    # ----- detection -----
    def detect(self, text: str) -> dict[str, bool]:
        """Return {name: True/False} for every entry in the taxonomy."""
        return {e.name: e.matches(text) for e in self._entries.values()}

    # ----- serialisation -----
    def to_dict(self) -> dict:
        out = {}
        for e in self._entries.values():
            out[e.name] = {
                "level": e.level,
                "description": e.description,
                "discovered_from": sorted(e.discovered_from),
                "n_patterns": len(e.patterns),
                "examples": e.examples[:3],
            }
        return out

    def matches(self, text: str, label: str) -> bool:
        """Check if text matches patterns for a given taxonomy label."""
        if label not in self._entries:
            return False
        return self._entries[label].matches(text)

    @classmethod
    def from_dict(cls, d: dict) -> "Taxonomy":
        """Reconstruct a Taxonomy with seed patterns from a serialised taxonomy dict.

        Builds a fresh seeded Taxonomy, then prunes to only entries whose names
        appear (at any nesting depth) in *d* (e.g. the taxonomy.json structure).
        Entries not in *d* are dropped so callers get a clean label set.
        """
        # Flatten all string keys from nested dict (skip metadata keys like _meta)
        def _collect_labels(obj: dict, out: set[str]) -> None:
            for k, v in obj.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, dict):
                    level_val = v.get("level")
                    if isinstance(level_val, str):
                        out.add(k)
                    else:
                        _collect_labels(v, out)

        flat_labels: set[str] = set()
        _collect_labels(d, flat_labels)

        tax = cls()  # initialises with updated seed patterns
        # Prune entries not present in the serialised dict
        for name in list(tax._entries.keys()):
            if name not in flat_labels:
                del tax._entries[name]
        return tax

    # ----- seed taxonomy -----
    def _init_seed(self) -> None:
        """Populate with well-known instruction types from the new taxonomy."""

        def _add(name, level, patterns_raw, desc=""):
            pats = [re.compile(p, re.I) for p in patterns_raw]
            self.add(InstructionType(name=name, level=level, patterns=pats,
                                     description=desc))

        # ── task_type: information tasks ──────────────────────────────
        _add("question_answering", "task_type", [
            r"\b(answer the (?:following )?question|what is|who is|when did|where is|how (?:many|much|did|does|do|can)|which of|yes or no|true or false|is it true that|does .+ (?:have|contain|exist)|choose the (?:correct|best)|select (?:the|one)|which of the following|options?:\s*\(?[A-D]|\([A-D]\)\s|explain why|describe how|what (?:are|were) the (?:reasons|causes|effects)|based on the (?:passage|text|context|paragraph)|according to|read the following|given the (?:passage|text|context))\b",
        ], "Answer questions in any form (open, yes/no, multiple-choice, reading comprehension)")

        _add("fact_verification", "task_type", [
            r"\b(verify|fact.?check|is it (?:true|correct) that|confirm whether)\b",
        ], "Verify factual claims")

        _add("information_extraction", "task_type", [
            r"\b(extract|identify .+ (?:entities|names|dates|numbers)|named entity|NER|find .+ in the (?:text|passage))\b",
        ], "Extract structured info from unstructured text")

        _add("summarization", "task_type", [
            r"\b(summarize|summary|summarise|write a (?:brief |short )?summary|provide a summary|tldr|tl;dr)\b",
        ], "Condense text into a summary")

        # ── task_type: reasoning tasks ─────────────────────────────────
        _add("mathematical_reasoning", "task_type", [
            r"\b(solve|calculate|compute|math|equation|algebra|arithmetic|(\d+\s*[\+\-\*\/]\s*\d+))\b",
        ], "Mathematical problem solving")

        _add("logical_deductive_reasoning", "task_type", [
            r"\b(logic|deduc|induc|if .+ then|syllogism|premise|conclusion|infer)\b",
        ], "Logical or deductive reasoning")

        _add("commonsense_reasoning", "task_type", [
            r"\b(common sense|commonsense|most likely|plausible|what would happen if)\b",
        ], "Commonsense inference")

        _add("argumentation", "task_type", [
            r"\b(argue|argument|debate|persuade|justify|defend|refute|pro(?:s)? and con(?:s)?|counter.?argument)\b",
        ], "Construct, evaluate, or refute arguments")

        _add("prediction", "task_type", [
            r"\b(predict|forecast|estimate .+ future|will happen|likely to|projec(?:t|tion)|anticipate)\b",
        ], "Forecast future states or outcomes")

        # ── task_type: generative tasks ────────────────────────────────
        _add("creative_writing", "task_type", [
            r"\b(write a (?:story|poem|essay|letter|song|script|novel|blog)|compose|creative writing|fiction)\b",
        ], "Produce creative or literary text")

        _add("text_completion", "task_type", [
            r"\b(complete the|fill in|continue the|finish the (?:sentence|paragraph|story))\b",
        ], "Complete or continue text")

        _add("dialogue_generation", "task_type", [
            r"\b(conversation|dialogue|chat)\b",
        ], "Generate conversation turns or a full dialogue")

        _add("translation", "task_type", [
            r"\b(translate|translation|convert .+ to .+ language|in (?:French|Spanish|German|Chinese|Japanese|Korean|Arabic|Hindi|Russian))\b",
        ], "Translate between languages")

        _add("rewriting_paraphrasing", "task_type", [
            r"\b(rewrite|paraphrase|rephrase|reword|simplify the (?:following|text|sentence))\b",
        ], "Rephrase or simplify existing text")

        _add("communication_writing", "task_type", [
            r"\b(write (?:(?:an?|the)\s+)?(?:\w+\s+){0,3}(?:email|e-mail|letter|memo|message|subject line)|compose (?:(?:an?|the)\s+)?(?:\w+\s+){0,2}(?:email|message))\b",
        ], "Compose functional communication artifacts")

        # ── task_type: structured output tasks ─────────────────────────
        _add("classification", "task_type", [
            r"\b(classify|categorize|categorise|label the|assign .+ to .+ categor|is this .+ positive or negative|sentiment|positive or negative|opinion .+ about)\b",
        ], "Assign labels or categories; includes sentiment analysis")

        _add("ranking_comparison", "task_type", [
            r"\b(rank|compare|contrast|differentiate|distinguish|order .+ by|rate .+ from|which is (?:better|worse|best|worst))\b",
        ], "Order, rank, or compare multiple items")

        _add("data_analysis", "task_type", [
            r"\b(analyz|data analysis|interpret .+ data|statistics|chart|graph|table|CSV)\b",
        ], "Analyze or interpret data")

        _add("code_generation", "task_type", [
            r"\b(write (?:a |the )?(?:code|program|function|script|class)|implement|coding|python|javascript|def |class |```)\b",
        ], "Write or complete code")

        _add("conversion", "task_type", [
            r"\b(convert|transform .+ into|change .+ to|turn .+ into|from .+ to (?:another|a different))\b",
        ], "Convert values, units, formats, or representations")

        # ── task_type: action/planning tasks ───────────────────────────
        _add("planning", "task_type", [
            r"\b(plan|schedule|organize|itinerary|step.?by.?step (?:plan|guide)|roadmap)\b",
        ], "Create a plan or schedule")

        _add("brainstorming", "task_type", [
            r"\b(brainstorm|generate (?:a list|ideas|suggestions|examples)|list (?:\d+ )?(?:ways|ideas|suggestions|examples|reasons))\b",
        ], "Generate a list of ideas or options")

        _add("role_playing", "task_type", [
            r"\b(role.?play|pretend you are|act as|imagine you are|you are a)\b",
        ], "Adopt a persona")

        _add("explanation", "task_type", [
            r"\b(explain|explanation|describe .+ works|elaborate|clarify|what does .+ mean)\b",
        ], "Explain a concept or process")

        # ── format_constraint ─────────────────────────────────────────
        _add("length_constraint", "format_constraint", [
            r"\b(\d+ (?:words?|sentences?|paragraphs?|characters?)|word (?:count|limit)|at (?:least|most) \d+ (?:words?|sentences?|paragraphs?)|between \d+ and \d+ (?:words?|sentences?|paragraphs?))\b",
            r"\b(at\s+(?:least|most)|(?:no|not)\s+(?:more|fewer|less)\s+than|between\s+\d+\s+and|exactly|up\s+to|minimum|maximum|limit\s+(?:to|of))\s*\d+\s*(words?|sentences?|paragraphs?|characters?)\b",
        ], "Constrain output length by words/sentences/paragraphs/characters")

        _add("structure_constraint", "format_constraint", [
            r"\b(bullet (?:point|list)|numbered list|\d+ bullets?|use bullets|include a title|add a title|with (?:a |the )?title|title:|section|multiple sections|divided into|P\.?S\.?|postscript)\b",
        ], "Require specific structural elements (sections, bullets, title, postscript)")

        _add("output_syntax_format", "format_constraint", [
            r"\b(JSON|json format|output .+ json|respond .+ json|XML|xml format|HTML|html format|in (?:JSON|XML|HTML|CSV|YAML|markdown)|respond .+ (?:JSON|XML|markdown))\b",
        ], "Require machine-readable or markup syntax")

        _add("response_count", "format_constraint", [
            r"\b(two (?:responses|answers|versions)|give .+ (?:two|2) (?:different )?(?:responses|answers)|multiple (?:responses|answers|versions))\b",
            r"\b(?:give|provide|write|generate|produce|offer)\s+(?:me\s+)?(?:\d+|two|three|four|five|multiple|several)\s+(?:different\s+)?(?:answers?|responses?|versions?)\b",
        ], "Specify how many distinct responses to provide")

        _add("number_placeholder", "format_constraint", [
            r"\b(placeholder|\[.+\])\b",
        ], "Include a specified number of placeholder tokens")

        # ── content_style_constraint ──────────────────────────────────
        _add("keyword_inclusion", "content_style_constraint", [
            r"\b(include the (?:word|keyword|term)|must (?:contain|include|use) the (?:word|keyword))\b",
        ], "Must include specific keywords")

        _add("keyword_frequency", "content_style_constraint", [
            r"\b(use .+ (?:at least|exactly|no more than) \d+ times?)\b",
        ], "Use a keyword a specific number of times")

        _add("forbidden_words", "content_style_constraint", [
            r"\b(do not (?:use|include|mention)|avoid (?:using|the word)|forbidden word|without (?:using|the word)|no commas|without commas|do not use commas)\b",
        ], "Must not include specific words or punctuation")

        _add("letter_frequency", "content_style_constraint", [
            r"\b(letter .+ appear .+ times|frequency of .+ letter)\b",
        ], "Control frequency of specific letters")

        _add("quotation_constraint", "content_style_constraint", [
            r"\b(quotation|quote|wrap .+ in quotes|use .+ quotes)\b",
        ], "Wrap response or specific parts in quotation marks")

        _add("highlighted_sections", "content_style_constraint", [
            r"\b(highlight|bold|italic|underline|emphasis)\b",
        ], "Bold, italicize, or otherwise highlight specific parts")

        _add("response_language", "content_style_constraint", [
            r"\b(respond in|answer in|write (?:in|your .+ in) (?:English|French|Spanish|German|Chinese|Japanese))\b",
            r"\b(?:respond|answer|reply|write|output)\s+(?:only\s+)?(?:in|using)\s+(?:English|French|Spanish|German|Italian|Portuguese|Chinese|Japanese|Korean|Arabic|Hindi|Russian)\b",
        ], "Write the response in a specified natural language")

        _add("casing_constraint", "content_style_constraint", [
            r"\b(all (?:in )?uppercase|ALL CAPS|capitalize all|entirely uppercase|all (?:in )?lowercase|entirely lowercase|no capital|title case|camel case|snake case)\b",
        ], "Apply a specific casing rule to the response")

        _add("tone_constraint", "content_style_constraint", [
            r"\b(?:(?:in|use|with)\s+(?:a\s+)?(?:formal|informal|casual|academic|professional|friendly|humorous|sarcastic|polite|neutral|persuasive|conversational|technical)\s+(?:tone|style|register|voice|manner))\b",
        ], "Write in a specified tone or register")

        _add("audience_constraint", "content_style_constraint", [
            r"\b(?:(?:as\s+if|like)\s+(?:you\s+(?:are|were)\s+)?(?:speaking|talking|explaining|writing)\s+to\s+|for\s+(?:a\s+)?(?:child|kid|\d+[- ]?year[- ]?old|beginner|expert|layperson|general\s+audience|student|teenager|professional))\b",
        ], "Target the response toward a specified audience")

        _add("perspective_constraint", "content_style_constraint", [
            r"\b(?:(?:in|use|write\s+in)\s+(?:the\s+)?(?:first|second|third)\s+person)\b",
        ], "Write from a specified narrative or grammatical perspective")

        _add("topic_scope", "content_style_constraint", [
            r"\b(?:(?:only|exclusively|solely|strictly)\s+(?:about|on|regarding|concerning|related\s+to)|(?:focus|concentrate|stick)\s+(?:on|to)|(?:limit|restrict|confine)\s+(?:your(?:self)?\s+)?(?:to|the\s+(?:scope|topic)))\b",
        ], "Restrict or focus content to a specific topic or domain")

        _add("end_with", "content_style_constraint", [
            r"\b(end (?:with|your .+ with)|last (?:word|sentence|line))\b",
        ], "End the response with a specific word, phrase, or sentence")

        # ── process_directive ─────────────────────────────────────────
        _add("chain_of_thought", "process_directive", [
            r"\b(?:(?:think|reason|work)\s+(?:(?:it\s+)?(?:out\s+)?)?step[- ]by[- ]step|show\s+(?:your\s+)?(?:work(?:ing)?|reasoning|thought\s+process|steps)|chain[- ]?of[- ]?thought|let'?s\s+think)\b",
        ], "Show reasoning steps explicitly before giving the final answer")

        _add("repeat_prompt", "process_directive", [
            r"\b(repeat .+ prompt|restate .+ question|echo .+ input|first repeat the request)\b",
        ], "Repeat the original prompt verbatim")

        _add("self_evaluation", "process_directive", [
            r"\b(?:rate\s+(?:your\s+)?(?:confidence|certainty|answer|response)|how\s+(?:sure|confident|certain)\s+(?:are\s+you)|on\s+a\s+scale\s+of\s+\d+\s+to\s+\d+)\b",
        ], "Rate or score the generated response")

        _add("conditional_execution", "process_directive", [
            r"\b(?:if .+ then .+ (?:else|otherwise)|depending on|based on whether)\b",
        ], "Follow different instructions depending on a condition")

        _add("meta_directive", "process_directive", [
            r"\b(follow(?:ing)? (?:the )?instructions?|must (?:include|contain|use|end|start|be)|format your|your response (?:should|must)|answer only|do not say)\b",
        ], "Explicit high-level instruction about how to interpret other instructions")


# ═══════════════════════════════════════════════════════════════════════════
# 2.  DYNAMIC TAXONOMY ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════

# Map known explicit category labels → existing taxonomy names + level
CATEGORY_LABEL_MAP: dict[str, tuple[str, str]] = {
    # DollyV2
    "brainstorming":            ("brainstorming", "task_type"),
    "classification":           ("classification", "task_type"),
    "closed_qa":                ("question_answering", "task_type"),
    "creative_writing":         ("creative_writing", "task_type"),
    "general_qa":               ("question_answering", "task_type"),
    "information_extraction":   ("information_extraction", "task_type"),
    "open_qa":                  ("question_answering", "task_type"),
    "summarization":            ("summarization", "task_type"),
    # FLAN / BIGBench common keys
    "natural_questions":        ("question_answering", "task_type"),
    "trivia_qa":                ("question_answering", "task_type"),
    "bool_q":                   ("question_answering", "task_type"),
    "sentiment":                ("classification", "task_type"),
    "translation":              ("translation", "task_type"),
    "paraphrase":               ("rewriting_paraphrasing", "task_type"),
    "nli":                      ("logical_deductive_reasoning", "task_type"),
    "commonsense":              ("commonsense_reasoning", "task_type"),
    "math":                     ("mathematical_reasoning", "task_type"),
    "code":                     ("code_generation", "task_type"),
    "dialogue":                 ("dialogue_generation", "task_type"),
    # KCIF operation labels → taxonomy entries
    "capitalize":               ("casing_constraint", "content_style_constraint"),
    "alt_case":                 ("casing_constraint", "content_style_constraint"),
    "reverse_correct_answer":   ("rewriting_paraphrasing", "task_type"),
    "reverse_correct_answer_alt_case": ("casing_constraint", "content_style_constraint"),
    "correct_answer_words":     ("output_syntax_format", "format_constraint"),
    "numformat_numeric":        ("output_syntax_format", "format_constraint"),
    "correct_answer_append":    ("keyword_inclusion", "content_style_constraint"),
    "increment_correct_answer_by_one": ("output_syntax_format", "format_constraint"),
    "increment_incorrect_answers_by_one": ("output_syntax_format", "format_constraint"),
    "sort_incorrect":           ("output_syntax_format", "format_constraint"),
    "sort_options_to_string":   ("output_syntax_format", "format_constraint"),
    "options_to_string":        ("output_syntax_format", "format_constraint"),
    "incorrect_options_to_string": ("output_syntax_format", "format_constraint"),
    "print_label":              ("output_syntax_format", "format_constraint"),
    "correct_answer_text":      ("question_answering", "task_type"),
}

# ─── Verb canonicalization ──────────────────────────────────────────────
# Synonymous imperative verbs grouped under canonical forms.  During
# enrichment the *object noun* (not the verb) determines the task, so
# "write a summary" and "produce a summary" both resolve to "summarization".

_VERB_GROUPS: dict[str, list[str]] = {
    "create":    ["write", "compose", "draft", "produce", "generate", "build",
                  "construct", "develop", "craft", "make", "prepare",
                  "formulate", "create", "design", "devise", "author"],
    "explain":   ["explain", "describe", "clarify", "elaborate", "illustrate",
                  "elucidate", "discuss"],
    "summarize": ["summarize", "summarise", "condense", "recap", "abridge",
                  "shorten"],
    "analyze":   ["analyze", "analyse", "examine", "evaluate", "assess",
                  "review", "inspect", "study", "investigate", "critique",
                  "criticize"],
    "compare":   ["compare", "contrast", "differentiate", "distinguish"],
    "extract":   ["extract", "identify", "find", "locate", "detect",
                  "retrieve"],
    "edit":      ["edit", "revise", "rewrite", "rephrase", "paraphrase",
                  "reword", "modify", "correct", "fix", "proofread",
                  "improve", "refine", "polish"],
    "list":      ["list", "enumerate", "name", "outline", "itemize",
                  "brainstorm"],
    "suggest":   ["suggest", "recommend", "propose", "advise"],
    "answer":    ["answer", "respond", "reply", "address"],
    "classify":  ["classify", "categorize", "categorise", "label", "tag",
                  "sort", "group"],
    "translate": ["translate", "render", "localize"],
    "predict":   ["predict", "forecast", "estimate", "speculate", "anticipate"],
    "argue":     ["argue", "justify", "debate", "defend"],
    "rank":      ["rank", "order", "prioritize", "rate", "score"],
    "define":    ["define", "interpret"],
    "convert":   ["convert", "transform", "change", "turn"],
    "solve":     ["solve", "calculate", "compute"],
    "plan":      ["plan", "schedule", "organize", "arrange"],
}

VERB_CANON: dict[str, str] = {}
for _canon, _variants in _VERB_GROUPS.items():
    for _v in _variants:
        VERB_CANON[_v] = _canon

# ─── Object nouns → seed taxonomy entries ───────────────────────────────
# Maps common task-related nouns to existing taxonomy entry names.  When an
# instruction says "write a <noun>", this ensures different verbs for the
# same task converge to a single taxonomy entry.

TASK_OBJECT_MAP: dict[str, str] = {
    # summarization
    "summary": "summarization", "summaries": "summarization",
    "synopsis": "summarization", "abstract": "summarization",
    "overview": "summarization", "gist": "summarization",
    # creative writing
    "story": "creative_writing", "stories": "creative_writing",
    "poem": "creative_writing", "poems": "creative_writing",
    "essay": "creative_writing", "essays": "creative_writing",
    "letter": "creative_writing", "song": "creative_writing",
    "narrative": "creative_writing", "novel": "creative_writing",
    "blog": "creative_writing",
    # code generation
    "code": "code_generation", "program": "code_generation",
    "function": "code_generation", "algorithm": "code_generation",
    "implementation": "code_generation",
    # translation
    "translation": "translation", "translations": "translation",
    # classification
    "classification": "classification",
    # explanation
    "explanation": "explanation", "explanations": "explanation",
    "definition": "explanation", "definitions": "explanation",
    "meaning": "explanation",
    # question answering
    "question": "question_answering", "questions": "question_answering",
    "answer": "question_answering", "answers": "question_answering",
    # brainstorming
    "ideas": "brainstorming", "suggestions": "brainstorming",
    "examples": "brainstorming", "reasons": "brainstorming",
    "ways": "brainstorming", "tips": "brainstorming",
    "options": "brainstorming", "alternatives": "brainstorming",
    "list": "brainstorming", "lists": "brainstorming",
    # planning
    "plan": "planning", "schedule": "planning",
    "itinerary": "planning", "roadmap": "planning",
    # data analysis
    "analysis": "data_analysis", "report": "data_analysis",
    "chart": "data_analysis", "graph": "data_analysis",
    # dialogue
    "dialogue": "dialogue_generation",
    "conversation": "dialogue_generation",
    "chat": "dialogue_generation",
    # paraphrasing
    "paraphrase": "rewriting_paraphrasing",
    # sentiment
    "sentiment": "classification", "opinion": "classification",
    # information extraction
    "entities": "information_extraction",
    "names": "information_extraction",
    # math
    "equation": "mathematical_reasoning", "calculation": "mathematical_reasoning",
    # ranking
    "comparison": "ranking_comparison",
    "ranking": "ranking_comparison",
    # fact verification
    "fact": "fact_verification", "claim": "fact_verification",
    # text completion
    "sentence": "text_completion", "paragraph": "text_completion",
}

# ─── Canonical verb → generalised task (intransitive fallback) ──────────
# When a canonical verb is used without a recognised object noun, fall back
# to this mapping to find the correct generalised task.

CANONICAL_VERB_TO_TASK: dict[str, str | None] = {
    "summarize": "summarization",
    "classify":  "classification",
    "translate": "translation",
    "compare":   "ranking_comparison",
    "analyze":   "data_analysis",
    "explain":   "explanation",
    "plan":      "planning",
    "solve":     "mathematical_reasoning",
    "extract":   "information_extraction",
    "edit":      "rewriting_paraphrasing",
    "rank":      "ranking_comparison",
    "argue":     "argumentation",
    "predict":   "prediction",
    "define":    "explanation",
    "suggest":   "brainstorming",
    "list":      "brainstorming",
    "answer":    "question_answering",
    "convert":   "conversion",
    "create":    None,   # too generic without an object
}

# ─── Compiled verb-phrase regex ─────────────────────────────────────────
# Matches any known verb (from VERB_CANON) + up to 5 following words,
# anywhere in the text.  More robust than anchoring on sentence starts
# since instructions often embed verbs after context phrases:
#   "Given the passage, summarize the key points."

_all_verbs_sorted = sorted(VERB_CANON.keys(), key=len, reverse=True)
_verb_alt = "|".join(re.escape(v) for v in _all_verbs_sorted)
VERB_PHRASE_RE = re.compile(rf"\b({_verb_alt})\s+((?:\S+\s*){{1,5}})", re.I)

# Words to skip when searching for the object noun after a verb
_ARTICLES = frozenset({
    "a", "an", "the", "this", "that", "those", "these",
    "some", "your", "my", "his", "her", "its", "our", "their",
    "each", "every", "all", "any",
})
_SKIP_ADJECTIVES = frozenset({
    "short", "brief", "detailed", "long", "new", "simple", "good", "quick",
    "possible", "different", "specific", "main", "important", "best",
    "following", "given", "own", "various", "relevant", "common", "clear",
    "complete", "full", "comprehensive", "small", "large", "original",
    "first", "second", "similar", "unique", "creative", "appropriate",
    "correct", "proper", "formal", "informal", "concise", "basic",
    "above", "below", "certain", "particular", "potential", "key",
})

# Generic nouns that should not become their own taxonomy entries
_GENERIC_OBJECTS = frozenset({
    # meta / abstract
    "text", "data", "document", "documents", "input", "output",
    "result", "results", "response", "information", "content",
    "thing", "things", "stuff", "item", "items", "piece", "pieces",
    "part", "parts", "way", "method", "approach", "process",
    "form", "type", "kind", "version", "copy", "task", "passage",
    "sample", "example", "step", "steps", "word", "words",
    "point", "points", "section", "sections", "line", "lines",
    "set", "sets", "model", "system", "value", "values",
    "number", "numbers", "case", "sense", "sure", "order",
    # common passage / content words that leak through
    "game", "team", "season", "year", "years", "time", "times",
    "people", "man", "woman", "men", "women", "children", "child",
    "city", "state", "country", "area", "place", "home", "house",
    "school", "company", "group", "family", "world", "life",
    "day", "days", "week", "month", "end", "side", "hand",
    "work", "war", "death", "body", "head", "power", "money",
    "bill", "act", "law", "lead", "down", "back", "name",
    "later", "eventually", "estimated", "comeback", "universal",
    "funny", "sure", "changes",
    # sports teams / proper nouns that shouldn't be tasks
    "lions", "steelers", "dodgers", "bears", "eagles", "giants",
    "cowboys", "packers", "broncos", "ravens", "patriots",
    "celtics", "lakers", "yankees", "warriors", "red", "sox",
})

# Minimum count of unmatched (verb, object) pairs for Phase 3 object
# clustering to promote a new entry.  Set higher than MIN_NEW_IMPERATIVE_COUNT
# because passages contain many spurious verb+noun co-occurrences.
MIN_OBJECT_CLUSTER_COUNT = 50

# Threshold: an instruction type is "present" in a dataset if ≥ this fraction
PRESENCE_THRESHOLD = 0.03

# Minimum count of a NEW imperative to promote it into the taxonomy
MIN_NEW_IMPERATIVE_COUNT = 15

# Stop words — common English words that are NOT instruction verbs
_STOPWORDS = frozenset({
    # determiners / pronouns / prepositions / conjunctions / adverbs
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "he", "she", "his", "her", "they", "them", "their", "we", "our",
    "you", "your", "my", "me", "i", "who", "whom", "which", "what",
    "where", "when", "why", "how", "there", "here", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "same", "so", "than", "too", "very", "just",
    "also", "now", "then", "once", "many", "much", "any", "own",
    # prepositions
    "in", "on", "at", "to", "for", "with", "by", "from", "up", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further",
    "against", "until", "while", "of", "since", "unlike", "although",
    "because", "besides", "despite", "among", "along", "across",
    # auxiliaries / modals / copula
    "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "can", "could", "must",
    # conjunctions
    "and", "but", "or", "nor", "if", "as", "yet",
    # common non-imperative sentence starters
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "according", "based", "given", "please", "note", "remember",
    "one", "two", "three", "four", "five", "first", "second", "third",
    # common nouns / proper nouns that frequently start sentences
    "people", "john", "liang", "come", "time", "way", "day", "year",
    "world", "life", "part", "work", "place", "case", "group", "number",
    # misc non-verbs
    "new", "old", "good", "well", "even", "back", "still", "already",
    "often", "usually", "sometimes", "never", "always",
})


def enrich_from_explicit_categories(
    taxonomy: Taxonomy,
    category_counts: Counter,
    dataset_name: str,
) -> list[str]:
    """
    Map a dataset's explicit category labels into the taxonomy.
    Creates new entries for unmapped labels.
    Returns list of taxonomy names that were activated.
    """
    activated: list[str] = []
    for label, count in category_counts.items():
        label_lower = label.lower().strip()

        # Check if any known mapping applies
        mapped = False
        for key, (tax_name, tax_level) in CATEGORY_LABEL_MAP.items():
            if key in label_lower:
                entry = taxonomy.get_or_create(tax_name, tax_level)
                entry.discovered_from.add(dataset_name)
                activated.append(tax_name)
                mapped = True

        # If no mapping found, create a new taxonomy entry from the label
        if not mapped and count >= 5:
            slug = re.sub(r"[^a-z0-9]+", "_", label_lower).strip("_")
            if slug and slug not in taxonomy:
                entry = InstructionType(
                    name=slug,
                    level="content_task",  # default: assume it's a task
                    description=f"Discovered from '{label}' in {dataset_name}",
                    discovered_from={dataset_name},
                    patterns=[re.compile(re.escape(label_lower), re.I)],
                )
                taxonomy.add(entry)
                logger.info(f"    [ENRICH] New taxonomy entry from category: '{slug}' (level=content_task)")
                activated.append(slug)

    return activated


def enrich_from_instructions(
    taxonomy: Taxonomy,
    instruction_texts: list[str],
    dataset_name: str,
) -> list[str]:
    """
    Mine instruction texts for task patterns and map them to *generalised*
    taxonomy entries.  Uses verb canonicalization + object-noun mapping so
    that ``write a summary``, ``produce a summary``, ``generate a summary``
    all converge to the single *summarization* entry.

    Three-phase pipeline:
      1. Extract (canonical_verb, object_noun) pairs via regex.
      2. Resolve each pair to a taxonomy task (object map → verb fallback
         → pattern match).  Attribute hits to existing entries or create
         generalised new ones.
      3. Cluster remaining unmatched pairs by object noun and promote
         frequent novel objects to new taxonomy entries.

    Returns list of newly created taxonomy entry names.
    """
    # ── Phase 1: extract (canonical_verb, object_noun) pairs ─────────
    task_counter: Counter = Counter()    # resolved_task_name → count
    unmatched: Counter = Counter()       # (canon_verb, obj_noun) → count

    for text in instruction_texts:
        for m in VERB_PHRASE_RE.finditer(text):
            raw_verb = m.group(1).lower()
            canon = VERB_CANON.get(raw_verb)
            if canon is None:
                continue

            # Parse trailing words to find the first significant noun
            following = m.group(2).lower().split()
            obj_noun: str | None = None
            for w in following:
                w_clean = re.sub(r"[^a-z]", "", w)
                if not w_clean or len(w_clean) < 3:
                    continue
                if w_clean in _ARTICLES or w_clean in _STOPWORDS:
                    continue
                if w_clean in _SKIP_ADJECTIVES:
                    continue
                obj_noun = w_clean
                break

            # ── Resolve to a taxonomy task ───────────────────────────
            task_name: str | None = None

            # Strategy 1: object noun → TASK_OBJECT_MAP
            if obj_noun and obj_noun in TASK_OBJECT_MAP:
                task_name = TASK_OBJECT_MAP[obj_noun]

            # Strategy 2: canonical verb → CANONICAL_VERB_TO_TASK
            if task_name is None and canon in CANONICAL_VERB_TO_TASK:
                task_name = CANONICAL_VERB_TO_TASK[canon]

            # Strategy 3: test if phrase matches any existing entry
            if task_name is None:
                snippet = f"{raw_verb} {obj_noun}" if obj_noun else raw_verb
                for entry in taxonomy:
                    if entry.matches(snippet):
                        task_name = entry.name
                        break

            if task_name:
                task_counter[task_name] += 1
            elif obj_noun:
                unmatched[(canon, obj_noun)] += 1

    # ── Phase 2: attribute resolved tasks to taxonomy ────────────────
    new_entries: list[str] = []

    for task_name, cnt in task_counter.most_common():
        if cnt < MIN_NEW_IMPERATIVE_COUNT:
            continue
        if task_name in taxonomy:
            # Existing entry — just record provenance
            taxonomy[task_name].discovered_from.add(dataset_name)
        else:
            # Task referenced by CANONICAL_VERB_TO_TASK but not yet seeded
            # → create a generalised entry with patterns for all verb synonyms
            canon_verb = None
            for cv, tn in CANONICAL_VERB_TO_TASK.items():
                if tn == task_name:
                    canon_verb = cv
                    break

            verb_alts: list[str] = []
            if canon_verb and canon_verb in _VERB_GROUPS:
                verb_alts = _VERB_GROUPS[canon_verb]
            alt_pattern = (
                "|".join(re.escape(v) for v in verb_alts)
                if verb_alts
                else re.escape(task_name)
            )
            task_word = re.escape(task_name.replace("_", " "))

            entry = InstructionType(
                name=task_name,
                level="content_task",
                description=(
                    f"Generalised task discovered in {dataset_name} (count={cnt})"
                ),
                discovered_from={dataset_name},
                patterns=[re.compile(rf"\b({alt_pattern}|{task_word})\b", re.I)],
            )
            taxonomy.add(entry)
            new_entries.append(task_name)
            logger.info(
                f"    [ENRICH] New generalised task: '{task_name}' (count={cnt})"
            )

    # ── Phase 3: cluster unmatched (verb, object) pairs by object ────
    # If many different verbs target the same novel object noun, create
    # a new generalised entry named after the object (not the verb).
    obj_agg: Counter = Counter()
    obj_verbs: dict[str, set[str]] = defaultdict(set)
    for (cv, obj), cnt in unmatched.items():
        obj_agg[obj] += cnt
        obj_verbs[obj].add(cv)

    for obj, cnt in obj_agg.most_common(20):
        if cnt < MIN_OBJECT_CLUSTER_COUNT:
            break

        # Skip generic / uninformative nouns
        if obj in _GENERIC_OBJECTS:
            continue

        # Skip if any existing entry already matches this noun
        already = False
        for entry in taxonomy:
            if entry.matches(obj):
                entry.discovered_from.add(dataset_name)
                already = True
                break
        if already:
            continue

        slug = re.sub(r"[^a-z0-9]+", "_", obj).strip("_")
        if not slug or slug in taxonomy:
            continue

        verbs_str = ", ".join(sorted(obj_verbs[obj]))
        entry = InstructionType(
            name=slug,
            level="content_task",
            description=(
                f"Task involving '{obj}' (verbs: {verbs_str}), "
                f"discovered in {dataset_name} (count={cnt})"
            ),
            discovered_from={dataset_name},
            patterns=[re.compile(rf"\b{re.escape(obj)}\b", re.I)],
        )
        taxonomy.add(entry)
        new_entries.append(slug)
        logger.info(
            f"    [ENRICH] New task from object clustering: '{slug}' "
            f"(verbs: {verbs_str}, count={cnt})"
        )

    return new_entries


# ─── Constraint templates ────────────────────────────────────────────────
# Each template is (regex, level, name_template).
#   - regex: pattern to search for in instruction text
#   - level: which taxonomy level to assign
#   - name_template: callable(match) → taxonomy entry slug, or a fixed string
#
# Templates are ordered from most specific to most general.  When a match is
# found the corresponding taxonomy entry is created/attributed.

_CONSTRAINT_TEMPLATES: list[tuple[re.Pattern, str, str | None]] = [
    # ── format_constraint: length constraints ("at most 200 words", "between 3 and 5 paragraphs") ──
    (re.compile(
        r"\b(?:at\s+(?:least|most)|(?:no|not)\s+(?:more|fewer|less)\s+than|between\s+\d+\s+and|exactly|up\s+to|minimum|maximum|limit\s+(?:to|of))\s*\d+\s*(words?|sentences?|paragraphs?|characters?|lines?|tokens?|pages?|syllables?|bullet\s*points?|items?|points?)",
        re.I,
    ), "format_constraint", "length_constraint"),

    # ── format_constraint: output structure ("in JSON", "as a table", "as a numbered list") ──
    (re.compile(
        r"\b(?:(?:respond|answer|output|write|format|present|return|give|provide)\s+(?:it\s+|your\s+(?:answer|response|output)\s+)?(?:in|as)\s+(?:a\s+)?)(JSON|XML|CSV|YAML|markdown|HTML|table|bullet(?:ed)?\s+(?:list|points?)|numbered\s+list|ordered\s+list|unordered\s+list|dictionary|array|code\s+block)",
        re.I,
    ), "format_constraint", None),  # slug derived from matched group

    # ── format_constraint: structural markers ("include a title", "add a header", "start with") ──
    (re.compile(
        r"\b(?:include|add|insert|begin\s+with|start\s+with|end\s+with|contain)\s+(?:a\s+)?(?:title|header|heading|footer|introduction|conclusion|preamble|postscript|P\.?S\.?|table\s+of\s+contents|appendix|references?\s+section|bibliography)",
        re.I,
    ), "format_constraint", "structure_constraint"),

    # ── content_style_constraint: casing ("all uppercase", "in lowercase", "title case") ──
    (re.compile(
        r"\b(?:all\s+(?:in\s+)?(?:uppercase|lowercase|caps|capitals)|entirely\s+(?:in\s+)?(?:uppercase|lowercase)|(?:title|camel|snake|kebab)\s+case|ALL\s+CAPS|no\s+capital(?:s|ization)?)",
        re.I,
    ), "content_style_constraint", "casing_constraint"),

    # ── content_style_constraint: tone / register ("formal tone", "in a casual style", "as if speaking to a child") ──
    (re.compile(
        r"\b(?:(?:in|use|with)\s+(?:a\s+)?(?:formal|informal|casual|academic|professional|friendly|humorous|sarcastic|polite|neutral|objective|persuasive|authoritative|conversational|technical|simple|eloquent)\s+(?:tone|style|register|voice|manner|language))",
        re.I,
    ), "content_style_constraint", "tone_constraint"),

    # ── content_style_constraint: person / perspective ("in first person", "use third person") ──
    (re.compile(
        r"\b(?:(?:in|use|write\s+in)\s+(?:the\s+)?(?:first|second|third)\s+person)",
        re.I,
    ), "content_style_constraint", "perspective_constraint"),

    # ── content_style_constraint: audience targeting ("for a 5-year-old", "for beginners") ──
    (re.compile(
        r"\b(?:(?:as\s+if|like)\s+(?:you\s+(?:are|were)\s+)?(?:speaking|talking|explaining|writing)\s+to\s+|for\s+(?:a\s+)?(?:child|kid|\d+[- ]?year[- ]?old|beginner|expert|layperson|non[- ]?expert|general\s+audience|technical\s+audience|scientist|student|teenager|professional))",
        re.I,
    ), "content_style_constraint", "audience_constraint"),

    # ── content_style_constraint: language of response ("respond in French", "answer in Spanish") ──
    (re.compile(
        r"\b(?:respond|answer|reply|write|output)\s+(?:only\s+)?(?:in|using)\s+(?:English|French|Spanish|German|Italian|Portuguese|Chinese|Japanese|Korean|Arabic|Hindi|Russian|Dutch|Swedish|Polish|Turkish|Hebrew|Greek|Thai|Vietnamese|Indonesian|Malay|Czech|Romanian|Hungarian|Finnish|Danish|Norwegian|Ukrainian|Bengali|Tamil|Urdu|Swahili)",
        re.I,
    ), "content_style_constraint", "response_language"),

    # ── content_style_constraint: keyword inclusion ("must include the word", "use the phrase") ──
    (re.compile(
        r"\b(?:(?:must|should|need\s+to|have\s+to|make\s+sure\s+to|be\s+sure\s+to|ensure\s+(?:you|to))\s+(?:include|contain|use|mention|incorporate|reference|cite|feature))\s+(?:the\s+)?(?:(?:word|keyword|term|phrase|name|expression|string)s?\s+)?[\"\']?",
        re.I,
    ), "content_style_constraint", "keyword_inclusion"),

    # ── content_style_constraint: keyword frequency ("use ... exactly N times") ──
    (re.compile(
        r"\b(?:use|mention|include|repeat|say)\s+.{1,40}?(?:at\s+least|at\s+most|exactly|no\s+more\s+than|no\s+fewer\s+than)\s+\d+\s+times?",
        re.I,
    ), "content_style_constraint", "keyword_frequency"),

    # ── content_style_constraint: forbidden content ("do not use", "avoid", "without", "never mention") ──
    (re.compile(
        r"\b(?:(?:do\s+not|don'?t|never|avoid|refrain\s+from|without)\s+(?:us(?:e|ing)|includ(?:e|ing)|mention(?:ing)?|refer(?:ring)?\s+to|writ(?:e|ing)|say(?:ing)?)|(?:no\s+(?:use\s+of|mention\s+of|references?\s+to)))",
        re.I,
    ), "content_style_constraint", "forbidden_words"),

    # ── content_style_constraint: specific exclusion ("no commas", "without numbers") ──
    (re.compile(
        r"\b(?:(?:no|without|do\s+not\s+use|avoid(?:\s+using)?)\s+(?:commas?|periods?|semicolons?|colons?|exclamation\s+marks?|question\s+marks?|numbers?|digits?|abbreviations?|acronyms?|jargon|slang|contractions?|passive\s+voice|rhetorical\s+questions?))",
        re.I,
    ), "content_style_constraint", "forbidden_words"),

    # ── content_style_constraint: topic scoping ("only about", "focus on", "stick to") ──
    (re.compile(
        r"\b(?:(?:only|exclusively|solely|strictly)\s+(?:about|on|regarding|concerning|related\s+to|discussing|covering)|(?:focus|concentrate|stick)\s+(?:on|to)|(?:limit|restrict|confine)\s+(?:your(?:self)?\s+)?(?:to|the\s+(?:scope|topic|subject)))",
        re.I,
    ), "content_style_constraint", "topic_scope"),

    # ── process_directive: repeat / echo prompt ──
    (re.compile(
        r"\b(?:repeat|restate|echo|copy|reproduce)\s+(?:the\s+)?(?:prompt|question|instruction|task|query|input|request|original)",
        re.I,
    ), "process_directive", "repeat_prompt"),

    # ── process_directive: multiple answers ("give 2 answers", "provide N different responses") ──
    (re.compile(
        r"\b(?:(?:give|provide|write|generate|produce|offer)\s+(?:me\s+)?(?:\d+|two|three|four|five|multiple|several|various)\s+(?:different\s+)?(?:answers?|responses?|versions?|variations?|alternatives?|options?|solutions?|attempts?))",
        re.I,
    ), "process_directive", "response_count"),

    # ── process_directive: step-by-step reasoning ("think step by step", "show your reasoning") ──
    (re.compile(
        r"\b(?:(?:think|reason|work)\s+(?:(?:it\s+)?(?:out\s+)?)?step[- ]by[- ]step|show\s+(?:your\s+)?(?:work(?:ing)?|reasoning|thought\s+process|steps)|chain[- ]?of[- ]?thought|let'?s\s+think)",
        re.I,
    ), "process_directive", "chain_of_thought"),

    # ── process_directive: self-evaluation ("rate your confidence", "how sure are you") ──
    (re.compile(
        r"\b(?:rate\s+(?:your\s+)?(?:confidence|certainty)|how\s+(?:sure|confident|certain)\s+(?:are\s+you)|on\s+a\s+scale\s+of\s+\d+\s+to\s+\d+)",
        re.I,
    ), "process_directive", "self_evaluation"),
]

# Minimum number of matches for a constraint template to create / attribute
# a taxonomy entry in a given dataset.
MIN_CONSTRAINT_COUNT = 3


def enrich_constraints_from_instructions(
    taxonomy: Taxonomy,
    instruction_texts: list[str],
    dataset_name: str,
) -> list[str]:
    """
    Mine instruction texts for constraint patterns across format, style,
    content_constraint, and meta levels.  Uses structural regex templates
    rather than verb+object extraction (constraints are expressed very
    differently from content tasks).

    For each template that fires ≥ MIN_CONSTRAINT_COUNT times, either
    attributes an existing taxonomy entry or creates a new one.

    Returns list of newly created taxonomy entry names.
    """
    new_entries: list[str] = []
    template_hits: dict[int, list[str]] = defaultdict(list)  # template idx → matched snippets

    # ── Scan all texts against all templates ──────────────────────────
    for text in instruction_texts:
        for idx, (pattern, _level, _slug) in enumerate(_CONSTRAINT_TEMPLATES):
            m = pattern.search(text)
            if m:
                template_hits[idx].append(m.group(0)[:200])

    # ── Process templates that fired frequently enough ───────────────
    for idx, snippets in template_hits.items():
        if len(snippets) < MIN_CONSTRAINT_COUNT:
            continue

        pattern, level, slug_or_none = _CONSTRAINT_TEMPLATES[idx]

        # Determine the slug for this entry
        if slug_or_none is not None:
            slug = slug_or_none
        else:
            # Derive slug from the most common matched content
            token_counts: Counter = Counter()
            for s in snippets:
                # extract the most specific token (last significant word)
                words = re.findall(r"[a-z]{3,}", s.lower())
                # skip common filler words to find the format-type keyword
                for w in reversed(words):
                    if w not in _STOPWORDS and w not in _ARTICLES and w not in _SKIP_ADJECTIVES:
                        token_counts[w] += 1
                        break
            if token_counts:
                dominant = token_counts.most_common(1)[0][0]
                slug = re.sub(r"[^a-z0-9]+", "_", dominant).strip("_") + "_format"
            else:
                continue

        # Attribute to existing entry if possible
        if slug in taxonomy:
            taxonomy[slug].discovered_from.add(dataset_name)
            continue

        # Check if any existing entry at the same level already covers this
        already = False
        sample_snippet = snippets[0]
        for entry in taxonomy:
            if entry.level == level and entry.matches(sample_snippet):
                entry.discovered_from.add(dataset_name)
                already = True
                break
        if already:
            continue

        # Create new entry
        cnt = len(snippets)
        entry = InstructionType(
            name=slug,
            level=level,
            description=(
                f"Constraint discovered in {dataset_name} (count={cnt}). "
                f"Sample: '{snippets[0][:120]}'"
            ),
            discovered_from={dataset_name},
            patterns=[pattern],
            examples=snippets[:3],
        )
        taxonomy.add(entry)
        new_entries.append(slug)
        logger.info(
            f"    [ENRICH] New constraint: '{slug}' (level={level}, count={cnt})"
        )

    return new_entries


# ═══════════════════════════════════════════════════════════════════════════
# 3.  DATASET REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "IFEval": {
        "source": "github_raw",
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl",
        "format": "jsonl",
        "instruction_fields": ["prompt"],
        "sample_size": 2000,
    },
    "Self-Instruct": {
        "source": "huggingface",
        "hf_path": "yizhongw/self_instruct",
        # Loading script no longer supported → use auto-converted Parquet
        "parquet_url": "hf://datasets/yizhongw/self_instruct@refs/convert/parquet/self_instruct/train/0000.parquet",
        "hf_split": "train",
        "instruction_fields": ["prompt"],
        "sample_size": 2000,
    },
    "Alpaca": {
        "source": "github_raw",
        "url": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
        "format": "json",
        "instruction_fields": ["instruction", "input"],
        "sample_size": 2000,
    },
    "WizardLM (Evol-Instruct)": {
        "source": "huggingface",
        "hf_path": "WizardLMTeam/WizardLM_evol_instruct_V2_196k",
        "hf_split": "train",
        "instruction_fields": ["instruction", "conversations"],
        "sample_size": 2000,
    },
    "LIMA": {
        "source": "huggingface",
        "hf_path": "GAIR/lima",
        # Gated dataset: accept terms at https://huggingface.co/datasets/GAIR/lima
        # Loading script no longer supported → use auto-converted Parquet
        "parquet_url": "hf://datasets/GAIR/lima@refs/convert/parquet/plain_text/train/0000.parquet",
        "hf_split": "train",
        "instruction_fields": ["conversations"],
        "sample_size": 1500,
    },
    "DollyV2": {
        "source": "huggingface",
        "hf_path": "databricks/databricks-dolly-15k",
        "hf_split": "train",
        "instruction_fields": ["instruction", "context"],
        "category_field": "category",
        "sample_size": 2000,
    },
    "P3": {
        "source": "huggingface",
        "hf_path": "bigscience/P3",
        "hf_split": "train",
        "instruction_fields": ["inputs_pretokenized"],
        "sample_size": 2000,
        # Sample across diverse prompt templates (QA, NLI, sentiment, etc.)
        "hf_configs": [
            "adversarial_qa_dbert_answer_the_following_q",
            "super_glue_rte_does_it_follow_that",
            "imdb_Movie_Expressed_Sentiment",
            "common_gen_sentence_to_concepts",
            "ag_news_classify",
            "cnn_dailymail_3.0.0_news_summary",
            "cos_e_v1.11_question_option_description_text",
            "wiki_qa_Is_This_True_",
        ],
    },
    "FollowBench": {
        "source": "huggingface",
        "hf_path": "YuxinJiang/FollowBench",
        "hf_split": "train",
        "instruction_fields": ["prompt", "instruction"],
        "sample_size": 2000,
    },
    "Arena Hard": {
        "source": "github_raw",
        "url": "https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/resolve/main/data/arena-hard-v0.1/question.jsonl",
        "format": "jsonl",
        "instruction_fields": ["prompt"],
        "category_field": "category",
        "sample_size": 1000,
    },
    "FLAN": {
        "source": "huggingface",
        "hf_path": "Muennighoff/flan",
        "hf_split": "train",
        "instruction_fields": ["inputs"],
        "category_field": "task",
        "sample_size": 3000,
    },
    "ARC": {
        "source": "huggingface",
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Challenge",
        "hf_split": "train",
        "instruction_fields": ["question", "choices"],
        "sample_size": 2000,
    },
    "TriviaQA": {
        "source": "huggingface",
        "hf_path": "mandarjoshi/trivia_qa",
        "hf_name": "rc",
        "hf_split": "train",
        "instruction_fields": ["question"],
        "sample_size": 2000,
    },
    "SciQ": {
        "source": "huggingface",
        "hf_path": "allenai/sciq",
        "hf_split": "train",
        "instruction_fields": ["question", "support"],
        "sample_size": 2000,
    },
    "GSM8K": {
        "source": "huggingface",
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "hf_split": "train",
        "instruction_fields": ["question"],
        "sample_size": 2000,
    },
    "DROP": {
        "source": "huggingface",
        "hf_path": "ucinlp/drop",
        "hf_split": "train",
        "instruction_fields": ["question", "passage"],
        "sample_size": 2000,
    },
    "BIGBench": {
        "source": "huggingface",
        # google/bigbench uses a script → use Parquet-ready mirror instead
        "hf_path": "tasksource/bigbench",
        "hf_split": "train",
        "instruction_fields": ["inputs"],
        # BIG-Bench Lite tasks for broad coverage
        "hf_configs": [
            "emoji_movie",
            "auto_debugging",
            "causal_judgment",
            "date_understanding",
            "logical_deduction",
            "navigate",
            "snarks",
            "sports_understanding",
            "tracking_shuffled_objects",
            "hyperbaton",
            "movie_recommendation",
            "ruin_names",
        ],
        "sample_size": 3000,
    },
    "Boolq": {
        "source": "huggingface",
        "hf_path": "google/boolq",
        "hf_split": "train",
        "instruction_fields": ["question", "passage"],
        "sample_size": 2000,
    },
    "KCIFData": {
        "source": "kcif_github",
        "url": "https://github.com/IBM/KCIF",
        "instruction_fields": ["instruction_text"],
        "category_field": "operation",
        "sample_size": 5000,
    },
    "WildIFEval": {
        "source": "huggingface",
        "hf_path": "gililior/wild-if-eval",
        "hf_split": "test",
        "instruction_fields": ["decomposition"],
        "sample_size": 3000,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 4.  DATA LOADING  (streaming, no disk saves)
# ═══════════════════════════════════════════════════════════════════════════

def load_hf_dataset(config: dict, max_samples: int = 2000) -> list[dict]:
    """Stream samples from a HuggingFace dataset.

    Supports three modes:
      1. **parquet_url** – load a single Parquet file via ``hf://`` URI.
         Use for datasets whose original loading scripts are no longer
         supported but that have auto-converted Parquet refs.
      2. **hf_configs** – load a list of config names and sample across
         them (P3, BIGBench, …).
      3. **single config** – the standard ``load_dataset(path, name, ...)``.
    """
    from datasets import load_dataset

    hf_path = config["hf_path"]
    hf_name = config.get("hf_name")
    hf_configs = config.get("hf_configs")      # list of configs → sample across them
    hf_split = config.get("hf_split", "train")
    sample_size = config.get("sample_size", max_samples)
    parquet_url = config.get("parquet_url")    # direct Parquet URI (hf://…)

    # ── Parquet-URL mode (scripted datasets with auto-converted Parquet) ─
    if parquet_url:
        logger.info(f"  Streaming Parquet: {parquet_url}")
        try:
            ds = load_dataset("parquet", data_files=parquet_url,
                              split="train", streaming=True)
            samples = []
            for i, item in enumerate(ds):
                if i >= sample_size:
                    break
                samples.append(dict(item))
            return samples
        except Exception as e:
            logger.error(f"  Failed to load Parquet {parquet_url}: {e}")
            return []

    # ── multi-config mode (e.g. P3, BIGBench) ───────────────────────
    if hf_configs:
        per_config = max(sample_size // len(hf_configs), 50)
        all_samples: list[dict] = []
        for cfg_name in hf_configs:
            logger.info(f"  Streaming HF: {hf_path} (name={cfg_name}, split={hf_split})")
            try:
                ds = load_dataset(hf_path, name=cfg_name, split=hf_split,
                                  streaming=True)
                for i, item in enumerate(ds):
                    if i >= per_config:
                        break
                    all_samples.append(dict(item))
            except Exception as e:
                logger.warning(f"    Skipping config {cfg_name}: {e}")
        logger.info(f"  Loaded {len(all_samples)} samples across {len(hf_configs)} configs")
        return all_samples[:sample_size]

    # ── single-config mode ───────────────────────────────────────────
    logger.info(f"  Streaming HF: {hf_path} (name={hf_name}, split={hf_split})")
    try:
        ds = load_dataset(hf_path, name=hf_name, split=hf_split,
                          streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= sample_size:
                break
            samples.append(dict(item))
        return samples
    except Exception as e:
        logger.error(f"  Failed to load {hf_path}: {e}")
        return []


def load_github_data(config: dict) -> list[dict]:
    """Fetch data from a GitHub raw URL in-memory."""
    import random
    url = config["url"]
    fmt = config.get("format", "jsonl")

    logger.info(f"  Fetching GitHub: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    content = resp.text

    data: list[dict] = []
    if fmt == "jsonl":
        for line in content.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))
    elif fmt == "json":
        parsed = json.loads(content)
        if isinstance(parsed, list):
            data = parsed
        elif isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    data = v
                    break

    sample_size = config.get("sample_size", 2000)
    if len(data) > sample_size:
        random.seed(42)
        data = random.sample(data, sample_size)
    return data


def load_kcif_data(config: dict) -> list[dict]:
    """
    KCIF is a benchmark framework (not a raw dataset).  We fetch its
    config.json (which lists instruction names per knowledge task) and
    then fetch each instruction .py file to extract the actual instruction
    texts (stored in ``ext`` or ``instruction_text`` fields of the schema).

    Each extracted instruction text becomes one sample with keys:
        instruction_text, instruction_id, operation, dataset
    so it can be analysed by the standard pipeline.
    """
    base_raw = "https://raw.githubusercontent.com/IBM/KCIF/main/src/construct_data"
    logger.info(f"  Fetching KCIF config from {base_raw}/config.json")

    # 1. Get config.json → list of instruction names per knowledge task
    resp = requests.get(f"{base_raw}/config.json", timeout=30)
    resp.raise_for_status()
    kcif_config: dict = json.loads(resp.text)

    # Collect unique instruction names across all knowledge-task datasets
    all_instruction_names: set[str] = set()
    for task_instructions in kcif_config.values():
        all_instruction_names.update(task_instructions)

    # 2. For each instruction, fetch its .py file and extract text + operation
    samples: list[dict] = []
    schema_re = re.compile(
        r'"(?:ext|instruction_text)"\s*:\s*\[([^\]]+)\]', re.S
    )
    operation_re = re.compile(r'"operation"\s*:\s*\[\s*"([^"]+)"')

    for instr_name in sorted(all_instruction_names):
        url = f"{base_raw}/instruction/{instr_name}.py"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            source = r.text
        except Exception:
            logger.warning(f"    Could not fetch {instr_name}.py")
            continue

        # Extract instruction texts
        m = schema_re.search(source)
        if not m:
            continue

        # Parse the string list (simple heuristic, handles multi-line)
        raw_texts = re.findall(r'"([^"]{10,})"', m.group(1))

        # Extract operation label
        op_match = operation_re.search(source)
        operation = op_match.group(1).strip() if op_match else instr_name

        # Which knowledge-task datasets use this instruction?
        used_in = [ds for ds, instrs in kcif_config.items() if instr_name in instrs]

        for text in raw_texts:
            samples.append({
                "instruction_text": text,
                "instruction_id": instr_name,
                "operation": operation,
                "datasets": ", ".join(used_in),
            })

    logger.info(f"  KCIF: extracted {len(samples)} instruction-text samples "
                f"from {len(all_instruction_names)} instruction types")
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# 5.  INSTRUCTION TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_instruction_text(sample: dict, instruction_fields: list[str]) -> str:
    parts: list[str] = []
    for fld in instruction_fields:
        val = sample.get(fld)
        if val is None:
            continue
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    content = item.get("content") or item.get("value") or item.get("text", "")
                    role = item.get("role", item.get("from", ""))
                    if role in ("user", "human", ""):
                        parts.append(str(content))
                elif isinstance(item, str):
                    parts.append(item)
        elif isinstance(val, dict):
            parts.append(json.dumps(val))
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  ANALYSIS  (detect + enrich)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_dataset(
    taxonomy: Taxonomy,
    dataset_name: str,
    samples: list[dict],
    instruction_fields: list[str],
    category_field: str | None = None,
) -> dict[str, Any]:
    """
    Analyse one dataset:
      1. Enrich taxonomy from explicit category labels.
      2. Enrich taxonomy from frequent imperative verbs.
      3. Match every instruction against the (now enriched) taxonomy.
    """
    if not samples:
        return {"n_samples": 0, "error": "No samples loaded"}

    schema = list(samples[0].keys())
    n = len(samples)

    # ── extract all instruction texts ────────────────────────────────
    texts: list[str] = []
    for s in samples:
        t = extract_instruction_text(s, instruction_fields)
        if t.strip():
            texts.append(t)

    # ── explicit categories ──────────────────────────────────────────
    category_counts: Counter = Counter()
    if category_field:
        for s in samples:
            cat = s.get(category_field)
            if cat:
                category_counts[str(cat)] += 1
        enrich_from_explicit_categories(taxonomy, category_counts, dataset_name)

    # ── discover new instruction types from imperative verbs ─────────
    enrich_from_instructions(taxonomy, texts, dataset_name)

    # ── discover new constraints (format, style, content, meta) ──────
    enrich_constraints_from_instructions(taxonomy, texts, dataset_name)

    # ── count hits per taxonomy entry (uses enriched taxonomy) ───────
    counts: Counter = Counter()
    for text in texts:
        detected = taxonomy.detect(text)
        for name, hit in detected.items():
            if hit:
                counts[name] += 1

    prevalence = {name: round(cnt / n, 4) for name, cnt in counts.items()}

    # Mark which dataset contributed to which entries
    for name, prev in prevalence.items():
        if prev >= PRESENCE_THRESHOLD:
            taxonomy[name].discovered_from.add(dataset_name)

    # Collect a few example snippets for new entries
    for text in texts[:20]:
        detected = taxonomy.detect(text)
        for name, hit in detected.items():
            entry = taxonomy[name]
            if hit and len(entry.examples) < 3:
                entry.examples.append(text[:300])

    return {
        "n_samples": n,
        "schema": schema,
        "category_counts": category_counts,
        "counts": counts,
        "prevalence": prevalence,
        "sample_instructions": [t[:500] for t in texts[:5]],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def process_all_datasets(
    registry: dict | None = None,
) -> tuple[pd.DataFrame, Taxonomy]:
    """Process all datasets with a shared, growing taxonomy."""
    if registry is None:
        registry = DATASET_REGISTRY

    taxonomy = Taxonomy()
    rows: list[dict[str, Any]] = []

    for name, config in registry.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {name}")
        logger.info(f"{'='*60}")

        if config.get("source") == "skip":
            logger.info(f"  Skipping: {config.get('note', 'needs manual handling')}")
            continue

        # Load
        try:
            if config["source"] == "huggingface":
                samples = load_hf_dataset(config)
            elif config["source"] == "github_raw":
                samples = load_github_data(config)
            elif config["source"] == "kcif_github":
                samples = load_kcif_data(config)
            else:
                logger.warning(f"  Unknown source: {config['source']}")
                continue
        except Exception as e:
            logger.error(f"  Error loading {name}: {e}")
            continue

        if not samples:
            logger.warning(f"  No samples for {name}")
            continue

        # Analyze + enrich
        analysis = analyze_dataset(
            taxonomy, name, samples,
            instruction_fields=config.get("instruction_fields", []),
            category_field=config.get("category_field"),
        )

        logger.info(f"  Samples analysed: {analysis['n_samples']}")
        logger.info(f"  Schema: {analysis['schema']}")
        logger.info(f"  Taxonomy size: {len(taxonomy)} entries")
        if analysis.get("category_counts"):
            logger.info(f"  Explicit categories: {dict(analysis['category_counts'].most_common(10))}")
        top = analysis["counts"].most_common(12)
        logger.info(f"  Top hits: {dict(top)}")

        # Build row — presence booleans for every entry *currently* in taxonomy
        row: dict[str, Any] = {"dataset": name}
        for entry in taxonomy:
            prev = analysis["prevalence"].get(entry.name, 0)
            row[entry.name] = prev >= PRESENCE_THRESHOLD
        row["_n_samples"] = analysis["n_samples"]
        row["_schema"] = ", ".join(analysis["schema"])
        if analysis.get("category_counts"):
            row["_explicit_categories"] = ", ".join(
                f"{k}({v})" for k, v in analysis["category_counts"].most_common(20)
            )
        rows.append(row)

    # Backfill columns: earlier datasets may not have columns for entries
    # discovered later → fill with False
    all_cols = ["dataset"] + [e.name for e in taxonomy] + ["_n_samples", "_schema", "_explicit_categories"]
    df = pd.DataFrame(rows)
    for col in all_cols:
        if col not in df.columns:
            df[col] = False if not col.startswith("_") else ""
    df = df.fillna(False)

    return df, taxonomy


# ═══════════════════════════════════════════════════════════════════════════
# 8.  OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def save_results(df: pd.DataFrame, taxonomy: Taxonomy) -> None:
    """Write CSV + taxonomy JSON + human-readable summary."""

    # ── CSV ───────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "unified_taxonomy.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nCSV → {csv_path}")

    # ── Taxonomy JSON ─────────────────────────────────────────────────
    tax_path = OUTPUT_DIR / "taxonomy.json"
    with open(tax_path, "w") as f:
        json.dump(taxonomy.to_dict(), f, indent=2)
    logger.info(f"Taxonomy → {tax_path}")

    # ── Summary ───────────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "taxonomy_summary.txt"
    with open(summary_path, "w") as f:
        f.write("UNIFIED INSTRUCTION TAXONOMY\n")
        f.write("=" * 70 + "\n\n")

        for level in LEVELS:
            entries = taxonomy.by_level(level)
            if not entries:
                continue
            f.write(f"\n{'─'*60}\n")
            f.write(f"  {level.upper()}  ({len(entries)} entries)\n")
            f.write(f"{'─'*60}\n")
            for e in sorted(entries, key=lambda x: x.name):
                datasets = sorted(e.discovered_from) if e.discovered_from else ["(seed only)"]
                f.write(f"  {e.name:40s}  <- {', '.join(datasets)}\n")
                if e.description:
                    f.write(f"    {e.description}\n")

        f.write(f"\n\n{'='*70}\n")
        f.write("PER-DATASET BREAKDOWN\n")
        f.write("=" * 70 + "\n")
        instruction_cols = [e.name for e in taxonomy]
        for _, row in df.iterrows():
            f.write(f"\n{row['dataset']}:\n")
            present = [c for c in instruction_cols if c in df.columns and row.get(c) is True]
            by_level: dict[str, list[str]] = defaultdict(list)
            for c in present:
                by_level[taxonomy[c].level].append(c)
            for level in LEVELS:
                items = by_level.get(level, [])
                if items:
                    f.write(f"  [{level}] {', '.join(items)}\n")
            if not present:
                f.write("  (none detected)\n")

    logger.info(f"Summary → {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect instruction types across instruction-tuning datasets "
                    "with a unified, self-enriching taxonomy."
    )
    parser.add_argument("--datasets", nargs="*",
                        help="Process only these datasets (substring match)")
    parser.add_argument("--list", action="store_true",
                        help="List registered datasets and exit")
    args = parser.parse_args()

    if args.list:
        print("\nRegistered datasets:")
        for name, config in DATASET_REGISTRY.items():
            skip = " [SKIP]" if config.get("source") == "skip" else ""
            print(f"  {name}{skip} — {config.get('source', '?')}")
        return

    registry = DATASET_REGISTRY
    if args.datasets:
        filtered = {}
        for d in args.datasets:
            for k, v in registry.items():
                if d.lower() in k.lower():
                    filtered[k] = v
        if not filtered:
            logger.error("No matching datasets found")
            return
        registry = filtered

    df, taxonomy = process_all_datasets(registry=registry)

    if not df.empty:
        save_results(df, taxonomy)
        print(f"\n{'='*60}")
        print(f"Done — {len(df)} datasets, {len(taxonomy)} instruction types")
        print(f"  (seed: {sum(1 for e in taxonomy if not e.discovered_from)} | "
              f"enriched: {sum(1 for e in taxonomy if e.discovered_from)})")
        print(f"CSV:      {OUTPUT_DIR / 'unified_taxonomy.csv'}")
        print(f"Taxonomy: {OUTPUT_DIR / 'taxonomy.json'}")
        print(f"Summary:  {OUTPUT_DIR / 'taxonomy_summary.txt'}")
        print(f"{'='*60}")
    else:
        print("No datasets processed.")


if __name__ == "__main__":
    main()
