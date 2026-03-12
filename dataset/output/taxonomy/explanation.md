# A Taxonomy of Instructions for Instruction-Tuned LLMs

## Motivation and Design Principles

Instruction-tuned LLMs are typically evaluated against flat lists of "instruction types" derived inductively from existing benchmarks (IFEval, WildIFEval, Alpaca, Arena Hard, etc.). While useful as a starting point, these flat lists suffer from three recurring problems:

1. **Category conflation** — task types (what the model is asked to do cognitively) are mixed with output constraints (how the response should be shaped), making it hard to reason about either independently.
2. **Redundancy from induction** — categories proliferate because each benchmark contributes its own labels, resulting in near-duplicates like `yes_no_qa`, `open_ended_qa`, `multiple_choice_qa`, and `reading_comprehension` sitting at the same level.
3. **Flat structure where hierarchy is needed** — a single `reasoning` concept spawns three separate top-level entries; a single `length` concept spawns four.

The redesign proposed here is **compositional**: any instruction is described as a combination of a **task type** and zero or more **constraints**, drawn from three orthogonal constraint axes. This matches how users actually compose instructions and how recent benchmarks (IFEval, WildIFEval, AGENTIF) have converged on modeling the problem.

```
instruction = task_type × [format_constraints] × [content_style_constraints] × [process_directives]
```

---

## Axis 1 — Task Type

What cognitive work is being requested? Task types are organized into five clusters.

### 1.1 Information Tasks

| Task | Description | Merged from |
|---|---|---|
| `question_answering` | Answer a question from knowledge or a provided passage. Format differences (yes/no, multiple choice, open-ended) are captured via format constraints, not separate task types. | `open_ended_qa`, `yes_no_qa`, `multiple_choice_qa`, `reading_comprehension`, `question_answering` |
| `fact_verification` | Verify whether a factual claim is true or false, with justification. | — |
| `information_extraction` | Extract structured information (entities, relations, keywords) from unstructured text. | — |
| `summarization` | Condense a longer text into a shorter summary while preserving key information. | — |

### 1.2 Reasoning Tasks

| Task | Description | Merged from |
|---|---|---|
| `mathematical_reasoning` | Solve mathematical problems: arithmetic, algebra, probability, formal proofs. | `math_reasoning` |
| `logical_deductive_reasoning` | Apply formal or deductive logic to reach conclusions from premises. | `logical_reasoning` |
| `commonsense_reasoning` | Reason about everyday world knowledge and intuitive inference. | — |
| `argumentation` | Construct, evaluate, or refute arguments; includes debate and persuasion. | — |
| `prediction` | Forecast future states or outcomes based on patterns or given data. | — |

### 1.3 Generative Tasks

| Task | Description | Merged from |
|---|---|---|
| `creative_writing` | Produce original creative text: stories, poems, scripts, blog posts. | — |
| `text_completion` | Continue or complete a partial piece of text. | — |
| `dialogue_generation` | Generate conversational turns or a full dialogue. | `dialogue_conversation` |
| `translation` | Translate text from one language to another. | — |
| `rewriting_paraphrasing` | Rephrase, simplify, or stylistically transform existing text. | — |
| `communication_writing` | Compose functional artifacts: emails, subject lines, memos, letters. | `email`, `subject` |

### 1.4 Structured Output Tasks

| Task | Description | Merged from |
|---|---|---|
| `classification` | Assign labels or categories to input. Includes sentiment analysis as a subtype. | `classification`, `sentiment_analysis` |
| `ranking_comparison` | Order, rank, or compare multiple items or options. | `ranking`, `comparison` |
| `data_analysis` | Analyze or interpret structured or semi-structured data. | — |
| `code_generation` | Write, complete, debug, or explain code. | — |
| `conversion` | Convert values, units, or representations from one form to another. | — |

### 1.5 Action and Planning Tasks

| Task | Description |
|---|---|
| `planning` | Create a plan, schedule, itinerary, or step-by-step strategy. |
| `brainstorming` | Generate a diverse list of ideas, options, or possibilities. |
| `role_playing` | Adopt a specific character or persona for the response. |
| `explanation` | Explain a concept, process, or phenomenon clearly and accurately. |

---

## Axis 2 — Format Constraints

How should the output be structured or sized? These are independently composable with any task type.

### 2.1 Length Constraint

Constrain the size of the output by a unit and an operator.

- **Units:** words · sentences · paragraphs · characters
- **Operators:** exactly · at least · at most · between

> *"Write a blog post using less than 300 words."*
> *"Write an email template with at least 500 words."*
> *"Write a story of exactly 2 paragraphs."*

Merges: `number_words`, `number_sentences`, `number_paragraphs`, `length_constraint`

### 2.2 Structure Constraint

Require specific structural elements in the output.

- Sections with headings
- Bullet points or numbered lists (with exact counts)
- Title
- Postscript
- Separator strings or markdown dividers

> *"Your answer must contain exactly 3 bullet points."*
> *"Include a title wrapped in double angular brackets: `<<title>>`."*

Merges: `number_bullets`, `title_required`, `multiple_sections`, `structural_marker`, `postscript`

### 2.3 Output Syntax Format

Require output to conform to a specific machine-readable or markup format.

- **Formats:** JSON · XML · HTML · Markdown · CSV · YAML

> *"Your entire output should be a JSON block, nothing else."*
> *"Provide your response in HTML."*

Merges: `json_format`, `xml_format`, `html_format`, `specific_format`

### 2.4 Response Count

Specify how many distinct responses or alternatives to provide.

> *"Give exactly two different responses, separated by `******`."*
> *"Generate two alternative versions."*

Merges: `two_responses`, `multiple_responses`

### 2.5 Placeholder Count

Require a specified number of placeholder tokens (e.g. `[name]`, `[address]`) in the output.

> *"Include at least 12 placeholders represented by square brackets."*

---

## Axis 3 — Content and Style Constraints

These constrain what words appear or how the text is written. Split into lexical constraints (mechanically verifiable) and style constraints (requiring semantic judgment).

### 3.1 Lexical Constraints

| Constraint | Description |
|---|---|
| `keyword_inclusion` | The response must include one or more specified keywords. |
| `keyword_frequency` | A specified keyword must appear exactly N times. |
| `forbidden_words` | The response must not contain specific words, punctuation, or character classes (e.g. no commas, no numbers). |
| `letter_frequency` | Control the frequency of specific letters in the output. |
| `quotation_constraint` | Wrap the response or parts of it in quotation marks, or include a relevant direct quote. |
| `highlighted_sections` | Bold, italicize, or capitalize specific parts of the text. |

Merges: `forbidden_words` + `no_commas` + `punctuation_or_token_exclusion` + `forbidden_content` → `forbidden_words`

### 3.2 Style Constraints

| Constraint | Description | Dimensions |
|---|---|---|
| `response_language` | Write the response in a specified natural language. | language code |
| `casing_constraint` | Apply a casing rule to the entire response. | `all_lowercase` · `all_uppercase` · `title_case` · `sentence_case` |
| `tone_constraint` | Write in a specified tone or register. | formal · informal · humorous · persuasive · Shakespearean · … |
| `audience_constraint` | Target a specified audience type. | teenager · professional · expert · student · … |
| `perspective_constraint` | Write from a specified narrative perspective. | first person · third person |
| `topic_scope` | Restrict content to a specific topic or domain. | — |
| `end_with` | End the response with a specific word, phrase, or sentence. | — |

Merges: `all_uppercase` + `all_lowercase` + `casing_constraint` → `casing_constraint`

---

## Axis 4 — Process Directives

Meta-instructions about *how* the model should reason or structure its response process, rather than what the output should contain.

| Directive | Description |
|---|---|
| `chain_of_thought` | Show reasoning steps explicitly before giving the final answer. |
| `repeat_prompt` | Reproduce the original prompt verbatim before answering. |
| `self_evaluation` | Rate or score the generated response as part of the output. |
| `conditional_execution` | Follow different instructions depending on a condition in the input. |
| `meta_directive` | Explicit high-level instruction about how to interpret or prioritize other instructions. Formerly called `instruction_following` — renamed to avoid circularity. |

---

## Merge and Removal Summary

### Merges

| Original entries | Merged into | Rationale |
|---|---|---|
| `yes_no_qa`, `open_ended_qa`, `multiple_choice_qa`, `reading_comprehension`, `question_answering` | `question_answering` | Same cognitive task; format differences belong in Axis 2 |
| `logical_reasoning`, `commonsense_reasoning`, `math_reasoning` | Three subtypes under **Reasoning Tasks** | Different subcategories of reasoning, not independent top-level tasks |
| `classification` + `sentiment_analysis` | `classification` (with subtype) | Sentiment analysis is a specific instantiation of classification |
| `ranking` + `comparison` | `ranking_comparison` | Closely related operations |
| `email` + `subject` | `communication_writing` | Dataset-specific labels; belong in a functional writing category |
| `number_words` + `number_sentences` + `number_paragraphs` + `length_constraint` | `length_constraint` | Same concept, different units |
| `number_bullets` + `title_required` + `multiple_sections` + `structural_marker` + `postscript` | `structure_constraint` | All constrain output structure |
| `json_format` + `xml_format` + `html_format` + `specific_format` | `output_syntax_format` | Same concept, different format values |
| `two_responses` + `multiple_responses` | `response_count` | Same concept, different counts |
| `all_uppercase` + `all_lowercase` + `casing_constraint` | `casing_constraint` | Same concept, different values |
| `forbidden_words` + `no_commas` + `punctuation_or_token_exclusion` + `forbidden_content` | `forbidden_words` | All prohibit specific tokens |
| `instruction_following` | `meta_directive` | Renamed — "instruction following" is circular as a category label |

### Removed

| Entry | Reason |
|---|---|
| `arena_hard_v0_1` | Dataset identifier, not a task type |
| `aeslc_10templates` | Dataset identifier, not a task type |

---

## Key Design Decisions

**Why compositional?** Any instruction in the wild combines a task with multiple constraints. Encoding them as independent axes allows precise description of complex instructions (e.g. "write a formal email in JSON format, under 200 words, without using commas") without creating a combinatorially explosive flat list.

**Why separate process directives from format constraints?** Format constraints describe the *output artifact*; process directives describe the *generation process*. A chain-of-thought directive changes how the model should reason, not just what the output looks like. This distinction is consequential for evaluation: format constraints can often be checked mechanically, while process directives require inspecting reasoning traces.

**Why keep `fact_verification` separate from `yes_no_qa`?** Although both produce a binary judgment, fact verification carries an implicit requirement for sourced justification and is typically evaluated differently. Collapsing them obscures an important distinction for benchmark design.