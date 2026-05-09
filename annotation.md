# The Annotation Process

Since the original dataset has no types, the documents were annotated based on IBM Writing Style and heuristic domain knowledge. This page introduces the annotation guidelines and the whole annotation process.

## Annotation Criteria

Each page is labeled by the type of information a user would **primarily** use it for.

### `conceptual` — understand or look up information

Typical pages:
- overviews, introductions, explanations, definitions, and architecture pages
- feature and capability descriptions
- parameter, configuration, API, syntax, limit, field, and supported-value pages
- glossary, reference, list, table, and lookup-heavy pages
- hub or navigation pages that mostly link to other documentation pages

### `how-to` — perform an action or fix a problem

Typical pages:
- step-by-step instructions
- procedures with prerequisites and ordered actions
- pages with commands, code examples, or workflow sequences
- task pages: creating, configuring, deploying, connecting, importing, or managing something
- troubleshooting pages where the user intent is to resolve an error or failure

### Tie-breaking rules

- **Action verb in the title is not sufficient.** A gerund title like "Deploying..." is still `conceptual` if the page only describes a capability without providing actionable steps.
- **"You can…" statements are weak evidence.** If the page describes what is possible but provides no steps, commands, or workflow, label it `conceptual`.
- **A short conceptual introduction does not override a procedural body.** If the main body is procedural, the page is `how-to`.
- **Troubleshooting pages are `how-to`**, even if they look structurally similar to reference pages, when the primary user goal is to fix something.

## Step 1 - Heuristic pre-annotation

A script (`build_annotation_candidates.py`) generated candidate labels from
title and body signals before human review. Each candidate was assigned a
`label_source` confidence tier:

| Tier | Meaning |
|---|---|
| `title_high` | Title matched strong documentation-type patterns |
| `body_medium` | Title was inconclusive; body signals suggested a class |
| `default_low` | Weak body signal; defaulted to `conceptual` |

The candidate CSV was used to prioritize human checking, not to produce the
final dataset directly.

## Step 2 - Human review and filtering

Annotators reviewed candidates using three markers in the working CSV:

| Marker | Interpretation |
|---|---|
| *(empty)* | Accept the heuristic label |
| `?` | Flip to the other class |
| `X` | Remove — incomplete, low-quality, or too ambiguous to judge |

The subset tagged `how-to + body_medium` (333 rows) was excluded entirely
because it had not been manually verified and was expected to contain many
borderline pages where weak procedural signals could be misleading.

After all exclusions:

| Outcome | Count |
|---|---|
| Excluded (`how-to + body_medium`, unchecked) | 333 |
| Removed (`X`) | 17 |
| Usable — `conceptual` | 563 |
| Usable — `how-to` | 231 |

## Step 3 - Balancing

Because usable `conceptual` examples (563) far outnumbered usable `how-to`
examples (231), 200 examples were randomly sampled from each class (seed 42)
to produce the final balanced 400-example dataset.