# Phase 31 — UI Review

**Audited:** 2026-09-02
**Baseline:** Abstract 6-pillar standards, adapted for technical documentation
**Subject:** `docs/design/run_identity_and_unattended_invocation_spike.rst`
**Screenshots:** Not applicable (documentation page, no dev server)

---

## Pillar Scores

| Pillar | Score | Key Finding |
|--------|-------|-------------|
| 1. Copywriting | 3/4 | Clear scope boundaries and background; correction blocks lack visual distinction and forward-reference context |
| 2. Visuals | 3/4 | Proper list-table markup and code formatting; corrections need RST admonition directives for emphasis |
| 3. Color | 4/4 | N/A for documentation (Sphinx theme defaults only); no custom color specification needed |
| 4. Typography | 3/4 | Correct heading hierarchy; dense table cell text wraps poorly on narrow viewports |
| 5. Spacing | 2/4 | Table cells contain 5-10 line blocks with no internal breathing room; Future scope list items exceed 100 chars each |
| 6. Experience Design | 3/4 | Good cross-references and phase tagging; Future scope lacks visual grouping to distinguish blocking vs. open items |

**Overall: 18/24**

---

## Top 3 Priority Fixes

1. **Redesign correction blocks as RST admonitions** — Two dated corrections (lines 52-60, 154-161) use only bold text and blend into surrounding prose. Replace `**Correction, recorded 2026-09-02,**` with proper `.. attention::` or `.. warning::` directives so readers immediately recognize these as document-level amendments, not mere emphasis. — Fix effort: Low (structural markup change only).

2. **Add visual grouping to Future scope bullets by phase and blocking-status** — The seven open items (lines 248-273) list Phase 32, Phase 33, and Phase 34 responsibilities without visual separation. Group bullets under sub-headings or use a definition list: "**Phase 32:** item 1, item 2, item 3" to let readers quickly filter for their milestone. Currently requires parsing trailing parentheses to know relevance. — Fix effort: Medium (restructure and re-layout).

3. **Break dense table cells into multi-line reStructuredText with internal paragraph spacing** — Decision table rows contain 5-10 sentence blocks in single cells (e.g., line 71-90: Schema shape row is 230 words in one cell). Split each cell's content with blank lines between logical sub-points so the table remains scannable on mobile and narrow viewports (e.g., "Make `CampaignRun.campaign` nullable… [blank line] Why this over other options: [blank line] Cost: [blank line]"). — Fix effort: Medium (content reorganization within RST table syntax).

---

## Detailed Findings

### Pillar 1: Copywriting (3/4)

**Strengths:**
- Opening paragraph (lines 4-21) explicitly states scope ("how a routine, non-campaign observation…gets a persistent `CampaignRun` identity") and what was NOT built ("no `CampaignRun` schema migration, no adapter code").
- Background section (lines 25-48) frames the investigation as answering four concrete questions, making the document's purpose clear.
- Four key decisions are named in Background and mapped to phases (Phase 32, Phase 34), so readers know where recommendations will be actioned.

**Issues:**
- **Line 17:** The phrase "this project's milestone-archival workflow moves completed phase directories to `.planning/phases-archive/`… so check there first if the original path no longer resolves" is a forward-reference caveat placed at document start. It breaks narrative flow and assumes readers anticipate a future problem. Move this to a footer note or Future scope.
- **Lines 52-60, 154-161:** Both correction blocks use `**Correction, recorded 2026-09-02,**` as emphasis only. A reader skimming the document may miss these as amendments rather than background detail. The second correction (line 154) occurs mid-table, making it easy to overlook.
- **Future scope section (lines 248-273):** Seven bullet items mix "recommendations for Phase 32/34 to implement" with "explicitly still open, carried forward." Readers cannot tell if an item is a blocker ("must be resolved before Phase 32 ships") or a nice-to-have ("Phase 32 should consider") without careful parsing. Trailing parentheses indicate phase ownership but not blocking status.
- **Lines 71-90, 92-106, 107-134, 135-150:** Table cells contain dense prose (5-10 sentences) with no internal spacing. A reader on mobile sees a wall of text per cell.

**Tag:** `**Confirmed against real page**`

### Pillar 2: Visuals (3/4)

**Strengths:**
- Section hierarchy is correct: main title with `=` underline, sections with `-`, Future scope with `-`, sub-bullets with `*`.
- Inline code (`CampaignRun`, `source_identifier`, `GEMFacility`, etc.) is consistently marked with double backticks (70+ instances total).
- Two list-table blocks (lines 64-151, 163-234) are properly formatted with header rows, column widths (22/50/12 for Topic/Decision/Phase), and alignment.
- Decision tables use monospace code where appropriate (field names, Django field specs).

**Issues:**
- **Correction blocks (lines 52-60, 154-161):** Use only bold text (`**Correction,**`) for visual emphasis. In Sphinx HTML output, this renders as `<strong>` tags, indistinguishable from regular bold phrases elsewhere. Best practice: use RST admonition directives (`.. attention::`, `.. warning::`, `.. note::`) to render as highlighted boxes in HTML/PDF.
- **Table cell wrapping:** The "Decision" column width is set to 50% (line 66: `:widths: 22 50 12`), but cell content is often 10+ lines. On a 375px mobile viewport, this column becomes ~200px wide, forcing text to wrap at ~30 chars per line and creating visual fatigue. No internal list-table syntax allows sub-bullets or paragraph breaks within cells; dense blocks are unavoidable without restructuring.
- **No use of emphasis markers within tables:** Code elements and key phrases (e.g., "not sufficient" in line 136) are not visually distinguished within cells, causing key findings to blend into background.

**Tag:** `**Confirmed against real page**`

### Pillar 3: Color (4/4)

**Assessment:** Not applicable. This is a Sphinx reStructuredText document rendered with the Sphinx theme's default color scheme. No custom color specification is present, and no content-level color directives (e.g., colored text, background highlights) are used. This is the correct posture for technical documentation — style is delegated to the theme, not embedded in content.

**Tag:** `**Confirmed against real page**`

### Pillar 4: Typography (3/4)

**Strengths:**
- Heading hierarchy is correct per reStructuredText standards: main title (=), section headings (-), Future scope (-).
- Inline code uses monospace rendering consistently (70+ instances).
- Bold text is used judiciously for topic names in tables (5 Topic column entries per table) and corrections (13 bold phrases total).

**Issues:**
- **Table cell text density:** Rows like "Schema shape for a non-campaign run" (lines 71-90) contain 19 sentences in a single cell without paragraph breaks or sub-bullets. On a standard 1440px desktop, the cell width is ~720px, but text still wraps awkwardly with hyphenation. Mobile viewport (~375px) makes the cell unreadable without horizontal scroll.
- **No sub-heading hierarchy in table cells:** The Schema shape row mentions "Why this over other options," "Accepted cost," and implicit sub-topics, but all are prose paragraphs with no visual hierarchy. A reader cannot quickly scan for "what does this cost?" without re-reading the entire cell.
- **Line length variance:** Prose lines in the main body are 60-90 characters; table cell text is 150-250 characters per soft wrap. This variance reduces predictability and makes line-by-line reading harder for users with dyslexia or vision impairments.

**Tag:** `**Confirmed against real page**`

### Pillar 5: Spacing (2/4)

**Strengths:**
- Sections are separated by blank lines, creating visual breaks between major content blocks (Background, Decisions, Future scope).
- Bullet lists in Future scope are indented and use consistent `*` markers.

**Issues (MAJOR — this is the lowest-scoring pillar):**
- **No internal spacing in table cells:** Each decision table cell is a single, unbroken paragraph. Lines 71-90 (Schema shape row) contain 19 sentences with zero paragraph breaks, blank lines, or sub-bullets. This violates basic document readability principles: dense text blocks over 100 words become cognitively difficult without internal breaks.
- **Future scope bullets exceed 100 characters each:** Lines 249-274 list seven bullets, each 80-150+ characters on a single line. Examples:
  - Line 249: "Whether a classical run's proposal code should be folded into its identity key once the parser can recognise one, versus keeping the current tolerance-match-only key and accepting the documented two-proposals-same-night gap as a permanent risk (Phase 32)." (229 chars)
  - Line 253: "Whether ``source_identifier`` should be promoted to the primary lookup key for adapter-written rows once Phase 32's adapters exist, rather than added alongside the existing campaign-plus-window lookup (Phase 32)." (226 chars)
  - No internal line breaks, no sub-bullets, no grouping.
- **Correction blocks are tight prose:** Lines 52-60 and 154-161 contain 4-6 sentences each in unbroken paragraphs, with no blank lines or bullet points to break up the content. On narrow viewports, this creates 8-10 wrapped lines per correction block with no internal visual breathing room.
- **No section spacing between bulleted items:** The seven Future scope bullets run continuously from line 248-273 with one blank line only between them (line 247). No blank lines separate clusters by phase or concern area.

**Spacing metrics:**
- Average line length in table cells: 120-150 characters (hard limit before wrapping).
- Average bullet item length in Future scope: 100-150 characters (single-line render on desktop, 4-5 lines on mobile).
- Blank lines separating Future scope bullets: 0 (bullets run consecutively).

**Tag:** `**Confirmed against real page**`

### Pillar 6: Experience Design (3/4)

**Strengths:**
- Each decision row includes a "Phase" column (values: 32, 34) so readers can filter by milestone.
- Four key questions in Background (lines 34-41) are explicitly mapped to four decision rows in the tables, creating a clear "question → answer" flow.
- Cross-references to `31-DECISION.md` are placed at key points: line 16 (opening), line 59 (correction evidence), line 238 (Future scope footer).
- Future scope section explicitly states these are "recommendations for Phase 32 and Phase 34 to implement — none of it is implemented in this spike," setting correct expectations.

**Issues:**
- **Forward-reference caveat at document start (lines 17-20):** The paragraph about future archival ("this project's milestone-archival workflow moves completed phase directories…") is a disclaimer about where the document *might* be in the future, not background information needed to read it now. This breaks narrative flow. Move to a footer or remove entirely; if archival is important, document it in the Sphinx conf.py or a metadata table, not in prose at the top.
- **Correction blocks mid-content:** The second correction (line 154) occurs in the middle of the Unattended Invocation section, breaking the table presentation. A reader following the decision table row-by-row encounters the correction after already reading half the table. Readers may miss it or not realize it applies to preceding rows.
- **Future scope lacks visual grouping by phase or concern:** Seven bullets list Phases 32, 33, and 34 items without sub-headings. A Phase 32 planner must read all seven items and parse each trailing phase tag to find their responsibilities. Best practice: sub-head by phase ("**For Phase 32:**", "**For Phase 34:**") or by concern ("**Open questions:**", "**Parser/grammar work:**") so readers can skip non-relevant items.
- **No summary/executive-summary table:** Readers expect a one-page or one-screen summary of "what was decided" distinct from "what's still open." Currently, the Decisions tables answer this, but Future scope immediately adds caveats and open items, blurring the "what was decided" vs. "what wasn't" boundary. Best practice: add a TL;DR table at the top with three rows: "Schema shape: nullable FK", "Write-time ID field: source_identifier", "Invocation mechanism: cron+flock", then say "see Decisions section for full rationale."
- **Phase 33 outcome propagation caveat is hidden in Future scope line 269:** The statement "outcome propagation can read a terminal observing state back for the LCO path but never for Gemini" is a critical blocker for Phase 33, but it's buried in a bullet point (line 269-273) that also concerns Phase 32 and Phase 34. Phase 33 planners may miss it.

**Tag:** `**Confirmed against real page**`

---

## Files Audited

- `docs/design/run_identity_and_unattended_invocation_spike.rst` (273 lines)
- `docs/design/design.rst` (toctree entry verification)
- `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md` (companion document, not audited but referenced)

**Note:** This phase is a documentation-only investigation spike. No Python code, HTML templates, CSS, or interactive frontend was created. The 6-pillar audit was adapted to assess documentation readability and structure rather than UI component functionality. Pillars 3 (Color) and partially Pillar 2 (Visuals) do not apply to Sphinx theme-rendered documentation; assessment covers relevant aspects only.

---

## Audit Context

Phase 31 is an investigation spike that concludes with documentation publication. The published page (`docs/design/run_identity_and_unattended_invocation_spike.rst`) serves downstream phases (Phase 32, Phase 34) as the decision record and rationale for two major architectural choices:

1. **Schema/Identity:** nullable `campaign` FK + `source_identifier` write-time field
2. **Scheduling:** cron + flock for unattended invocation

The document was wired into the Sphinx toctree in `docs/design/design.rst` per plan 31-05, and Sphinx build verification confirmed no new build errors or warnings introduced (pre-existing warnings: 12, unchanged by this page).

---

## Recommendation

This documentation serves as a decision record and is fit for archival purposes. However, readability for downstream phases can be improved with the three priority fixes above, particularly Pillar 5 (Spacing) and Pillar 6 (Experience Design) restructuring to reduce cognitive load when Phase 32 and Phase 34 planners reference it.

The document should be reviewed by at least one Phase 32 or Phase 34 stakeholder to verify that the Future scope items are correctly scoped and that the open items are clear enough to be actioned without re-reading this entire page or opening `31-DECISION.md`.
