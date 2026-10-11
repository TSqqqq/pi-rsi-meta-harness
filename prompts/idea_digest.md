You distil a research note into a checklist of testable mechanisms for an automated research loop.

Read the research note and the codebase map. Extract the distinct mechanisms the note proposes (not
background, not citations). For each, say what it claims and how it could be realized in THIS codebase with
a small edit; if it needs components the codebase lacks, give the closest faithful proxy that exists here.

Return STRICT JSON only:
{
  "mechanisms": [
    {"id": "M1", "name": "short name", "claim": "what should improve and why",
     "code_mapping": "file/function and the concrete change in this codebase"}
  ]
}
Return 4-10 mechanisms, most central first.
