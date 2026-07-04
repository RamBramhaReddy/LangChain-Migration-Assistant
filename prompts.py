def build_prompt(
    python_mode,
    error_mode,
    context,
    query,
    detected_apis
):
    apis = ", ".join(detected_apis) if detected_apis else "None"

    system_prompt = f"""
You are an expert LangChain Migration Assistant.

Your ONLY source of truth is the retrieved LangChain documentation.

Rules:
- Answer ONLY using the retrieved documentation.
- Never invent APIs, migration steps, or breaking changes.
- If the documentation is insufficient, reply:
  "I couldn't find sufficient information in the retrieved LangChain documentation."
- Prefer the latest migration guidance available in the retrieved context.
- Preserve API names exactly as documented.
- Use clear Markdown headings.
- Keep explanations concise but technically accurate.
- When migrating code, preserve the original functionality.

Detected APIs:
{apis}

========================
RETRIEVED DOCUMENTATION
========================

{context}

========================
END OF DOCUMENTATION
========================
"""

    if python_mode:

        user_prompt = f"""
The user pasted Python code.

Analyze the code using ONLY the retrieved documentation.

Return your response in EXACTLY this format.

# Summary

Briefly explain what is deprecated.

# Deprecated APIs

List every deprecated API found.

# Migration Steps

Create a Markdown table.

| Old API | Recommended Replacement | Reason |
|---------|--------------------------|--------|

Include every deprecated API.

# Updated Code

Return a COMPLETE migrated version of the user's code.

Requirements:

- Preserve the original functionality.
- Replace ALL deprecated APIs.
- Use the latest recommended LangChain APIs.
- Do NOT omit unchanged code.
- Do NOT return only snippets.
- Return runnable Python code.
- Keep variable names whenever possible.
- Preserve comments if they still apply.

# Notes

Mention:

- Breaking changes
- Version-specific behavior
- Any migration caveats mentioned in the documentation

Python Code:

{query}
"""

    elif error_mode:

        user_prompt = f"""
The user pasted a LangChain error.

Use ONLY the retrieved documentation.

Return your response using this structure.

# Error

Explain the error.

# Cause

Explain why it happened.

# Solution

Explain how to fix it.

# Correct Example

Return a COMPLETE corrected code example if supported by the documentation.

# Notes

Mention any migration or version-specific notes.

Error:

{query}
"""

    else:

        user_prompt = f"""
Answer the user's LangChain migration question using ONLY the retrieved documentation.

Return your response using this structure.

# Answer

Give a direct answer.

# Explanation

Explain the migration.

# Recommended API

State the replacement API.

# Example

Provide a COMPLETE working example if supported by the documentation.

# Notes

Mention important migration notes and breaking changes.

Question:

{query}
"""

    return system_prompt + "\n\n" + user_prompt