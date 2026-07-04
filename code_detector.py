import ast
import re

COMMON_WORDS = {
    "what", "how", "why", "when", "where", "which", "who",
    "the", "this", "that", "these", "those",
    "replace", "replaced", "replacing",
    "migrate", "migration", "migrating",
    "use", "using", "used"
}

ERROR_KEYWORDS = [
    "ImportError",
    "ModuleNotFoundError",
    "AttributeError",
    "TypeError",
    "ValueError",
    "NameError",
    "No module named",
    "cannot import",
    "deprecated"
]


def extract_version(text):

    versions = [
        "v0.2",
        "v0.3",
        "v1.0",
        "latest"
    ]

    text = text.lower()

    for version in versions:

        if version.lower() in text:
            return version

    return None


def is_python_code(text):

    try:
        ast.parse(text)
        return True

    except SyntaxError:
        return False


def is_error_query(text):

    text = text.lower()

    return any(
        keyword.lower() in text
        for keyword in ERROR_KEYWORDS
    )


def extract_identifiers(code):

    identifiers = set()

    try:
        tree = ast.parse(code)

    except SyntaxError:
        return []

    for node in ast.walk(tree):

        if isinstance(node, ast.Import):

            for alias in node.names:

                identifiers.add(alias.name)
                identifiers.add(alias.name.split(".")[-1])

        elif isinstance(node, ast.ImportFrom):

            if node.module:

                identifiers.add(node.module)
                identifiers.add(node.module.split(".")[-1])

            for alias in node.names:

                identifiers.add(alias.name)

        elif isinstance(node, ast.Call):

            if isinstance(node.func, ast.Name):

                identifiers.add(node.func.id)

            elif isinstance(node.func, ast.Attribute):

                identifiers.add(node.func.attr)

        elif isinstance(node, ast.Attribute):

            identifiers.add(node.attr)

        elif isinstance(node, ast.ClassDef):

            identifiers.add(node.name)

        elif isinstance(node, ast.FunctionDef):

            identifiers.add(node.name)

    return sorted(identifiers)


def extract_identifiers_from_text(text):

    # NL counterpart to extract_identifiers(). Text is not valid Python,
    # so we can't use ast. Instead, pull out identifier-like tokens
    # (including dotted ones like "LLMChain.run") via regex, and also
    # add each dot-separated part individually — mirroring how
    # extract_identifiers() adds both "module.sub" and "sub" for
    # imports. This lets detect_apis() match against either form.

    identifiers = set()

    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
        text
    )

    for token in tokens:

        identifiers.add(token)

        for part in token.split("."):

            if part:
                identifiers.add(part)

    return sorted(identifiers)


def detect_apis(
    identifiers,
    api_list
):

    normalized = {}

    for api in api_list:

        normalized[
            api.replace("()", "")
        ] = api

    detected = []

    for identifier in identifiers:

        if identifier in normalized:

            detected.append(
                normalized[identifier]
            )

        elif (
            identifier
            and identifier[0].isupper()
            and len(identifier) > 2
            and identifier.lower() not in COMMON_WORDS
        ):

            # Not in deprecated_apis.json, but looks like a class name
            # (e.g. ChatOpenAI, OpenAIEmbeddings) — keep it, since it's
            # a strong retrieval signal even without a JSON match.
            # Length check filters out short noise like "X", "Db".
            detected.append(identifier)

        elif (
            "." in identifier
            and len(identifier) > 4
        ):

            # Dotted, method-call-shaped identifier (e.g. chain.run,
            # llm.predict) that's lowercase and not in deprecated_apis.json.
            # Still a strong retrieval signal — without this, natural
            # language questions about lowercase method calls fall back
            # to the full vague sentence as the retrieval query instead
            # of the specific API name, and retrieval grabs generic
            # overview chunks instead of the actual deprecation entry.
            detected.append(identifier)

    return sorted(
        list(set(detected))
    )


def extract_requested_api(query):

    matches = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_()]*",
        query
    )

    if matches:
        return matches[0]

    return ""


def analyze_input(
    text,
    deprecated_apis
):

    version = extract_version(text)

    python_mode = is_python_code(text)

    error_mode = is_error_query(text)

    if python_mode:

        identifiers = extract_identifiers(text)

    else:

        identifiers = extract_identifiers_from_text(text)

    detected_apis = detect_apis(
        identifiers,
        deprecated_apis
    )

    requested_api = extract_requested_api(text)

    return {
        "version": version,
        "python_mode": python_mode,
        "error_mode": error_mode,
        "identifiers": identifiers,
        "detected_apis": detected_apis,
        "requested_api": requested_api
    }