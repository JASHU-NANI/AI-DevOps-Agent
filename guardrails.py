
import re

REDACTED = "***REDACTED***"

SECRET_PATTERNS = [
    # GitHub tokens
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),

    # Google API keys
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),

    # Bearer tokens
    re.compile(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"
    ),
   
]

SECRET_ASSIGNMENT = re.compile(
    r"""(?i)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|"""
    r"""password|passwd|client[_-]?secret|aws_secret_access_key)"""
    r"""\b\s*[:=]\s*)(["'])(.*?)(\2)"""
)

PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    r".*?"
    r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    re.DOTALL,
)


def redact(text: str) -> str:
    #mask common secret formats before external AI processing.
    if not text:
        return text

    text = PRIVATE_KEY.sub(REDACTED, text)

    def replace_assignment(match):
        return (
            match.group(1)
            + match.group(2)
            + REDACTED
            + match.group(4)
        )

    text = SECRET_ASSIGNMENT.sub(replace_assignment, text)

    for pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)

    return text


def sanitize_for_ai(text: str) -> str:
    #common sanitization entry point for logs and source code.
    return redact(text)
