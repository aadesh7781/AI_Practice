import re

PII_PATTERNS = {
    "EMAIL": r'\b[\w.-]+@[\w.-]+\.\w+\b',
    "PHONE": r'\b\d{10}\b',
    "PAN": r'\b[A-Z]{5}\d{4}[A-Z]\b'
}

def detect_pii(text):

    detected = {}

    for pii_type, pattern in PII_PATTERNS.items():

        matches = re.findall(pattern, text)

        if matches:
            detected[pii_type] = matches

    return detected

text = """
My email is rahul@example.com.
My phone number is 9876543210.
My PAN is ABCDE1234F.
"""

result = detect_pii(text)

print(result)

def redact_pii(text):

    for pii_type, pattern in PII_PATTERNS.items():

        text = re.sub(
            pattern,
            f"[{pii_type}_REDACTED]",
            text
        )

    return text

text = """
Email: rahul@example.com
Phone: 9876543210
PAN: ABCDE1234F
"""

print(redact_pii(text))

def mask_phone(phone):

    return "*" * (len(phone) - 4) + phone[-4:]

phone = "9876543210"

print(mask_phone(phone))