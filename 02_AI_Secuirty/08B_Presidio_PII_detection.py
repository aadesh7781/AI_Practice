from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

text = """
My name is Rahul Sharma.
My email is rahul@example.com.
"""

results = analyzer.analyze(
    text=text,
    language="en"
)

anonymized = anonymizer.anonymize(
    text=text,
    analyzer_results=results
)

print(anonymized.text)