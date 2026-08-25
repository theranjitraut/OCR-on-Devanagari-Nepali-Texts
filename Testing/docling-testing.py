'''
pip install docling
'''
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("/content/nepali-ocr-testing.pdf")

print(result.document.export_to_markdown())
