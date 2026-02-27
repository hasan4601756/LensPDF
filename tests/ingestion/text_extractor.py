import pytest
from app.ingestion.extractor import extract_text

from pathlib import Path

BASE_PATH = Path(__file__).parent / "test_data"


def test_text_pdf_extraction():
    pdf_path = BASE_PATH / "AO-18-2021.pdf"

    result = extract_text(pdf_path)

    assert result is not None, "extract_text returned None"
    assert result.get('text') is not None
    assert "POLICY GUIDELINES GOVERNING ENCASHMENT OF LPR" in result['text']


def test_text_scanned_pdf_extraction():
    pdf_path = BASE_PATH / "OC-083-2006.pdf"

    result = extract_text(pdf_path)

    assert result is not None, "extract_text returned None"
    assert result.get('text') is not None
    assert "ORGANIZATIONAL ANNOUNCEMENT" in result['text']

def test_text_scanned_urdu_pdf_extraction():
    pdf_path = BASE_PATH / "Safety Policy May 2022.pdf"

    result = extract_text(pdf_path)

    assert result is not None, "extract_text returned None"
    assert result.get('text') is not None
    assert "میں، بطور چیف ایگزیکٹو آفیسر ائیرشل سیفٹی، پیشہ ورانہ صحت اور حفاظات (OH & S)، سیکیورٹی، ماحولیات اور کوالٹی مینجمنٹ کے عزم کا اعادہ کرتا ہوں۔ اسی مناسبت سے، میں نے پی آئی اے سی ایل کی انتظامیہ کو یہ یقینی بنانے کا حکم دیا ہے کہ روزمرہ کی کارروائیوں کا انتظام کرتے ہوئے ان تمام مرکزی نکات کو مناسب ترجیح دی جائے۔ہم اس بات کو یقینی بنائیں گے کہ اس کارپوریٹ پالیسی کے نفاذ کے لیے ذمہ داری اور جوابدہی کے تنظیمی ڈھانچے کے تمام سطحوں کے ذریعے، بشمول میرے، محکمہ کے سربراہان، فنکشنل مینیجرز، فرنٹ لائن آپریشنل ملازمین سے لے کر کام کا محدود دائرہ کار رکھنے والے ملازمین تک پھیلے ہیں۔ میری ہدایات پر، یہ پالیسی تمام بیرونی خدمات فراہم کرنے والوں اور اسٹیک ہولڈنگ ایجنٹس تک بھی پھیلی ہوگی۔ میں اور PIACL کی پوری انتظامیہ اس بات کے عہد کرتی ہے کہ اس کارپوریٹ پالیسی کے بارے میں آگاہی، سمجھ بوجھ، نفاذ اور فروغ کو یقینی بنانے کے لیے ہر ممکن کوشش کی جائے گی۔" in result['text']

if __name__=="__main__":
    test_text_pdf_extraction()
    test_text_scanned_pdf_extraction()
    test_text_scanned_urdu_pdf_extraction()