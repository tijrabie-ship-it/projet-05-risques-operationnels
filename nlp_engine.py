import os
import json
import logging
import fitz # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

# Configure Gemini API if available
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use gemini-flash-latest as requested by user's API key
    model = genai.GenerativeModel('gemini-flash-latest')
else:
    model = None
    logging.warning("GEMINI_API_KEY is not set. NLP features will use mock data.")

# Mock Bank Policy for contradiction detection
BANK_POLICY = """
سياسة المصرف الداخلية:
1. نسبة غرامة التأخير يجب ألا تتجاوز 5%.
2. لا يجوز إعفاء المورد بالكامل من المسؤولية في حال انقطاع الخدمة (الحد الأقصى للإعفاء هو ظروف القوة القاهرة فقط).
3. يجب أن يكون الاختصاص القضائي للمحاكم المحلية.
4. مدة الإشعار لإنهاء العقد يجب ألا تقل عن 30 يوماً.
"""

def extract_text(file_path):
    text = ""
    try:
        if file_path.endswith('.pdf'):
            import fitz
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_path.endswith('.docx'):
            import docx
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
            
        if not text or len(text.strip()) == 0:
            return None
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return None
    return text

def analyze_contract_text(contract_text):
    if not contract_text or len(contract_text.strip()) == 0:
        return {"status": "error", "message": "لا يوجد نص لتحليله في العقد."}

    if model:
        prompt = f"""
أنت مساعد قانوني ذكي خبير في اللغة العربية. قم بتحليل العقد التالي واستخرج المعلومات المطلوبة بشكل JSON.

العقد:
{contract_text}

سياسات المصرف للمقارنة:
{BANK_POLICY}

المطلوب استخراجه:
1. تواريخ_حرجة: استخرج تواريخ بداية العقد، ونهايته، وتواريخ الأقساط بدقة متناهية كما هي مكتوبة تماماً في النص (احذر من تغيير السنوات أو الأيام). اذكر الحدث والتاريخ.
2. شروط_جزائية: قائمة بالشروط الجزائية المذكورة ونسب التأخير.
3. أسماء_الأطراف: أسماء الجهات المتعاقدة.
4. بنود_عالية_المخاطر: أي بند يعفي المورد من المسؤولية أو يشكل خطراً كبيراً.
5. التناقضات: هل يوجد أي تعارض بين العقد وسياسات المصرف المذكورة أعلاه؟

يجب أن يكون الرد بصيغة JSON حصراً بهذا الهيكل (دون أي نص إضافي، فقط JSON صحيح):
{{
    "critical_dates": ["تاريخ 1", "تاريخ 2"],
    "penalty_clauses": ["شرط 1", "شرط 2"],
    "parties": ["طرف 1", "طرف 2"],
    "high_risk_clauses": ["بند 1"],
    "contradictions": ["تناقض 1"]
}}
"""
        try:
            response = model.generate_content(prompt)
            # Clean up the response to get only JSON
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
            
            return {"status": "success", "data": json.loads(response_text)}
        except Exception as e:
            logging.error(f"Error parsing Gemini response: {e}")
            return {
                "status": "error",
                "message": f"حدث خطأ أثناء الاتصال بنموذج الذكاء الاصطناعي: {str(e)}"
            }
    else:
        # Mock Response
        return {
            "status": "success",
            "data": {
                "critical_dates": ["01/01/2024 (بداية العقد)", "31/12/2024 (نهاية العقد)"],
                "penalty_clauses": ["غرامة تأخير بنسبة 10% من قيمة الدفعة المتبقية"],
                "parties": ["الطرف الأول: مصرف أمنية", "الطرف الثاني: شركة التقنية للتوريدات"],
                "high_risk_clauses": ["الطرف الثاني غير مسؤول عن أي انقطاع للخدمة لأي سبب كان."],
                "contradictions": ["غرامة التأخير 10% تتجاوز الحد المسموح به في سياسة المصرف (5%).", "إعفاء المورد يتعارض مع سياسة المصرف."]
            },
            "warning": "ملاحظة: هذه بيانات تجريبية (Mock) لعدم وجود مفتاح GEMINI_API_KEY."
        }

def evaluate_supplier_sentiment(supplier_data):
    # supplier_data contains: supplier_name, delay_rate, unstructured_text
    delay_rate = float(supplier_data.get('delay_rate', 0))
    unstructured_text = supplier_data.get('unstructured_text', '')

    if model:
        prompt = f"""
أنت محلل مخاطر موردين. قم بتحليل البيانات التالية للمورد:
اسم المورد: {supplier_data.get('supplier_name', 'غير معروف')}
نسبة التأخير التاريخية: {delay_rate}%
النصوص المتوفرة (أخبار، إيميلات، تقارير):
{unstructured_text[:5000]}

المطلوب:
1. تحليل مشاعر النصوص واستخراج العلامات التحذيرية (Early Warning Signals) مثل إضرابات، مشاكل مالية، دعاوى قضائية.
2. بناء مؤشر مخاطر (Risk Score) من 0 إلى 100، حيث يأخذ بعين الاعتبار نسبة التأخير (كل 1% تأخير تزيد المخاطرة) بالإضافة للمشاكل المستخرجة من النصوص.
3. قرار نهائي قصير.

يجب أن يكون الرد بصيغة JSON حصراً بهذا الهيكل:
{{
    "early_warnings": ["تحذير 1", "تحذير 2"],
    "sentiment": "سلبي/إيجابي/محايد",
    "risk_score": 85,
    "decision": "قرار قصير"
}}
"""
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
            
            return {"status": "success", "data": json.loads(response_text)}
        except Exception as e:
            logging.error(f"Gemini API Error: {e}")
            return {"status": "error", "message": "حدث خطأ أثناء تقييم المورد بواسطة الذكاء الاصطناعي."}
    else:
        # Mock response based on input
        risk_score = min(100, int(delay_rate * 2) + 20)
        warnings = []
        sentiment = "محايد"
        if "إضراب" in unstructured_text or "مالي" in unstructured_text or "ديون" in unstructured_text:
            risk_score += 40
            warnings.append("إشارات قوية على وجود مشاكل مالية أو عمالية بناءً على النصوص.")
            sentiment = "سلبي"

        return {
            "status": "success",
            "data": {
                "early_warnings": warnings if warnings else ["لا توجد علامات تحذيرية واضحة في النصوص."],
                "sentiment": sentiment,
                "risk_score": min(100, risk_score),
                "decision": "يوصى بالمتابعة المستمرة" if risk_score < 50 else "يحتاج المورد لتدخل أو إيقاف مؤقت."
            },
            "warning": "ملاحظة: هذه بيانات تجريبية (Mock) لعدم وجود مفتاح GEMINI_API_KEY."
        }


def extract_aml_features_from_text(text):
    if not text:
        return None
        
    prompt = '''
    أنت خبير في كشف غسيل الأموال ومراجعة العمليات البنكية.
    استخرج المتغيرات التالية من النص التالي بصيغة JSON بدقة (ويجب أن تكون القيم مطابقة للخيارات إن وجدت):
    
    - 'دخل' (رقم صحيح)
    - 'قيمة المعاملة' (رقم صحيح)
    - 'تكرار المعاملات (أسبوع)' (رقم صحيح)
    - 'عمر العميل' (رقم صحيح)
    - 'عمر الحساب (أيام)' (رقم صحيح)
    - 'درجة الامتثال للشريعة (1-5)' (رقم صحيح من 1 إلى 5)
    - 'أيام منذ آخر معاملة' (رقم صحيح)
    - 'نوع الزبون' (نص: 'فرد' أو 'شركة')
    - 'نوع العملة' (نص: مثلا 'MAD', 'USD', 'EUR')
    - 'معاملة عابرة للحدود' (نص: 'نعم' أو 'لا')
    - 'حالة التحقق KYC' (نص: 'مكتمل', 'قيد المراجعة', أو 'غير مكتمل')
    - 'درجة مخاطر العميل' (نص: 'منخفض', 'متوسط', 'مرتفع')
    - 'مستوى الحساب' (نص: 'عادي', 'مميز', 'VIP')
    - 'شخصية سياسية معرضة' (نص: 'نعم' أو 'لا')
    - 'أخبار سلبية' (نص: 'نعم' أو 'لا')
    
    ملاحظة: يجب أن يكون الناتج JSON فقط بدون أي نص إضافي، بحيث تكون أسماء الحقول متطابقة تماماً مع المكتوب أعلاه.
    
    النص:
    ''' + text
    
    if model is None:
        return {
            'دخل': 150000,
            'قيمة المعاملة': 500000,
            'تكرار المعاملات (أسبوع)': 5,
            'عمر العميل': 45,
            'عمر الحساب (أيام)': 30,
            'درجة الامتثال للشريعة (1-5)': 3,
            'أيام منذ آخر معاملة': 2,
            'نوع الزبون': 'فرد',
            'نوع العملة': 'USD',
            'معاملة عابرة للحدود': 'نعم',
            'حالة التحقق KYC': 'قيد المراجعة',
            'درجة مخاطر العميل': 'مرتفع',
            'مستوى الحساب': 'VIP',
            'شخصية سياسية معرضة': 'نعم',
            'أخبار سلبية': 'نعم'
        }
        
    try:
        response = model.generate_content(prompt)
        result_text = response.text
        import json
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = result_text[json_start:json_end]
            return json.loads(json_str)
        else:
            return None
    except Exception as e:
        import logging
        logging.error(f"Error in AML NLP extraction: {e}")
        return None
