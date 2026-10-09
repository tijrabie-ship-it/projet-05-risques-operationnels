import joblib
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import logging
import nlp_engine


app = Flask(__name__)

# Load the credit scoring ML model
try:
    model = joblib.load('xgboost_mourabaha_final.pkl')
except Exception as e:
    logging.warning(f"Could not load credit model: {e}")
    model = None

# Load the AML ML model
try:
    aml_model = joblib.load('aml_xgboost_model.pkl')
except Exception as e:
    logging.warning(f"Could not load AML model: {e}")
    aml_model = None

# Expected feature order based on inspection
FEATURES = [
    'age_client', 'monthly_salary', 'financing_amount', 'monthly_installment',
    'employment_years', 'nb_children', 'gender', 'education_level',
    'family_status', 'murabaha_type', 'margin_rate', 'contract_duration_months',
    'collateral_value', 'contract_year', 'ead', 'dti', 'ltv',
    'collateral_coverage', 'ead_ratio', 'installment_to_amount'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'The model failed to load.'})

    try:
        # Extract features from form (only the basic ones)
        FORM_FEATURES = [
            'age_client', 'monthly_salary', 'financing_amount', 'monthly_installment',
            'employment_years', 'nb_children', 'gender', 'education_level',
            'family_status', 'murabaha_type', 'margin_rate', 'contract_duration_months',
            'collateral_value', 'contract_year'
        ]
        
        data = {}
        for feature in FORM_FEATURES:
            val = request.form.get(feature)
            if val is None:
                return jsonify({'status': 'error', 'message': f'Missing field: {feature}'})
            
            # Convert to numeric if it's a number
            if feature in ['gender', 'education_level', 'family_status', 'murabaha_type']:
                data[feature] = val
            else:
                if val.strip() == '':
                    return jsonify({'status': 'error', 'message': f'يرجى التأكد من إدخال جميع البيانات (الحقل المفقود: {feature})'})
                try:
                    data[feature] = float(val)
                except ValueError:
                    return jsonify({'status': 'error', 'message': f'قيمة غير صالحة في الحقل: {feature}'})

        # Feature Engineering: Auto-calculate derived features
        monthly_salary = data['monthly_salary']
        financing_amount = data['financing_amount']
        monthly_installment = data['monthly_installment']
        collateral_value = data['collateral_value']
        
        data['dti'] = monthly_installment / monthly_salary if monthly_salary > 0 else 0
        data['ltv'] = financing_amount / collateral_value if collateral_value > 0 else 0
        data['ead'] = financing_amount * 0.8  # Defaulting EAD to 80% of financing amount
        data['ead_ratio'] = data['ead'] / financing_amount if financing_amount > 0 else 0
        data['collateral_coverage'] = collateral_value / financing_amount if financing_amount > 0 else 0
        data['installment_to_amount'] = monthly_installment / financing_amount if financing_amount > 0 else 0

        # Create DataFrame for prediction
        df = pd.DataFrame([data])
        
        # Ensure order
        df = df[FEATURES]

        # Business Rules (قواعد الأعمال الصارمة)
        age_at_end = data['age_client'] + (data['contract_duration_months'] / 12.0)
        
        # 1. Age Rule
        if age_at_end > 75:
            prediction = 1
            probability = 0.99
            rejection_reason = "تجاوز العمر المسموح بنهاية العقد (75 عاماً)."
            
        # 2. DTI Rule (Debt to Income)
        elif data['dti'] > 0.60:
            prediction = 1
            probability = 0.99
            rejection_reason = "نسبة الاستقطاع (DTI) تتجاوز 60% من الراتب."
            
        # 3. LTV Rule (Loan to Value / Collateral)
        elif data['ltv'] > 1.5 and collateral_value > 0:
            prediction = 1
            probability = 0.99
            rejection_reason = "مبلغ التمويل أعلى بكثير من قيمة الضمان (LTV > 150%)."
            
        # 4. Absurd Math Rule (Installment cannot cover principal)
        elif monthly_installment > 0 and data['contract_duration_months'] > 0 and (monthly_installment * data['contract_duration_months']) < financing_amount:
            prediction = 1
            probability = 0.99
            rejection_reason = "قيمة القسط الشهري والمدة لا تغطي أصل مبلغ التمويل!"
            
        else:
            rejection_reason = None
            # Predict using the ML Model
            prediction = model.predict(df)[0]
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(df)[0][1] # Probability of default
            else:
                probability = float(prediction)

            # --- Expert Risk Calibration Layer (طبقة تعديل المخاطر) ---
            age = data['age_client']
            employment_years = data['employment_years']

            # 1. Age rules
            if age < 20:
                probability += 0.40  # +40% risk for very young clients
            elif age > 55:
                probability += (age - 55) * 0.025  # +2.5% risk for each year over 55

            # 2. Employment rules
            if employment_years < 1:
                probability += 0.30  # +30% risk for less than 1 year of work
            elif employment_years < 3:
                probability += 0.15  # +15% risk for less than 3 years
            elif employment_years >= 10:
                probability -= 0.05  # -5% risk (Bonus) for stable employment > 10 years

            # Bound the probability between 0 and 0.98 (leave 0.99 for hard rejections)
            probability = max(0.0, min(0.98, probability))

            # Re-evaluate final decision based on new calibrated probability
            prediction = 1 if probability >= 0.5 else 0

        if prediction == 1 and probability >= 0.99 and rejection_reason:
            final_text = f"عميل عالي المخاطر - مرفوض آلياً ({rejection_reason})"
        else:
            final_text = 'عميل عالي المخاطر (قد يتخلف عن السداد)' if prediction == 1 else 'عميل جيد (مؤهل للتمويل)'

        return jsonify({
            'status': 'success',
            'prediction': final_text,
            'probability': float(probability) * 100
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

# --- NLP Endpoints ---

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/analyze_contract', methods=['POST'])
def analyze_contract():
    if 'contract_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'لم يتم العثور على ملف العقد.'})
    
    file = request.files['contract_file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'لم يتم اختيار ملف.'})
    
    if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt') or file.filename.endswith('.docx')):
        import uuid
        ext = os.path.splitext(file.filename)[1]
        filename = str(uuid.uuid4()) + ext
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text
        text = nlp_engine.extract_text(filepath)
        
        # Optional: delete file after extraction
        try:
            os.remove(filepath)
        except:
            pass
            
        if text is None:
            return jsonify({'status': 'error', 'message': 'فشل في استخراج النص من الملف. يرجى التأكد من أن الملف ليس صورة.'})
            
        # Analyze text
        result = nlp_engine.analyze_contract_text(text)
        return jsonify(result)
    else:
        return jsonify({'status': 'error', 'message': 'يرجى رفع ملف بصيغة PDF أو TXT أو DOCX.'})

@app.route('/evaluate_supplier', methods=['POST'])
def evaluate_supplier():
    data = request.json if request.is_json else request.form.to_dict()
    if not data:
        return jsonify({'status': 'error', 'message': 'بيانات غير مكتملة.'})
        
    supplier_data = {
        'supplier_name': data.get('supplier_name'),
        'delay_rate': data.get('delay_percentage'),
        'unstructured_text': data.get('feedback_text')
    }
    result = nlp_engine.evaluate_supplier_sentiment(supplier_data)
    return jsonify(result)

@app.route('/analyze_aml_transaction', methods=['POST'])
def analyze_aml_transaction():
    if 'aml_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'لم يتم العثور على الملف.'})
        
    file = request.files['aml_file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'لم يتم اختيار ملف.'})
        
    if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt') or file.filename.endswith('.docx')):
        import uuid
        ext = os.path.splitext(file.filename)[1]
        filename = str(uuid.uuid4()) + ext
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text
        text = nlp_engine.extract_text(filepath)
        
        # Optional: delete file after extraction
        try:
            os.remove(filepath)
        except:
            pass
            
        if text is None:
            return jsonify({'status': 'error', 'message': 'فشل في استخراج النص من الملف.'})
            
        # Analyze text using NLP to extract features
        features_dict = nlp_engine.extract_aml_features_from_text(text)
        if not features_dict:
            return jsonify({'status': 'error', 'message': 'فشل الذكاء الاصطناعي في استخراج البيانات من التقرير.'})
            
        # Predict AML risk if model is loaded
        decision = "غير محدد"
        risk_probability = 0.0
        is_aml_alert = False
        
        if aml_model is not None:
            import pandas as pd
            try:
                # Convert to DataFrame
                df_features = pd.DataFrame([features_dict])
                
                # Ensure all expected columns are present
                expected_cols = [
                    'دخل', 'قيمة المعاملة', 'تكرار المعاملات (أسبوع)', 
                    'عمر العميل', 'عمر الحساب (أيام)', 'درجة الامتثال للشريعة (1-5)', 'أيام منذ آخر معاملة',
                    'نوع الزبون', 'نوع العملة', 'معاملة عابرة للحدود', 'حالة التحقق KYC', 
                    'درجة مخاطر العميل', 'مستوى الحساب', 'شخصية سياسية معرضة', 'أخبار سلبية'
                ]
                
                # Fill missing columns with defaults if any
                for col in expected_cols:
                    if col not in df_features.columns:
                        if col in ['نوع الزبون', 'نوع العملة', 'معاملة عابرة للحدود', 'حالة التحقق KYC', 'درجة مخاطر العميل', 'مستوى الحساب', 'شخصية سياسية معرضة', 'أخبار سلبية']:
                            df_features[col] = 'غير محدد'
                        else:
                            df_features[col] = 0
                            
                df_features = df_features[expected_cols]
                
                # Predict
                prediction = aml_model.predict(df_features)[0]
                proba = aml_model.predict_proba(df_features)[0]
                risk_probability = float(proba[1]) * 100
                is_aml_alert = bool(prediction == 1)
                
                if is_aml_alert:
                    decision = "تنبيه AML عالي الخطورة! معاملة مشبوهة."
                else:
                    decision = "معاملة آمنة."
            except Exception as e:
                logging.error(f"AML Model Prediction Error: {e}")
                decision = "فشل في تشغيل النموذج."
        else:
            # Fallback mock logic if ML model not loaded
            if features_dict.get('شخصية سياسية معرضة', 'لا') == 'نعم' and features_dict.get('معاملة عابرة للحدود', 'لا') == 'نعم':
                is_aml_alert = True
                risk_probability = 95.5
                decision = "تنبيه AML عالي الخطورة! (وضع المحاكاة)"
            else:
                is_aml_alert = False
                risk_probability = 15.0
                decision = "معاملة آمنة (وضع المحاكاة)."
                
        return jsonify({
            'status': 'success',
            'extracted_features': features_dict,
            'is_aml_alert': is_aml_alert,
            'risk_probability': risk_probability,
            'decision': decision
        })
    else:
        return jsonify({'status': 'error', 'message': 'يرجى رفع ملف بصيغة PDF أو TXT أو DOCX.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
