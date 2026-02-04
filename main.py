import os
import gradio as gr
import requests
import pandas as pd

# 127.0.0.1 is correct because Backend & Frontend are now in the same container
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict")

def predict_loan_ui(gender, age, income, location, prop_price, credit, card, prop_loc, co_app, expenses, exp1, exp2, request_amt, profession):
    # CRITICAL: Map UI strings to the shorthand your LabelEncoder was trained on
    gender_map = {"Male": "M", "Female": "F"}
    card_map = {"Yes": "Y", "No": "N", "Unpossessed": "U"}
    
    payload = {
        "Gender": gender_map.get(gender, gender), # Converts "Male" -> "M"
        "Age": age,
        "Income_USD": income,
        "Location": location,
        "Property_Price": prop_price,
        "Credit_Score": credit,
        "Has_Active_Credit_Card": card_map.get(card, card), # Converts "Yes" -> "Y"
        "Property_Location": prop_loc,
        "Co_Applicant": co_app,
        "Current_Loan_Expenses_USD": expenses,
        "Expense_Type_1": exp1,
        "Expense_Type_2": exp2,
        "Loan_Amount_Request_USD": request_amt,
        "Profession": profession
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            res = response.json()
            return res['status'], res['predicted_sanction_amount']
        # Provide better visibility into errors
        return "Error", f"Backend Error: {response.text}"
    except Exception as e:
        return "Connection Error", f"Is the backend running? {str(e)}"

# Define theme inside Blocks
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 Smart Loan Approval System")
    
    with gr.Row():
        with gr.Column():
            gender = gr.Dropdown(["Male", "Female"], label="Gender")
            age = gr.Number(value=18, label="Age")
            income = gr.Number(label="Income (USD)")
            location = gr.Radio(["Urban", "Semi-Urban", "Rural"], label="Location")
            profession = gr.Dropdown(["Working", "Commercial associate", "Pensioner", "State servant", "Other"], label="Profession")
        
        with gr.Column():
            prop_price = gr.Number(label="Property Price")
            credit = gr.Slider(300, 900, value=600, label="Credit Score")
            card = gr.Radio(["Yes", "No", "Unpossessed"], label="Active Credit Card?")
            prop_loc = gr.Dropdown(["Urban", "Semi-Urban", "Rural"], label="Property Location")
    
    with gr.Accordion("Additional Financial Details", open=False):
        co_app = gr.Number(0, label="Co-Applicant Income")
        expenses = gr.Number(0, label="Monthly Loan Expenses")
        exp1 = gr.Radio(["Y", "N"], value="N", label="Expense Type 1")
        exp2 = gr.Radio(["Y", "N"], value="N", label="Expense Type 2")
        request_amt = gr.Number(label="Loan Amount Request")

    btn = gr.Button("Evaluate Application", variant="primary")
    
    with gr.Row():
        out_status = gr.Textbox(label="Decision")
        out_amt = gr.Textbox(label="Sanctioned Amount")

    btn.click(
        predict_loan_ui, 
        inputs=[gender, age, income, location, prop_price, credit, card, prop_loc, co_app, expenses, exp1, exp2, request_amt, profession],
        outputs=[out_status, out_amt]
    )

if __name__ == "__main__":
    # Hugging Face specifically monitors 0.0.0.0:7860
    demo.launch(server_name="0.0.0.0", server_port=7860)