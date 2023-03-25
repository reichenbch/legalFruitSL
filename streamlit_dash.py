from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import streamlit as st
import json
from predict import run_prediction

st.set_page_config(layout="wide")

model_list = ['alex-apostolo/legal-bert-small-cuad']

model_checkpoint = model_list[0]

if model_checkpoint == "akdeniz27/deberta-v2-xlarge-cuad": import sentencepiece


def load_model():
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    return model, tokenizer


def load_questions():
    with open('test.json') as json_file:
        data = json.load(json_file)

    questions = []
    for i, q in enumerate(data['data'][0]['paragraphs'][0]['qas']):
        question = data['data'][0]['paragraphs'][0]['qas'][i]['question']
        questions.append(question)
    return questions


def load_contracts():
    with open('test.json') as json_file:
        data = json.load(json_file)

    contracts = []
    for i, q in enumerate(data['data']):
        contract = ' '.join(data['data'][i]['paragraphs'][0]['context'].split())
        contracts.append(contract)
    return contracts

def load_titles():
    with open('test.json') as json_file:
        data = json.load(json_file)

    titles = []
    for i, q in enumerate(data['data']):
        title = data['data'][i]['title']
        titles.append(title)
    return titles


model, tokenizer = load_model()
questions = load_questions()
contracts = load_contracts()
titles = load_titles()

contract = contracts[0]
title = titles[0]

st.header("Legal Document QA Engine")

selected_question = st.selectbox('Choose any one query from the pool:', questions)
question_set = [questions[0], selected_question]

search_type = st.radio("Select Title/Contract", ("Title Search", "Contract Search"))

if search_type == "Title Search":
    title_type = st.radio("Select Title", ("Sample Title", "New Title"))

    if title_type == "Sample Title":
        sample_title_num = st.slider("Select Sample Title #")
        title = titles[sample_title_num]
        with st.expander(f"Sample Title #{sample_title_num}"):
            st.write(title)
    else:
        title = st.text_area("Input New Title", "", height=128)
elif search_type == "Contract Search":
    contract_type = st.radio("Select Contract", ("Sample Contract", "New Contract"))
    if contract_type == "Sample Contract":
        sample_contract_num = st.slider("Select Sample Contract #")
        contract = contracts[sample_contract_num]
        with st.expander(f"Sample Contract #{sample_contract_num}"):
            st.write(contract)
    else:
        contract = st.text_area("Input New Contract", "", height=256)

Run_Button = st.button("Run", key=None)
if Run_Button == True and not len(contract) == 0 and not len(question_set) == 0:
    if search_type == "Title Search":
        predictions = run_prediction(question_set, title, 'alex-apostolo/legal-bert-small-cuad')
    else:
        predictions = run_prediction(question_set, contract, 'alex-apostolo/legal-bert-small-cuad')

    for i, p in enumerate(predictions):
        print(predictions[p])
        if i != 0: st.write(f"Question: {question_set[int(p)]}\n\nAnswer: {predictions[p]}\n\n")
