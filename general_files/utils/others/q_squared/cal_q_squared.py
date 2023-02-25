# Copyright 2020 The Q2 Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import spacy
import re
import string
from collections import Counter
from bert_score import score


INVALID_QUESTION = -1
NO_ANS = '[CLS]'
NO_VALID_QUESTIONS = 'NO_Q'
NO_NLI = 'NO_NLI'
NO_Q = -1
ENTAILMENT_SCORE = 1
CONTRADICTION_SCORE = 0
NEUTRAL_SCORE = 0.5

nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the|in|our)\b', ' ', text)
    return re.sub(' +', ' ', text).strip()


def f1_score(a_gold, a_pred):
    if a_pred == '':
        return 0
    gold_toks = clean_text(a_gold).split()
    pred_toks = clean_text(a_pred).split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_bert_score(a_gold, a_pred):
    P, R, F1 = score(a_pred, a_gold, lang="en", verbose=True)
    return F1.mean().item()



def get_answer(question, text, qa_model, qa_tokenizer, config):  # Code taken from https://huggingface.co/transformers/task_summary.html
    inputs = qa_tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt").to(config.default_device)
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = qa_model(**inputs, return_dict=False)

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    ans = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return ans


def get_answer_candidates(text):
    all_candidates = []
    doc = nlp(text)
    candidates = [ent.text for ent in list(doc.ents)]
    noun_chunks = list(doc.noun_chunks)
    for chunk in noun_chunks:
        found = False
        for cand in candidates:
            if chunk.text.lower() == cand.lower():
                found = True
        if not found:
            candidates.append(chunk.text)
    # candidates += [chunk.text for chunk in list(doc.noun_chunks) if chunk.text not in candidates]
    candidates = [cand for cand in candidates if cand.lower() != 'i']
    return candidates


# def get_answer_candidates(text):
#     doc = nlp(text)
#     candidates = [ent.text for ent in list(doc.ents)]
#     candidates_lower = [c.lower() for c in candidates]
#     noun_chunks = list(doc.noun_chunks)
#     candidates += [c.text for c in noun_chunks if c.text.lower() not in candidates_lower and c.text.lower() != 'i']
#     return candidates


def get_question_greedy(answer, context, qg_model, qg_tokenizer, max_length=128):
    input_text = ["answer: %s  context: %s </s>" % (a, c) for a, c in (answer, context)]
    features = qg_tokenizer(input_text, return_tensors='pt')
    output = qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                               max_length=max_length)

    question = [qg_tokenizer.decode(output[0]).replace("question: ", "", 1)]
    return question


def get_questions_beam(answer, context, qg_model, qg_tokenizer, config, max_length=128, beam_size=5, num_return=5):
    all_questions = []
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = qg_tokenizer([input_text], return_tensors='pt').to(config.default_device)

    beam_outputs = qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                                     max_length=max_length, num_beams=beam_size, no_repeat_ngram_size=3,
                                     num_return_sequences=num_return, early_stopping=True)

    for beam_output in beam_outputs:
        all_questions.append(qg_tokenizer.decode(beam_output, skip_special_tokens=True).replace("question: ", "", 1))
    return all_questions


def get_questions_sample(answer, context, qg_model, qg_tokenizer, max_length=128, top_k=50, top_p=0.95, num_return=5):
    all_questions = []
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = qg_tokenizer([input_text], return_tensors='pt')

    sampled_outputs = qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                                        max_length=max_length, do_sample=True, top_k=top_k, top_p=top_p,
                                        num_return_sequences=num_return)

    for sampled in sampled_outputs:
        all_questions.append(qg_tokenizer.decode(sampled, skip_special_tokens=True).replace("question: ", "", 1))

    return all_questions

def filter_questions(exp_ans, pred_ans):
    if pred_ans == NO_ANS:
        return 'NO MATCH'
    if clean_text(exp_ans) != clean_text(pred_ans):
        return 'NO MATCH'
    return 'VALID'


def non_personal(question):
    question_tok = nlp(question)
    for tok in question_tok:
        if tok.dep_ == 'nsubj':
            if tok.text.lower() == 'i' or tok.text.lower() == 'you':
                return False
        elif tok.dep_ == 'poss':
            if tok.text.lower() == 'my' or tok.text.lower() == 'your':
                return False
    return True


def single_question_score(question, cand, response, knowledge, qa_model, qa_tokenizer, config):
    pred_ans = get_answer(question, response, qa_model, qa_tokenizer, config)

    if filter_questions(cand, pred_ans) == 'VALID':
        knowledge_ans = get_answer(question, knowledge, qa_model, qa_tokenizer, config)
        if knowledge_ans != NO_ANS:
            return f1_score(cand, knowledge_ans), knowledge_ans
        else:
            return 0, NO_ANS
    else:
        return INVALID_QUESTION, INVALID_QUESTION


def get_response_score(response, knowledge, gen_method, single, remove_personal, qg_model, qg_tokenizer, qa_model, qa_tokenizer, config):
    f1 = 0
    num_questions = 0

    valid_questions = []
    valid_cands = []
    knowledge_answers = []
    scores = []

    candidates = get_answer_candidates(response)
    for cand in candidates:
        if gen_method == 'greedy':
            questions = [get_question_greedy(cand, response)]
        elif gen_method == 'beam':
            questions = get_questions_beam(cand, response, qg_model, qg_tokenizer, config)
        else:
            questions = get_questions_sample(cand, response)

        for question in questions:
            if not remove_personal or non_personal(question):
                question_score, knowledge_ans = single_question_score(question, cand, response, knowledge, qa_model, qa_tokenizer, config)
                if question_score != INVALID_QUESTION:
                    num_questions += 1
                    f1 += question_score

                    valid_questions.append(question)
                    valid_cands.append(cand)
                    knowledge_answers.append(knowledge_ans)
                    scores.append(question_score)

                    if single:
                        break
    if num_questions:
        avg_f1 = f1 / num_questions
    else:
        avg_f1 = INVALID_QUESTION
    return avg_f1, valid_questions, valid_cands, knowledge_answers, scores


def response_questions_stats(response, knowledge, gen_method, single, remove_personal):
    num_questions = 0
    num_no_ans = 0

    candidates = get_answer_candidates(response)
    for cand in candidates:
        if gen_method == 'greedy':
            questions = [get_question_greedy(cand, response)]
        elif gen_method == 'beam':
            questions = get_questions_beam(cand, response)
        else:
            questions = get_questions_sample(cand, response)

        for question in questions:
            if not remove_personal or non_personal(question):
                pred_ans = get_answer(question, response)

                if filter_questions(cand, pred_ans) == 'VALID':
                    num_questions += 1
                    knowledge_ans = get_answer(question, knowledge)
                    if knowledge_ans == NO_ANS:
                        num_no_ans += 1
                    if single:
                        break
    return num_questions, num_no_ans


def get_stats(in_path, gen_method, single, remove_personal):
    num_questions = 0
    num_no_ans = 0
    df = pd.read_csv(in_path)
    for _, row in df.iterrows():
        q, no_ans = response_questions_stats(row['response'], row['knowledge'], gen_method, single, remove_personal)
        num_questions += q
        num_no_ans += no_ans


def get_e2e_nli_score(response, knowledge, predictor):
    res = predictor([(knowledge, response)], ['contradiction', 'entailment', 'neutral'])

    nli_label = res[0]['labels'][res[0]['scores'].index(max(res[0]['scores']))]

    if nli_label == 'entailment':  # If entails, the score is 1
        return ENTAILMENT_SCORE
    elif nli_label == 'contradiction':  # If contradicts, the score is 0
        return CONTRADICTION_SCORE
    else:
        return NEUTRAL_SCORE


def get_nli_label(question, cand, evidence_ans, predictor):
    premise = question + ' ' + evidence_ans + '.'
    hypothesis = question + ' ' + cand + '.'

    res = predictor([(premise, hypothesis)], ['contradiction', 'entailment', 'neutral'])

    return res[0]['labels'][res[0]['scores'].index(max(res[0]['scores']))]


def scores_with_nli(score, knowledge_ans, question, cand, response, knowledge, predictor):
    nli_scores = []
    f1_scores = []

    for i, row in enumerate(response):
        f1_score = score[i]

        evidence_answer = str(knowledge_ans[i])

        nli_score = f1_score

        # Use NLI to determine answer similarity.
        # This is only applicable for responses that had at least one valid question generated

        if 0 <= f1_score < 1 and NO_ANS not in evidence_answer and evidence_answer != '' and evidence_answer != 'nan':
            f1_scores.append(f1_score)
            # If the score is 1, there is a full overlap between the
            # candidate and the predicted answer, so the score is 1
            # If there is no answer - can't run NLI, keep the original score (0)

            nli_label = get_nli_label(str(question[i]), str(cand[i]), evidence_answer, predictor)

            if nli_label == 'entailment':  # If entails, the score is 1
                nli_score = ENTAILMENT_SCORE
            elif nli_label == 'contradiction':  # If contradicts, the score is 0
                nli_score = CONTRADICTION_SCORE

        # Add fallback NLI to responses that are not covered by Q2 (no questions generated)
        elif f1_score == NO_Q:
            nli_fallback = get_e2e_nli_score(str(response[i]), str(knowledge[i]).lower(), predictor)
            nli_score = nli_fallback
            f1_scores.append(nli_fallback)
        else:
            f1_scores.append(f1_score)

        nli_scores.append(nli_score)

    return nli_scores, f1_scores


def aggregate_per_response(q2_no_nli, q2_score, id):
    f1_scores_by_id = dict()
    nli_scores_by_id = dict()
    # knowledge_by_id = dict()
    # response_by_id = dict()
    # label_by_id = dict()

    for i, _ in enumerate(q2_score):
        idx = id[i]
        f1_score = q2_no_nli[i]
        nli_score = q2_score[i]

        if idx in f1_scores_by_id:
            f1_scores_by_id[idx].append(f1_score)
            nli_scores_by_id[idx].append(nli_score)
        else:
            f1_scores_by_id[idx] = [f1_score]
            nli_scores_by_id[idx] = [nli_score]
            # response_by_id[idx] = response[i]
            # knowledge_by_id[idx] = knowledge[i]
            # if for_systems_simulation:
            #     label_by_id[idx] = label[i]

    mean_f1_scores = []
    mean_nli_scores = []
    # responses = []
    # knowledge = []
    # labels = []

    for idx in f1_scores_by_id.keys():
        mean_f1_scores.append(np.mean(f1_scores_by_id[idx]))
        mean_nli_scores.append(np.mean(nli_scores_by_id[idx]))
        # responses.append(response_by_id[idx])
        # knowledge.append(knowledge_by_id[idx])
        # if for_systems_simulation:
        #     labels.append(label_by_id[idx])

    # print('Q2:', np.mean(mean_nli_scores))
    # print('Q2, no nli:', np.mean(mean_f1_scores))
    # data = {'id': list(f1_scores_by_id.keys()), 'response': responses, 'knowledge': knowledge,
    #         'Q2_no_nli': mean_f1_scores, 'Q2': mean_nli_scores}
    #
    # res_df = pd.DataFrame(data=data)
    # if for_systems_simulation:
    #     res_df['label'] = labels

    return np.mean(mean_nli_scores), np.mean(mean_f1_scores)


def add_baseline_e2e_nli(response, knowledge):
    e2e_nli_scores = []
    for i, resp in enumerate(response):
        e2e_nli_scores.append(get_e2e_nli_score(str(resp), str(knowledge[i]).lower()))
    return e2e_nli_scores



def calc_scores(response, knowledge, gen_method='beam', single=True, remove_personal=True, config=None):
    q_scores = []

    all_questions = []
    all_cands = []
    all_answers = []
    all_scores = []
    all_responses = []
    all_knowledge = []
    ids = []
    qg_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap", cache_dir=config.cache_dir)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap", cache_dir=config.cache_dir).to(config.default_device)
    qa_tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2", cache_dir=config.cache_dir)
    qa_model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2", cache_dir=config.cache_dir).to(config.default_device)
    for idx, resp in enumerate(tqdm(response, desc="计算Q-squared分数中")):
        res, res_questions, res_cands, res_answers, res_scores =\
            get_response_score(resp, knowledge[idx], gen_method, single, remove_personal, qg_model, qg_tokenizer, qa_model, qa_tokenizer, config)

        all_questions.extend(res_questions)
        all_cands.extend(res_cands)
        all_answers.extend(res_answers)
        all_scores.extend(res_scores)
        all_responses.extend([resp] * len(res_questions))
        all_knowledge.extend([knowledge[idx]] * len(res_questions))
        ids.extend([idx] * len(res_questions))

        if res == INVALID_QUESTION:
            all_questions.extend([NO_VALID_QUESTIONS])
            all_cands.extend([NO_VALID_QUESTIONS])
            all_answers.extend([NO_VALID_QUESTIONS])
            all_scores.extend([INVALID_QUESTION])
            all_responses.extend([resp.lower()])
            all_knowledge.extend([knowledge[idx]])
            ids.extend([idx])

        q_scores.append(res)
    qg_model = qg_model.to("cpu")
    qa_model = qa_model.to("cpu")
    predictor = pipeline("zero-shot-classification", model='boychaboy/SNLI_roberta-large')
    # if save_steps:
    #     data = {'id': ids, 'response': all_responses, 'cand': all_cands, 'question': all_questions, 'knowledge': all_knowledge,
    #             'knowledge_ans': all_answers, 'score': all_scores}
    #     steps_df = pd.DataFrame(data=data)
    #     steps_df.to_csv(out_path + '.steps.csv')

    q2_score, q2_no_nli = scores_with_nli(all_scores, all_answers, all_questions, all_cands, response, knowledge, predictor)
    Q2_nli, Q2_f1 = aggregate_per_response(q2_no_nli, q2_score, ids)

    valid_scores = [s for s in q_scores if s != -1]
    print("total with at least 1 valid question:", len(valid_scores))
    # print("score:", np.mean(valid_scores))
    print("Q2_nli:", str(Q2_nli))
    print("Q2_f1:", str(Q2_f1))

    return Q2_nli, Q2_f1


if __name__ == '__main__':
    from datasets import Dataset
    ds = Dataset.from_csv("/home/dengyf/code/faith_dial/general_files/utils/q_squared/third_party/data/dodeca_consistent.csv")
    calc_scores(ds['response'], ds['knowledge'])


# def calc_scores(in_path, gen_method, single, remove_personal, out_path='', save_steps=False):
#     print(in_path, gen_method, single, remove_personal)
#     print(save_steps, flush=True)
#     q_scores = []
#     df = pd.read_csv(in_path)
#
#     all_questions = []
#     all_cands = []
#     all_answers = []
#     all_scores = []
#     all_responses = []
#     all_knowledge = []
#     ids = []
#
#     for idx, row in tqdm(df.iterrows()):
#         res, res_questions, res_cands, res_answers, res_scores =\
#             get_response_score(row['response'], row['knowledge'], gen_method, single, remove_personal)
#
#         all_questions.extend(res_questions)
#         all_cands.extend(res_cands)
#         all_answers.extend(res_answers)
#         all_scores.extend(res_scores)
#         all_responses.extend([row['response']] * len(res_questions))
#         all_knowledge.extend([row['knowledge']] * len(res_questions))
#         ids.extend([idx] * len(res_questions))
#
#         if res == INVALID_QUESTION:
#             all_questions.extend([NO_VALID_QUESTIONS])
#             all_cands.extend([NO_VALID_QUESTIONS])
#             all_answers.extend([NO_VALID_QUESTIONS])
#             all_scores.extend([INVALID_QUESTION])
#             all_responses.extend([row['response'].lower()])
#             all_knowledge.extend([row['knowledge']])
#             ids.extend([idx])
#
#         q_scores.append(res)
#
#     if out_path != '':
#         df['Q2'] = q_scores
#         df = df[df.Q2 >= 0]
#         df.to_csv(out_path + '.csv')
#
#     if save_steps:
#         data = {'id': ids, 'response': all_responses, 'cand': all_cands, 'question': all_questions, 'knowledge': all_knowledge,
#                 'knowledge_ans': all_answers, 'score': all_scores}
#         steps_df = pd.DataFrame(data=data)
#         steps_df.to_csv(out_path + '.steps.csv')
#
#     valid_scores = [s for s in q_scores if s != -1]
#     print("total with at least 1 valid question:", len(valid_scores))
#     print("score:", np.mean(valid_scores))
#
#     return valid_scores
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--infile", type=str, default="/home/dengyf/code/faith_dial/general_files/utils/q_squared/third_party/data/dodeca_inconsistent.csv",
#                         help="Path to a csv file containing dialogue model outputs.")
#     parser.add_argument("--gen_method", type=str, default="beam", choices=['greedy', 'beam', 'sampling'],
#                         help="Decoding method for question generation.")
#     parser.add_argument("--q_per_cand", type=str, choices=['single', 'multi'], default='single', required=False,
#                         help="Take only one question per candidate when using beam/sampling for decoding")
#     parser.add_argument("--personal", type=str, choices=['keep', 'remove'], default='remove', required=False,
#                         help="Whether to remove personal questions.")
#     parser.add_argument("--outfile", type=str, default='', required=False, help="Path to an output file")
#     parser.add_argument("--save_steps", default=False, action="store_true", help="Whether to save all pipeline steps")
#     args = parser.parse_args()
#
#     if args.q_per_cand == 'single':
#         single_q = True
#     else:
#         single_q = False
#
#     if args.personal == 'remove':
#         rm_personal = True
#     else:
#         rm_personal = False
#
#     calc_scores(args.infile, args.gen_method, single=single_q, remove_personal=rm_personal,
#                 out_path=args.outfile, save_steps=args.save_steps)