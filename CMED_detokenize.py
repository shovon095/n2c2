import math

import pandas as pd
import numpy as np


def detokenize(pred_token_test_path, pred_label_test_path):
    """
    convert suub-word level BioBERT-NER results to full words and labels.

    Args:
        pred_token_test_path: path to token_test.txt from output folder. ex) output/token_test.txt
        pred_label_test_path: path to label_test.txt from output folder. ex) output/label_test.txt
    Outs:
        A dictionary that contains full words and predicted labels.
    """

    # read predicted
    pred = {'toks': [], 'labels': []}  # dictionary for predicted tokens and labels.
    with open(pred_token_test_path, 'r') as in_tok, open(pred_label_test_path, 'r') as in_lab:  # 'token_test.txt'
        for lineIdx, (lineTok, lineLab) in enumerate(zip(in_tok, in_lab)):
            lineTok = lineTok.strip()
            pred['toks'].append(lineTok)

            lineLab = lineLab.strip()
            if lineLab in ['[CLS]', '[SEP]', 'X']:  # replace non-text tokens with O. These will not be evaluated.
                pred['labels'].append('O')
                continue
            pred['labels'].append(lineLab)

    assert (len(pred['toks']) == len(
        pred['labels'])), "Error! : len(pred['toks'])(%s) != len(pred['labels'])(%s) : Please report us " % (
    len(pred['toks']), len(pred['labels']))

    bert_pred = {'toks': [], 'labels': [], 'sentence': []}
    buf = []
    for t, l in zip(pred['toks'], pred['labels']):
        if t in ['[CLS]', '[SEP]']:  # non-text tokens will not be evaluated.
            bert_pred['toks'].append(t)
            bert_pred['labels'].append(t)  # Tokens and labels should be identical if they are [CLS] or [SEP]
            if t == '[SEP]':
                bert_pred['sentence'].append(buf)
                buf = []
            continue
        elif t[:2] == '##':  # if it is a piece of a word (broken by Word Piece tokenizer)
            bert_pred['toks'][-1] += t[2:]  # append pieces to make a full-word
            buf[-1] += t[2:]
        else:
            bert_pred['toks'].append(t)
            bert_pred['labels'].append(l)
            buf.append(t)

    assert (len(bert_pred['toks']) == len(bert_pred['labels'])), (
        "Error! : len(bert_pred['toks']) != len(bert_pred['labels']) : Please report us")

    return bert_pred
def Create_predict_annotation(data_doc_dir, tokens, tag_pred, output_dir):

    predict_annotation = [];
    Labels = ["B-Drug", "I-Drug"];
    #Labels = ["B-Disposition", "I-Disposition", "B-NoDisposition", "I-NoDisposition", "B-Undetermined", "I-Undetermined"]
    Record_ID_Flag = "RecordID";
    num_predictions = len(tokens);

    for token_index in range(num_predictions):
        #Check to see which record predictions are from
        if  type(tokens[token_index]) == str and  Record_ID_Flag in tokens[token_index]:
            ID_token = tokens[token_index];

            Record_ID = ID_token[8:11] + "-" + ID_token[11:];
            Record_path = data_doc_dir + Record_ID + ".txt";
            Record = open(Record_path, 'r').read();

            Output_path = output_dir + Record_ID + ".ann";
            Output_ann = open(Output_path, 'w');

            Term_index = 1;
            token_record_start_index = 0;
            token_record_end_index = 0;

        if tag_pred[token_index][0] == "B":
            entity = tokens[token_index];
            token_record_start_index = Record.index(tokens[token_index], token_record_end_index);
            token_record_end_index = token_record_start_index + len(tokens[token_index]);
            if tag_pred[token_index + 1] == "I":
                continue;
        elif tag_pred[token_index][0] == "I":
            entity = entity + " " + tokens[token_index];
            #add one to account for space between words
            token_record_end_index += (1 + len(tokens[token_index]));
            if tag_pred[token_index + 1][0] == "I":
                continue;
        else: continue;

        Annotation = "T" + str(Term_index) + "\t" + "Drug " + str(token_record_start_index) + " " + str(token_record_end_index) + "\t" + entity + "\n";
        Output_ann.write(Annotation);
        #predict_annotation.append(Annotation);
        Term_index += 1;

    return 0;

def main():
    #path containing predictions
    predicition_dir = "C:/Users/Tariq/OneDrive - Prairie View A&M University/PVAMU Research/n2c2_2022_Project/Test_Data_Clinical_Allnotes25/";
    #path with correct answers
    test_data_dir = "C:/Users/Tariq/OneDrive - Prairie View A&M University/PVAMU Research/n2c2_2022dataset_v3/trainingdata_v3/dev/";
    #path for annotated predictions
    output_dir = predicition_dir + "ann_pred/";

    predict_tokens_path = predicition_dir + "token_test.txt";
    predict_labels_path = predicition_dir + "label_test.txt";
    bert_pred = detokenize(predict_tokens_path, predict_labels_path);
    tokens = bert_pred["toks"];
    labels = bert_pred["labels"];


    Create_predict_annotation(test_data_dir, tokens, labels, output_dir);



    return 0;



if __name__ == '__main__':
    main()