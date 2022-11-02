import argparse
from tqdm import tqdm, trange

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from predict import set_device, load_pretrainModel, convert_input_file_to_RnnTensor, get_s2s_predict, convert_input_file_to_BertTensor, get_pretrain_predict, convert_to_labels
from utils import load_tokenizer, get_intent_labels, get_slot_labels, MODEL_CLASSES, get_word_vocab

def one_shot_inference(lines, model_dir):

    # loading train Model args
    args_path = os.path.join(pred_config.model_dir, 'train_args.bin')
    train_args = torch.load(args_path)

    pred_config = train_args
    pred_config.pad_label_id = 0
    device = set_device(pred_config)

    # load labels
    intent_vocab = get_intent_labels(train_args)
    slot_vocab = get_slot_labels(train_args)

    # load pretrain Model
    model = load_pretrainModel(
        pred_config, train_args, len(intent_vocab), len(slot_vocab))
    model.to(device)

    # Convert lines to TensorDataset
    pad_token_label_id = train_args.ignore_index

    pad_label_id = pred_config.pad_label_id

    all_slot_label_mask = None
    intent_preds = None
    slot_preds = None

    if pred_config.model_type.endswith('S2S'):
        # convert data to tensor
        tokenizer = get_word_vocab(train_args)
        dataset = convert_input_file_to_RnnTensor(lines, tokenizer, train_args,
                                                  pred_config, pad_token_id=pad_label_id)

        # get predict!
        intent_preds, slot_preds, slot_masks = get_s2s_predict(model, dataset, pred_config,
                                                               train_args, slot_vocab, device)
    elif pred_config.model_type.endswith('bert'):
        # convert data to tensor
        tokenizer = load_tokenizer(train_args)
        dataset = convert_input_file_to_BertTensor(lines, tokenizer, train_args,
                                                   pred_config, pad_label_id=pad_label_id)

        # get predict!
        intent_preds, slot_preds, slot_masks = get_pretrain_predict(model, dataset, pred_config,
                                                                    train_args, device)

    intent_labels, slot_labels = convert_to_labels(intent_vocab, slot_vocab,
                                                   intent_preds, slot_preds,
                                                   slot_masks, pad_label_id)

    #slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    #slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    #for i in range(slot_preds.shape[0]):
    #    for j in range(slot_preds.shape[1]):
    #        if all_slot_label_mask[i, j] != pad_token_label_id:
    #            slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    # Write to output file
    #with open(pred_config.output_file, "w", encoding="utf-8") as f:
    #    for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
    #        line = ""
    #        for word, pred in zip(words, slot_preds):
    #            if pred == 'O':
    #                line = line + word + " "
    #            else:
    #                line = line + "[{}:{}] ".format(word, pred)
    #        f.write("<{}> -> {}\n".format(intent_label_lst[intent_pred], line.strip()))

    #logger.info("Prediction Done!")

    for words, slot_preds in zip(lines, slot_labels):
        line = ""
        for word, pred in zip(words, slot_preds):
            if pred == 'O':
                line = line + word + " "
            else:
                line = line + "[{}:{}] ".format(word, pred)
        #f.write("<{}> -> {}\n".format(intent_label_lst[intent_pred], line.strip()))

    return intent_labels, line.strip()