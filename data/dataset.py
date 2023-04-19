import torch
from torch.utils.data import Dataset

import datasets

import fasttext.util

import os
from typing import List

LABEL_NUM = 28

label_names = {
    0: 'admiration',
    1: 'amusement',
    2: 'anger',
    3: 'annoyance',
    4: 'approval',
    5: 'caring',
    6: 'confusion',
    7: 'curiosity',
    8: 'desire',
    9: 'disappointment',
    10: 'disapproval',
    11: 'disgust',
    12: 'embarrassment',
    13: 'excitement',
    14: 'fear',
    15: 'gratitude',
    16: 'grief',
    17: 'joy',
    18: 'love',
    19: 'nervousness',
    20: 'optimism',
    21: 'pride',
    22: 'realization',
    23: 'relief',
    24: 'remorse',
    25: 'sadness',
    26: 'surprise',
    27: 'neutral'
}

chosen_labels = [0, 1, 2, 3, 7, 10, 15, 17, 18, 20, 24, 25, 26]

renumbered_labels = dict(zip(chosen_labels, list(range(len(chosen_labels)))))
renumbered_labels[27] = len(chosen_labels)

parent_labels_dir = "./go_emotions_labels"
labels_dir = f"{parent_labels_dir}/merged_labels.pt"

def build_masks():
    diag_mask = torch.ones((LABEL_NUM, LABEL_NUM)).type(torch.int)
    for i in range(LABEL_NUM):
        diag_mask[i,i] = 0

    unchosen_mask = torch.ones((LABEL_NUM, LABEL_NUM)).type(torch.int)
    unchosen_labels = list(range(LABEL_NUM))

    for label in chosen_labels:
        unchosen_labels.remove(label)

    for label in unchosen_labels:
        unchosen_mask[:,label].fill_(0)

    neutral_mask = torch.ones((LABEL_NUM, LABEL_NUM)).type(torch.int)
    neutral_mask[LABEL_NUM-1,:].fill_(0)
    neutral_mask[:,LABEL_NUM-1].fill_(0)

    return diag_mask, unchosen_mask, neutral_mask

def load_labels(train_labels: List[List[int]], valid_labels: List[List[int]], test_labels: List[List[int]]):
    train_split_point = len(train_labels)
    valid_split_point = train_split_point+len(valid_labels)

    data_labels = train_labels
    data_labels.extend(valid_labels)
    data_labels.extend(test_labels)


    relation_matrix = torch.zeros((LABEL_NUM, LABEL_NUM)).type(torch.int)

    for labels in data_labels:
        for main_label in labels:
            for relation_label in labels:
                relation_matrix[main_label, relation_label] += 1

    diag_mask, unchosen_mask, neutral_mask = build_masks()

    relation_matrix = relation_matrix * diag_mask * unchosen_mask * neutral_mask

    label_mappings = dict()

    for label in range(LABEL_NUM):
        if label in chosen_labels:
            label_mappings[label] = label
        else:
            label_mappings[label] = torch.argmax(relation_matrix[label]).item()
    label_mappings[27] = 27

    new_labels = []
    label_occurances = dict.fromkeys(chosen_labels, 0)
    label_occurances[27] = 0

    for labels in data_labels:
        min_label = label_mappings[labels[0]]
        for label in labels:
            mapped_label = label_mappings[label]
            if label_occurances[mapped_label] < label_occurances[min_label]:
                min_label = mapped_label
        
        label_occurances[min_label] += 1
        new_labels.append(renumbered_labels[min_label])

    new_labels = torch.tensor(new_labels)
    new_train_labels = new_labels[:train_split_point]
    new_valid_labels = new_labels[train_split_point:valid_split_point]
    new_test_labels = new_labels[valid_split_point:]

    return new_train_labels, new_valid_labels, new_test_labels

def labels_to_list(data):
    labels = []
    for row in data:
        labels.append(row['labels'])
    return labels

def save_labels():
    train_data = datasets.load_dataset('go_emotions', split="train")
    valid_data = datasets.load_dataset('go_emotions', split="validation")
    test_data = datasets.load_dataset('go_emotions', split="test")

    train_labels, valid_labels, test_labels = load_labels(
        labels_to_list(train_data),
        labels_to_list(valid_data),
        labels_to_list(test_data)
    )

    if not os.path.exists(parent_labels_dir):
        os.mkdir(parent_labels_dir)

    torch.save(
        {"train": train_labels,
         "valid": valid_labels,
         "test": test_labels}
         , labels_dir
    )

def totext(text):
    return text

class EmotionsDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()

        if not os.path.exists(labels_dir):
            print("No merged labels found locally, generting new labels now")
            save_labels()

        self.y = torch.load(labels_dir)[split].tolist()

        if split == 'valid':
            split = 'validation'
            
        data = datasets.load_dataset('go_emotions', split=split)

        self.x = []

        for row in data:
            self.x.append(row['text'])

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def download_model():
    if os.path.exists('./cc.en.300.bin'):
        return
    
    fasttext.util.download_model('en', if_exists='ignore')