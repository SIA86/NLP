def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels 

labels = [3, 0, 7, 0, 0, 0, 7, 0, 0]
word_ids = [None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]
print(labels)
print(align_labels_with_tokens(labels, word_ids))