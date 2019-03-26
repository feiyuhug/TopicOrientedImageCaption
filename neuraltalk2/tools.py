
fo = open('./captionsample_beam1.txt', 'w')
with open('./log_training_idxcaptionsample_beam1.txt') as f:
    for line in f:
        items = line.strip().split()
        if len(items) == 0 :
            continue
        if items[0] == 'image' :
            fo.write(line.strip().split(':')[1].strip() + '\n')

fo.close()

