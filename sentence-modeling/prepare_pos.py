import csv


def prepare_amazon(num_split):
    with open("./data/amazon-fine-foods/Reviews.csv", 'rU') as f:
        dataset = list(csv.reader(f))[1:]  # remove header
    text = [line[9].strip() for line in dataset]

    num_lines_processed = 0
    for sentence in text:
        if num_lines_processed % (len(text)/num_split) == 0:
            file_num = num_lines_processed / (len(text)/num_split)
            f = open("./data/amazon-fine-foods/Reviews_prep_for_pos_{}.txt".format(file_num), 'w')
        f.write(sentence + '\n')
        num_lines_processed += 1


if __name__ == '__main__':
    prepare_amazon(5)
