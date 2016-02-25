import csv
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import TweetTokenizer


def prepare_amazon():
    with open("./data/amazon-fine-foods/Reviews.csv", 'rU') as fin:
        with open("./data/amazon-fine-foods/Reviews_with_pos.csv", 'w') as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, ['Text_POS', 'Score'])
            pos_tagger = StanfordPOSTagger(
                    './data/pos-tag/english-bidirectional-distsim.tagger',
                    './data/pos-tag/stanford-postagger.jar')
            tokenizer = TweetTokenizer(reduce_len=True)

            writer.writeheader()
            for row in reader:
                review = row['Text']
                score = row['Score']
                processed_review = pos_tagger.tag(tokenizer.tokenize(review))  # list of tuples
                processed_review = [(token.encode("utf-8"), tag.encode("utf-8")) for (token, tag) in processed_review]
                writer.writerow({'Text_POS': processed_review, 'Score': score})


if __name__ == '__main__':
    prepare_amazon()
