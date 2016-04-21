import cPickle
import requests
import json

if __name__=="__main__":
    x = cPickle.load(open("tweet.p", "rb"))
    tweet_data, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]

    total = 0
    correct = 0
    for datum in tweet_data:
        target = int(datum["y"])
        tweet = datum["text"]
        resp = requests.post("http://sentiment.vivekn.com/api/text/", data={"txt": tweet})
        json_data = json.loads(resp.text)
        pred = json_data['result']['sentiment']
        if pred == 'Positive':
            pred = 0
        elif pred == 'Neutral':
            pred = 1
        elif pred == 'Negative':
            pred = 2
        else:
            pred = -1
            print 'Wrong label'
        total += 1
        if pred == target:
            correct += 1

    print "Total Tweets: {}".format(total)
    print "Final Accuracy of NB: {}".format(float(correct) / float(total))
