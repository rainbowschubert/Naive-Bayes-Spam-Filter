import csv
import re
from collections import defaultdict
import math


# splits csv file into individual words
def tokenize(message):
    message = message.lower()
    all_words = re.findall("[a-z0-9']+", message)
    return set(all_words)


# counts number of times specific words are in spam/non-spam messages
def count_words(training_set):
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam == "spam" else 1] += 1
    return counts


# returns probabilities of each word being in a spam/non-spam message (with pseudocount k)
def word_probabilities(counts, total_spams, total_non_spams, k=1):
    return [(w, (spam + k) / (total_spams + 2 * k), (non_spam + k) / (total_non_spams + 2 * k))
            for w, (spam, non_spam) in counts.items()]


# returns probability that message is spam
def spam_probability(word_probs, message, spam_prob, not_spam_prob):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0
    for word, prob_if_spam, prob_if_not_spam in word_probs:
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam * spam_prob / (prob_if_spam * spam_prob + prob_if_not_spam * not_spam_prob)


with open("spam.csv", mode="r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    ham_or_spam = []
    text_messages = []
    temp = 0
    for row in csv_reader:
        if temp != 0:
            ham_or_spam.append(row[0])
            text_messages.append(row[1])
        temp += 1

train_data = zip(text_messages[0:4459], ham_or_spam[0:4459])
test_data = zip(text_messages[4459:], ham_or_spam[4459:])
ham_count = ham_or_spam[0:4459].count("ham")
spam_count = ham_or_spam[0:4459].count("spam")
counts = count_words(train_data)
probabilities = word_probabilities(counts, spam_count, ham_count)

message = input("Enter a message: ")

predicted_probability = spam_probability(probabilities, message, spam_count / 4459, ham_count / 4459)
print("spam" if predicted_probability > 0.5 else "not spam")
