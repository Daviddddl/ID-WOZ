import Levenshtein


def cer(s1, s2):
    """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Levenshtein.distance(s1, s2)


def wer(s1, s2):
    """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    return Levenshtein.distance(''.join(w1), ''.join(w2)) / float(len(w1))


if __name__ == '__main__':
    ground_truth = "hello world i like monhty python what do you mean african or european swallow"
    hypothesis = "hello i like python what you mean swallow"

    print(wer(ground_truth, hypothesis))
    print(cer(ground_truth, hypothesis))
