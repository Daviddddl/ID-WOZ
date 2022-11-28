from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf(sentences: list):
    tfidf = TfidfVectorizer()

    response = tfidf.fit_transform(sentences)
    feature_names = tfidf.get_feature_names()

    tfidf_dict = dict()

    for col in response.nonzero()[1]:
        tfidf_dict[feature_names[col]] = float(response[0, col])

    sorted_dict = sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True)

    return sorted_dict

