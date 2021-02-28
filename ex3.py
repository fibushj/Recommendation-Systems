#########################
# PART 1
#########################

import pandas as pd


def get_simply_recommendation(k):
    ratings = pd.read_csv('ratings.csv', low_memory=False)
    books = pd.read_csv('books.csv', low_memory=False,
                        encoding="ISO-8859-1")
    temp = pd.Series(ratings.groupby('book_id')['rating'].mean(), name='vote_average')
    books = books.merge(temp, on='book_id', how='left')
    C = books['vote_average'].mean()
    temp = pd.Series(ratings.groupby('book_id')['rating'].count(), name='vote_count')
    books = books.merge(temp, on='book_id', how='left')
    m = books['vote_count'].quantile(0.90)
    q_books = books.copy().loc[books['vote_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * C)

    q_books['score'] = q_books.apply(weighted_rating, axis=1)
    q_books = q_books.sort_values('score', ascending=False)

    return q_books[['book_id', 'title', 'score']].head(k)


def get_simply_place_recommendation(place, k):
    users = pd.read_csv('users.csv', low_memory=False)
    ratings = pd.read_csv('ratings.csv', low_memory=False)
    books = pd.read_csv('books.csv', low_memory=False,
                        encoding="ISO-8859-1")
    users = users.loc[users['location'] == place]
    ratings = ratings.merge(users, on='user_id')
    temp = pd.Series(ratings.groupby('book_id')['rating'].mean(), name='vote_average')
    books = books.merge(temp, on='book_id', how='left')
    C = books['vote_average'].mean()
    temp = pd.Series(ratings.groupby('book_id')['rating'].count(), name='vote_count')
    books = books.merge(temp, on='book_id', how='left')
    m = books['vote_count'].quantile(0.90)
    q_books = books.copy().loc[books['vote_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * C)

    q_books['score'] = q_books.apply(weighted_rating, axis=1)
    q_books = q_books.merge(ratings, on='book_id').sort_values('score', ascending=False)
    q_books = q_books[['book_id', 'title', 'score']].drop_duplicates().head(k)

    return q_books


def get_simply_age_recommendation(age, k):
    range_bottom = age - age % 10
    range_up = range_bottom + 10
    ratings = pd.read_csv('ratings.csv', low_memory=False)
    books = pd.read_csv('books.csv', low_memory=False,
                        encoding="ISO-8859-1")
    users = pd.read_csv('users.csv', low_memory=False)
    users = users.loc[(users['age'] > range_bottom) & (users['age'] <= range_up)]
    ratings = ratings.merge(users, on='user_id')
    temp = pd.Series(ratings.groupby('book_id')['rating'].mean(), name='vote_average')
    books = books.merge(temp, on='book_id', how='left')
    C = books['vote_average'].mean()
    temp = pd.Series(ratings.groupby('book_id')['rating'].count(), name='vote_count')
    books = books.merge(temp, on='book_id', how='left')
    m = books['vote_count'].quantile(0.90)
    q_books = books.copy().loc[books['vote_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * C)

    q_books['score'] = q_books.apply(weighted_rating, axis=1)
    q_books = q_books.merge(ratings, on='book_id').sort_values('score', ascending=False)
    q_books = q_books[['book_id', 'title', 'score']].drop_duplicates().head(k)

    return q_books

# print(get_simply_recommendation(10))
# print(get_simply_place_recommendation("Ohio", 10))
# print(get_simply_age_recommendation(28, 10))

#########################
# END OF PART 1
#########################


#########################
# PART 3
#########################
import pandas as pd


def build_contact_sim_metrix():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    book_tags = pd.read_csv('books_tags.csv', low_memory=False)
    tags = pd.read_csv('tags.csv', low_memory=False)
    metadata = pd.read_csv('books.csv', low_memory=False,
                           encoding="ISO-8859-1")

    book_tags['tag_id'] = book_tags['tag_id'].astype(int)
    tags['tag_id'] = tags['tag_id'].astype(int)
    metadata['goodreads_book_id'] = metadata['goodreads_book_id'].astype(int)
    book_tags['goodreads_book_id'] = book_tags['goodreads_book_id'].astype(int)

    book_tags = book_tags.merge(tags, on='tag_id').groupby(
        'goodreads_book_id')['tag_name'].apply(list)
    metadata = metadata.merge(book_tags, on='goodreads_book_id', how='left')

    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    features = ['tag_name', 'authors', 'language_code']
    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)

    def create_soup(x):
        return ' '.join(x['tag_name']) + ' ' + x['authors'] + ' ' + x['language_code'] + ' ' + str(x['original_title'])

    metadata['soup'] = metadata.apply(create_soup, axis=1)

    from sklearn.feature_extraction.text import CountVectorizer

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['soup'])

    from sklearn.metrics.pairwise import cosine_similarity

    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    metadata = metadata.reset_index()
    return cosine_sim, metadata


def get_contact_recommendation(book_name, k):
    cosine_sim, metadata = build_contact_sim_metrix()
    indices = pd.Series(metadata.index, index=metadata['original_title'])
    idx = indices[book_name]
    if isinstance(indices[book_name], pd.core.series.Series):
        idx = indices[book_name][0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k + 1]
    book_indices = [i[0] for i in sim_scores]
    return metadata['original_title'].iloc[book_indices]


#########################
# END OF PART 3
#########################

import heapq

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

ratings_path = 'ratings.csv'
books_path = 'books.csv'

# Reading ratings file:
r_cols = ['user_id', 'book_id', 'rating']
ratings = pd.read_csv(ratings_path)

# Reading items file:
i_cols = ['book_id', 'goodreads_book_id', 'best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13', 'authors',
          'original_publication_year', 'original_title', 'title', 'language_code', 'image_url', 'small_image_url']
items = pd.read_csv(books_path, encoding='latin-1')
items_map = items['book_id'].to_list()

# print(ratings.head())
# print(items.head())

# calculate the number of unique users and movies.
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.book_id.unique().shape[0]

# create ranking table - that table is sparse
# data_matrix = np.empty((n_users, 5060))
data_matrix = np.empty((n_users, n_items))

data_matrix[:] = np.nan
for line in ratings.itertuples():
    user = line[1] - 1
    # book = line[2] - 1
    book = items_map.index(line[2])
    rating = line[3]
    data_matrix[user, book] = rating


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(predicted_ratings_row, data_matrix_row, items, k=10):
    predicted_ratings_row[~np.isnan(data_matrix_row)] = 0
    # print(predicted_ratings_unrated)

    idx = np.argsort(-predicted_ratings_row)
    # print (idx)
    sim_scores = idx[0:k]
    # print(sim_scores)

    return items[['book_id', 'title']].iloc[sim_scores]


def get_top_rated(data_matrix_row, items, k=20):
    srt_idx = np.argsort(-data_matrix_row)
    # print(~np.isnan(data_matrix_row[srt_idx]))
    srt_idx_not_nan = srt_idx[~np.isnan(data_matrix_row[srt_idx])]
    x = items[['book_id', 'title']].iloc[srt_idx_not_nan][:k]
    return x


def build_CF_prediction_matrix(sim):
    """
    the method builds the prediction matrix.
    :param sim: the similarity measure
    :return:
    """
    # calc mean
    mean_user_rating = np.nanmean(data_matrix, axis=1).reshape(-1, 1)

    ratings_diff = (data_matrix - mean_user_rating)
    # replace nan -> 0
    ratings_diff[np.isnan(ratings_diff)] = 0

    # calculate user x user similarity matrix
    if sim == 'jaccard':
        user_similarity = 1 - pairwise_distances(np.array(ratings_diff, dtype=bool), metric=sim)
    else:
        user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)
    # print(user_similarity.shape)

    # For each user (i.e., for each row) keep only k most similar users, set the rest to 0.
    # Note that the user has the highest similarity to themselves.
    def keep_top_k(arr, k):
        smallest = heapq.nlargest(k, arr)[-1]
        arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
        return arr

    k = 10  # todo - is it right?
    user_similarity = np.array([keep_top_k(np.array(arr), k) for arr in user_similarity])
    #print(user_similarity.shape)

    # since n-k users have similarity=0, for each user only k most similar users contribute to the predicted ratings
    pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    return pred


pred = build_CF_prediction_matrix('cosine')
pred1 = build_CF_prediction_matrix('euclidean')
pred2 = build_CF_prediction_matrix('jaccard')


def get_CF_recommendation(user_id, k):
    """
    :param user_id: user id
    :param k: the amount of recommendations
    :return: top k recommended books for the user and their ratings.
    """
    new_user_id = user_id
    user = new_user_id - 1
    predicted_ratings_row = pred[user]
    data_matrix_row = data_matrix[user]

    x = get_recommendations(predicted_ratings_row, data_matrix_row, items, k=k)

    # print('****** test user - user_prediction ******')
    # print(x)

    # return [row.title for index, row in x.iterrows()]
    return x


def get_CF_recommendation3(user_id, k):
    """
    :param user_id: user id
    :param k: the amount of recommendations
    :return: top k recommended books for the user and their ratings.
    """
    new_user_id = user_id
    user = new_user_id - 1
    predicted_ratings_row = pred[user]
    data_matrix_row = data_matrix[user]

    x = get_recommendations(predicted_ratings_row, data_matrix_row, items, k=k)

    # print('****** test user - user_prediction ******')
    # print(x)

    return x


def get_CF_recommendation1(user_id, k):
    """
    :param user_id: user id
    :param k: the amount of recommendations
    :return: top k recommended books for the user and their ratings.
    """
    new_user_id = user_id
    user = new_user_id - 1
    predicted_ratings_row = pred1[user]
    data_matrix_row = data_matrix[user]

    x = get_recommendations(predicted_ratings_row, data_matrix_row, items, k=k)

    # print('****** test user - user_prediction ******')
    # print(x)

    return x


def get_CF_recommendation2(user_id, k):
    """
    :param user_id: user id
    :param k: the amount of recommendations
    :return: top k recommended books for the user and their ratings.
    """
    new_user_id = user_id
    user = new_user_id - 1
    predicted_ratings_row = pred2[user]
    data_matrix_row = data_matrix[user]

    x = get_recommendations(predicted_ratings_row, data_matrix_row, items, k=k)

    # print('****** test user - user_prediction ******')
    # print(x)

    return x


def filter_data(k=10):
    df = pd.read_csv("test.csv")
    dict = {}
    for index, row in df.iterrows():
        if row.rating >= 4:
            if row.user_id not in dict:
                dict[row.user_id] = []
                dict[row.user_id].append(row.book_id)
            else:
                dict[row.user_id].append(row.book_id)
    updated_dict = {}
    for user_id in dict:
        if len(dict[user_id]) >= k:
            updated_dict[user_id] = dict[user_id]
    return updated_dict


def rec_list(df):
    return [row.book_id for index, row in df.iterrows()]


def precision_k(k):
    user_ratings = filter_data(k)
    hits = 0
    hits1 = 0
    hits2 = 0
    for user_id in user_ratings:
        hits += len(list(set(rec_list(get_CF_recommendation3(user_id, k))) & set(user_ratings[user_id])))
        hits1 += len(list(set(rec_list(get_CF_recommendation1(user_id, k))) & set(user_ratings[user_id])))
        hits2 += len(list(set(rec_list(get_CF_recommendation2(user_id, k))) & set(user_ratings[user_id])))
    # print("nailed it\n")
    return [hits / (k * len(user_ratings)), hits1 / (k * len(user_ratings)), hits2 / (k * len(user_ratings))]


def ARHR(k):
    user_ratings = filter_data(k)
    sum = 0
    sum1 = 0
    sum2 = 0
    users_amount = len(user_ratings)
    for user_id in user_ratings:
        recommendations_list = rec_list(get_CF_recommendation3(user_id, k))
        recommendations_list1 = rec_list(get_CF_recommendation1(user_id, k))
        recommendations_list2 = rec_list(get_CF_recommendation2(user_id, k))
        i = 1
        for book_id in recommendations_list:
            if book_id in user_ratings[user_id]:
                sum += 1 / i
            i += 1
        i = 1
        for book_id in recommendations_list1:
            if book_id in user_ratings[user_id]:
                sum1 += 1 / i
            i += 1
        i = 1
        for book_id in recommendations_list2:
            if book_id in user_ratings[user_id]:
                sum2 += 1 / i
            i += 1
    return [sum / users_amount, sum1 / users_amount, sum2 / users_amount]


def RMSE():
    def predict_rating(user_id, book_id):
        new_user_id = user_id
        user = new_user_id - 1
        predicted_ratings_row = pred[user]
        data_matrix_row = data_matrix[user]
        predicted_ratings_row[~np.isnan(data_matrix_row)] = 0
        return predicted_ratings_row[items_map.index(book_id)]

    def predict_rating1(user_id, book_id):
        new_user_id = user_id
        user = new_user_id - 1
        predicted_ratings_row = pred1[user]
        data_matrix_row = data_matrix[user]
        predicted_ratings_row[~np.isnan(data_matrix_row)] = 0
        return predicted_ratings_row[items_map.index(book_id)]

    def predict_rating2(user_id, book_id):
        new_user_id = user_id
        user = new_user_id - 1
        predicted_ratings_row = pred2[user]
        data_matrix_row = data_matrix[user]
        predicted_ratings_row[~np.isnan(data_matrix_row)] = 0
        return predicted_ratings_row[items_map.index(book_id)]

    df = pd.read_csv("test.csv")
    sum = 0
    sum1 = 0
    sum2 = 0
    count = 0
    count1 = 0
    count2 = 0
    for index, row in df.iterrows():
        book_id = row.book_id
        user_id = row.user_id
        rating = row.rating
        predicted = predict_rating(user_id, book_id)
        sum += pow(predicted - rating, 2)
        count += 1
    for index, row in df.iterrows():
        book_id = row.book_id
        user_id = row.user_id
        rating = row.rating
        predicted = predict_rating1(user_id, book_id)
        sum1 += pow(predicted - rating, 2)
        count1 += 1
    for index, row in df.iterrows():
        book_id = row.book_id
        user_id = row.user_id
        rating = row.rating
        predicted = predict_rating2(user_id, book_id)
        sum2 += pow(predicted - rating, 2)
        count2 += 1
    return [pow(sum / count, 0.5), pow(sum1 / count1, 0.5), pow(sum2 / count2, 0.5)]


# def main():
    # x = get_CF_recommendation(1, 10)
    # print(x)
    # print("the len is ", len(filter_data()))
    # print("cosine   euclidean    jaccard \n")
    # print("precision k is ", precision_k(10))
    # print("ARHR K is ", ARHR(10))
    # print("RSME is ", RMSE())
    # pass


# if __name__ == '__main__':
#     main()
