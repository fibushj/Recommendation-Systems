from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import heapq
from non_personalized import *
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
    sim_scores = sim_scores[1:k+1]
    book_indices = [i[0] for i in sim_scores]
    return metadata['original_title'].iloc[book_indices]
#


print(get_contact_recommendation("Twilight", 10))

# s


ratings_path = 'books_data/ratings.csv'
books_path = 'books_data/books.csv'

# Reading ratings file:
r_cols = ['user_id', 'book_id', 'rating']
ratings = pd.read_csv(ratings_path)

# Reading items file:
i_cols = ['book_id', 'goodreads_book_id', 'best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13', 'authors',
          'original_publication_year', 'original_title', 'title', 'language_code', 'image_url', 'small_image_url']
items = pd.read_csv(books_path, encoding='latin-1')
items_map = items['book_id'].to_list()

print(ratings.head())
print(items.head())

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

# calc mean
mean_user_rating = np.nanmean(data_matrix, axis=1).reshape(-1, 1)

ratings_diff = (data_matrix - mean_user_rating)
# replace nan -> 0
ratings_diff[np.isnan(ratings_diff)] = 0

# calculate user x user similarity matrix
user_similarity = 1 - pairwise_distances(ratings_diff, metric='euclidean')
print(user_similarity.shape)


# For each user (i.e., for each row) keep only k most similar users, set the rest to 0.
# Note that the user has the highest similarity to themselves.
def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
    return arr


k = 10
user_similarity = np.array([keep_top_k(np.array(arr), k)
                            for arr in user_similarity])
print(user_similarity.shape)

# since n-k users have similarity=0, for each user only k most similar users contribute to the predicted ratings
pred = mean_user_rating + \
    user_similarity.dot(ratings_diff) / \
    np.array([np.abs(user_similarity).sum(axis=1)]).T


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(predicted_ratings_row, data_matrix_row, items, k=5):
    predicted_ratings_row[~np.isnan(data_matrix_row)] = 0
    # print(predicted_ratings_unrated)

    idx = np.argsort(-predicted_ratings_row)
    # print (idx)
    sim_scores = idx[0:k]
    # print(sim_scores)

    # Return top k movies
    return items[['book_id', 'title']].iloc[sim_scores]


def get_top_rated(data_matrix_row, items, k=20):
    srt_idx = np.argsort(-data_matrix_row)
    # print(~np.isnan(data_matrix_row[srt_idx]))
    srt_idx_not_nan = srt_idx[~np.isnan(data_matrix_row[srt_idx])]
    x = items[['book_id', 'title']].iloc[srt_idx_not_nan][:k]
    return x


new_user_id = 1
user = new_user_id - 1
predicted_ratings_row = pred[user]
data_matrix_row = data_matrix[user]

print("Top rated movies by test user:")
print(get_top_rated(data_matrix_row, items))

print('****** test user - user_prediction ******')
print(get_recommendations(predicted_ratings_row, data_matrix_row, items, k=10))


def build_CF_prediction_matrix(sim):
    """
    the method builds the prediction matrix.
    :param sim: the similarity measure
    :return:
    """
    pass


def get_CF_recommendation(user_id, k):
    """
    :param user_id: user id
    :param k: the amount of recommendations
    :return: top k recommended books for the user.
    """
    pass
