from flask import Flask, request, jsonify
from flask_cors import CORS
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from threading import Lock
from functools import lru_cache
from typing import Generator, List, Dict, Any
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)
CORS(app)

DATA_FILE_PATH = "dataset.csv"
GOOGLE_BOOKS_API_KEY = ''

BOOKS_DF = pd.read_csv(DATA_FILE_PATH)


def search_titles_generator(prefix: str, dataframe: pd.DataFrame) -> Generator[str, None, None]:
    """
    Generate titles that match the given prefix from the dataframe.

    Args:
        prefix (str): The prefix to search for.
        dataframe (pd.DataFrame): The dataframe containing book titles and authors.

    Yields:
        str: A matching book title.
    """
    prefix = prefix.lower()
    titles_set = set()
    count = 0

    for title, author in zip(dataframe['Name'], dataframe['Authors']):
        if title.lower().startswith(prefix) and (title, author) not in titles_set:
            yield f"{title}, {author}"
            titles_set.add((title, author))
            count += 1
            if count >= 20:
                break


@app.route("/titles", methods=["POST"])
def get_titles() -> Dict[str, List[str]]:
    """
    Get matching titles based on a search prefix.

    Returns:
        Dict[str, List[str]]: A dictionary containing a list of matching titles.
    """
    try:
        data = request.json
        letters = data.get("search")

        if len(letters) > 3:
            matching_titles = list(search_titles_generator(letters, BOOKS_DF))
            return jsonify({"titles": matching_titles})
        else:
            return jsonify({"titles": []})

    except Exception as e:
        return jsonify({"Error": str(e)})


@app.route("/exist", methods=["POST"])
def book_exist() -> Dict[str, bool]:
    """
    Check if a book exists in the dataset.

    Returns:
        Dict[str, bool]: A dictionary indicating whether the book exists.
    """
    try:
        data = request.json
        author = data.get("author")
        title = data.get("title")

        similarities = BOOKS_DF["Name"].apply(lambda x: fuzz.ratio(x, title))
        similar_titles = BOOKS_DF.loc[similarities >= 80]

        if not similar_titles.empty:
            author_similarities = similar_titles["Authors"].apply(lambda x: fuzz.ratio(x, author))
            similar_authors = similar_titles.loc[author_similarities >= 90]

            if not similar_authors.empty:
                return jsonify({"exist": True})
        
        return jsonify({"exist": False})
    except Exception as e:
        return jsonify({"Error": str(e)})


class RecommendationAPI:
    """
    Recommendation API class.
    """
    def __init__(self) -> None:
        self.user_item_matrix = None
        self.book_titles = None
        self.user_similarity = None
        self.new_user_id = None
        self.book_authors = None
        self.file_path = DATA_FILE_PATH
        self.lock = Lock()
        self.load_data()

    def load_data(self) -> None:
        """
        Load data from the file path.
        """
        with self.lock:
            if self.user_item_matrix is None or self.book_titles is None:
                df = pd.read_csv(self.file_path)
                self.user_item_matrix = df.pivot_table(index='ID', columns='Id', values='Numeric_Rating', aggfunc='mean').fillna(0)
                self.book_titles = {book_id: title for book_id, title in zip(df['Id'], df['Name'])}
                self.book_authors = {book_id: title for book_id, title in zip(df['Id'], df['Authors'])}

    def get_max_id(self) -> None:
        """
        Get the maximum user ID.
        """
        with self.lock:
            self.new_user_id = self.user_item_matrix.index.max() + 1

    def add_new_user(self, user_input_data: List[Dict[str, Any]]) -> None:
        """
        Add a new user to the user-item matrix.

        Args:
            user_input_data (List[Dict[str, Any]]): Data containing user input.
        """
        with self.lock:
            user_input_df = pd.DataFrame(user_input_data)
            temp_user_ratings = {}

            if self.new_user_id not in self.user_item_matrix.index:
                self.user_item_matrix.loc[self.new_user_id] = 0

            for _, row in user_input_df.iterrows():
                input_title = row['title']
                similar_titles = self.find_similar_titles(input_title)

                for book_id in similar_titles.keys():
                    if book_id not in self.user_item_matrix.columns:
                        self.user_item_matrix[book_id] = 0

                    if self.user_item_matrix.at[self.new_user_id, book_id] == 0:
                        temp_user_ratings[book_id] = row['mark']

            for book_id, rating in temp_user_ratings.items():
                self.user_item_matrix.at[self.new_user_id, book_id] = rating

    @lru_cache(maxsize=None)
    def find_similar_titles(self, input_title: str) -> Dict[int, str]:
        """
        Find similar titles based on the input title.

        Args:
            input_title (str): The input title to find similarities.

        Returns:
            Dict[int, str]: A dictionary containing similar titles and their IDs.
        """
        similarities = {book_id: fuzz.ratio(title, input_title) for book_id, title in self.book_titles.items()}
        similar_titles = {book_id: title for book_id, title in similarities.items() if title >= 80}
        return similar_titles

    def calculate_similarity(self) -> None:
        """
        Calculate user similarity.
        """
        with self.lock:
            scaled_user_item_matrix = csr_matrix(self.user_item_matrix.values)
            self.user_similarity = cosine_similarity(scaled_user_item_matrix)

            if self.new_user_id not in self.user_item_matrix.index:
                self.user_similarity = np.concatenate([self.user_similarity, np.zeros((1, self.user_similarity.shape[1]))], axis=0)
                self.user_similarity[-1, :] = 0

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations for the new user.

        Returns:
            List[Dict[str, Any]]: A list of recommended books.
        """
        with self.lock:
            user_similarity_index = self.user_item_matrix.index.astype(int)
            user_to_index_mapping = {user_id: index for index, user_id in enumerate(user_similarity_index)}
            user_index = user_to_index_mapping[self.new_user_id]

            similar_users = self.user_similarity[user_index].argsort()[::-1][1:10 + 1]
            similar_users_ratings = self.user_item_matrix.iloc[similar_users].mean(axis=0)

            unrated_books = self.user_item_matrix.loc[self.new_user_id][self.user_item_matrix.loc[self.new_user_id] == 0].index
            unrated_books = [book_id for book_id in unrated_books if self.user_item_matrix.at[self.new_user_id, book_id] == 0]

            unique_recommendations = []
            recommended_titles = set()

            for book_id in similar_users_ratings[unrated_books].sort_values(ascending=False).index:
                title = self.book_titles[book_id]

                if title not in recommended_titles:
                    author = self.book_authors[book_id]
                    book_info = self.get_book_info(title, author)
                    unique_recommendations.append(
                        book_info
                    )
                    recommended_titles.add(title)

                if len(unique_recommendations) == 5:
                    break

            return unique_recommendations

    def run_api(self) -> None:
        """
        Run the Flask API.
        """
        @app.route('/recommendations', methods=['POST'])
        def get_recommendations() -> Dict[str, List[Dict[str, Any]]]:
            data = request.get_json()

            if not data or not isinstance(data, list):
                return jsonify({'error': 'Invalid JSON format'}), 400

            self.get_max_id()

            self.add_new_user(data)
            self.calculate_similarity()
            recommendations = self.generate_recommendations()

            return jsonify({'recommendations': recommendations})

        app.run(debug=True, port=5000, threaded=True)

    def get_book_info(self, book_title: str, author: str) -> Dict[str, Any]:
        """
        Get information about a book from the Google Books API.

        Args:
            book_title (str): The title of the book.
            author (str): The author of the book.

        Returns:
            Dict[str, Any]: Information about the book.
        """
        formatted_title = book_title.replace(' ', '+')
        formatted_author = author.replace(' ', '+')
        url = f'https://www.googleapis.com/books/v1/volumes?q={formatted_title}+inauthor:{formatted_author}&langRestrict=en&key={GOOGLE_BOOKS_API_KEY}'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if 'items' in data and len(data['items']) > 0:
                first_book_info = data['items'][0]['volumeInfo']
                authors = first_book_info.get('authors', ['Unknown Author'])
                cover_image = first_book_info.get('imageLinks', {}).get('thumbnail', 'No cover image available')

                return {
                    'title': first_book_info.get('title', 'Unknown Title'),
                    'authors': authors,
                    'description': first_book_info.get('description', 'No description available'),
                    'cover_image': cover_image
                }

        return {
            'title': 'Unknown Title',
            'authors': ['Unknown Author'],
            'description': 'No description available',
            'cover_image': 'No cover image available'
        }


if __name__ == '__main__':
    recommendation_api = RecommendationAPI()
    recommendation_api.run_api()
