## Book Recommendation System API
This project provides a book recommendation system API built using Flask. The system allows users to search for book titles, check if a book exists in the dataset, and receive personalized book recommendations based on their preferences.

## Features
- Search Titles: Users can search for book titles by providing a search prefix. The API returns matching titles from the dataset.
- Check Book Existence: Users can check if a book exists in the dataset by providing the book's title and author.
- Personalized Recommendations: Users can receive personalized book recommendations based on their input ratings for selected titles.

##Requirements
Python 3.x
Flask
Flask-CORS
FuzzyWuzzy
SciPy
scikit-learn
Pandas
NumPy
Requests

## Setup
1. Clone the repository:
```bash
git clone 
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Obtain a Google Books API key and update the GOOGLE_BOOKS_API_KEY variable in app.py.
4. Provide the path to your dataset CSV file by updating the DATA_FILE_PATH variable in app.py.
5. Run the Flask server:
```bash
python app.py 
```

## Endpoints
- /titles (POST): Search for book titles.
  - Request Body: {"search": "prefix"}
  - Response Body: {"titles": ["Title 1, Author 1", "Title 2, Author 2", ...]}
    
- /exist (POST): Check if a book exists.
  - Request Body: {"title": "book_title", "author": "book_author"}
  - Response Body: {"exist": true/false}
  - 
- /recommendations (POST): Get personalized recommendations.
  - Request Body: [{"title": "book_title", "mark": rating}, {"title": "book_title", "mark": rating}, ...]
  - Response Body: {"recommendations": [{"title": "book_title", "authors": ["Author 1"], "description": "Book description", "cover_image": "image_url"}, ...]}

## Dataset
The dataset used in this project contains information about books, including titles, authors, and numeric ratings. You can find the dataset in the CSV file format.

## Notebooks
For creating the dataset or conducting data analysis, you can refer to the Jupyter notebook provided in the repository.

## License
This project is licensed under the [MIT License](LICENSE).
