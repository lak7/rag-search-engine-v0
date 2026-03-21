from lib.load_data import load_data

def search_query(query):
    results = []
    movies = load_data()
    for movie in movies:
        if query.lower() in movie["title"].lower():
            results.append(movie["title"])
    return results
    

