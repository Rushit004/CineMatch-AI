import time
import pickle

similarity = pickle.load(open('similarity.pkl', 'rb'))
movies = pickle.load(open('movies.pkl', 'rb'))

def get_recommendations(title):
    idx = movies[movies['title'] == title].index[0]
    distances = similarity[idx]
    return sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

# average over many runs — a single call is too noisy to trust
n_runs = 1000
start = time.perf_counter()
for _ in range(n_runs):
    get_recommendations('Crank')  # swap in a real title from your dataset
end = time.perf_counter()

print(f"Average lookup time: {(end - start) / n_runs * 1000:.3f} ms")