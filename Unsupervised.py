from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.io import loadmat
from pandas import read_csv, read_excel
import numpy as np
import matplotlib.pyplot as plt

while True:
    try:
        # First: Import the data file location
        print('This program accepts only Excel, text, and MATLAB files.')
        file_type = input('Enter the type of file (Excel, text, or MATLAB):\n').strip().lower()

        # 1- Text files:
        if file_type == 'text':
            print('The text file type was chosen.')
            path = input("Enter the absolute path of the file (Ctrl+Shift+C to copy path):\n")
            data = read_csv(rf'{path[1 : len(path) - 1]}')
            X = np.array(data)

        # 2- Excel files:
        elif file_type == 'excel':
            print('The Excel file type was chosen.')
            path = input("Enter the absolute path of the file (Ctrl+Shift+C to copy path):\n")
            data = read_excel(rf'{path[1 : len(path) - 1]}')
            X = np.array(data)

        # 3- MATLAB files:
        elif file_type == 'matlab':
            print('The MATLAB file type was chosen.')
            path = input("Enter the absolute path of the file (Ctrl+Shift+C to copy path):\n")
            data = loadmat(rf'{path[1 : len(path) - 1]}')
            X = data['X']
            X = np.array(X)

        else:
            raise ValueError('Invalid file type chosen.')

    except ValueError as e:
        print(f"There was an issue: {e}. Please try again.")
        continue

    # Finally block for displaying data
    finally:
        while True:
            choice = input('Want to see the data you entered? (Yes/No):\n').strip().capitalize()
            if choice == 'Yes':
                print('The unlabeled data is:\n', X)
                print('Program will continue executing...')
                break
            elif choice == 'No':
                print('Program will continue executing...')
                break
            else:
                print('Choice not understood. Please enter Yes or No.')

    # KMeans Clustering
    no_clusters = int(input("Enter the number of clusters you want to use in the program:\n"))

    # Initialize centroids
    initial_centroids = np.zeros((no_clusters, X.shape[1]))# Set a seed to make the random selection reproducible
    np.random.seed(42)  # You can choose any seed value you prefer
    random_choice = np.random.choice(X.shape[0], no_clusters, replace=False)
    for i in range(no_clusters):
        initial_centroids[i, :] = X[random_choice[i], :]


    model = KMeans(n_clusters=no_clusters, init=initial_centroids, n_init=1).fit(X)
    y = model.predict(X)

    print('The output clusters are:', y)
    centroids = model.cluster_centers_
    print('The initial centroids were:\n', initial_centroids)
    print('The final centroids are:\n', centroids)

    # Count instances per cluster
    count = np.zeros((1, no_clusters))
    for cluster_label in range(no_clusters):
        count[0, cluster_label] = np.sum(y == cluster_label)

    print('Count of instances per cluster:\n', count)

# Visualization of clusters and centroids
    if X.shape[1] > 2:
        # Use PCA to reduce dimensionality if the data has more than 2 dimensions
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)
        centroids_transformed = pca.transform(centroids)
    else:
        X_transformed = X
        centroids_transformed = centroids

    plt.figure(figsize=(8, 6))
    
    # Scatter plot of data points
    for cluster in range(no_clusters):
        plt.scatter(X_transformed[y == cluster, 0], X_transformed[y == cluster, 1], label=f'Cluster {cluster + 1}')
    
    # Scatter plot of centroids
    plt.scatter(centroids_transformed[:, 0], centroids_transformed[:, 1], c='red', marker='X', s=200, label='Centroids')
    
    plt.title('KMeans Clustering Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Ask if the user wants to restart
    restart_choice = input('Do you want to start again from the beginning? (Yes/No):\n').strip().lower()
    if restart_choice != 'yes':
        break

print('Program has ended.')