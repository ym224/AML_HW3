import numpy as np
import random
import math
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing as pp

filename = 'faithful.dat.txt'
def loadData():
	data = np.loadtxt(filename, skiprows=26, dtype=np.float)
	return data, data[:,1], data[:,2]

def plotData():
	plt.figure()
	plt.title('Old Faithful Geyser Eruption and Waiting Times')
	plt.scatter(eruption_time, waiting_time)
	plt.xlabel('Eruption Duration (min)')
	plt.ylabel('Waiting Time (min)')
	plt.savefig('Faithful_Eruption_Waiting_Times.png')


def gmm(data, K):
	sample_size = np.shape(data)[0]
	dim = np.shape(data)[1]
	# initialize to k random centers
	idx = random.sample(sample_size, dim)
	centoid = data[idx]

	mean_vec1, mean_vec2 = [], []

	mean_vec1.append(centoids[0])
	mean_vec2.append(centoids[1])

	mu = centoid;
	pi = np.zeros([1, K])
	sigma = [np.eye(dim)] * K

	weight = [1/K]*K
	resp = np.zeros(sample_size,K)

	        # E - Step
            
            ## Vectorized implementation of e-step equation to calculate the 
            ## membership for each of k -gaussians
            for k in range(self.k):
                R[:, k] = w[k] * P(mu[k], Sigma[k])

            ### Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
            
            log_likelihoods.append(log_likelihood)
            
            ## Normalize so that the responsibility matrix is row stochastic
            R = (R.T / np.sum(R, axis = 1)).T
            
            ## The number of datapoints belonging to each gaussian            
            N_ks = np.sum(R, axis = 0)
            
            
            # M Step
            ## calculate the new mean and covariance for each gaussian by 
            ## utilizing the new responsibilities
            for k in range(self.k):
                
                ## means
                mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis = 1).T
                x_mu = np.matrix(X - mu[k])
                
                ## covariances
                Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
                
                ## and finally the probabilities
                w[k] = 1. / n * N_ks[k]
            # check for onvergence
            if len(log_likelihoods) < 2 : continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps: break
        
        ## bind all results together
        from collections import namedtuple
        self.params = namedtuple('params', ['mu', 'Sigma', 'w', 'log_likelihoods', 'num_iters'])
        self.params.mu = mu
        self.params.Sigma = Sigma
        self.params.w = w
        self.params.log_likelihoods = log_likelihoods
        self.params.num_iters = len(log_likelihoods)       
        
        return self.params


def predict(self, x):
        p = lambda mu, s : np.linalg.det(s) ** - 0.5 * (2 * np.pi) **\
                (-len(x)/2) * np.exp( -0.5 * np.dot(x - mu , \
                        np.dot(np.linalg.inv(s) , x - mu)))
        probs = np.array([w * p(mu, s) for mu, s, w in \
            zip(self.params.mu, self.params.Sigma, self.params.w)])
        return probs/np.sum(probs)


def runKMeans():
	scaler = pp.MinMaxScaler()
    data = scaler.fit_transform(data) 
    
    y_pred = KMeans(n_clusters=2, random_state=150).fit_predict(data)
    plt.figure(1)
    plt.scatter(data[y_pred==0][:,0],data[y_pred==0][:,1],c='pink')
    plt.scatter(data[y_pred==1][:,0],data[y_pred==1][:,1],c=u'b')
    plt.title("K-means Clustering")
    plt.savefig("kmeans_plot.png")

# https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/ 
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def update_center(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

data, eruption_time, waiting_time = loadData()
plotData()

