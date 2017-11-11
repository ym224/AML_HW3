import numpy as np
import random
import math
from scipy.stats import multivariate_normal as mn
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing as pp

filename = 'faithful.dat.txt'
def loadData():
	data = np.loadtxt(filename, skiprows=26, dtype=np.float)
	return data[:,1:]

def plotData():
	plt.figure()
	plt.title('Old Faithful Geyser Eruption and Waiting Times')
	plt.scatter(data[:,0], data[:,1])
	plt.xlabel('Eruption Duration (min)')
	plt.ylabel('Waiting Time (min)')
	plt.savefig('Faithful_Eruption_Waiting_Times.png')


def gmm(K, threshold, initializer):
	sample_size = np.shape(data)[0]
	dim = np.shape(data)[1]

	# initialize to k random centers, with k=2
	centroids = data[random.sample(range(sample_size), dim)]

	# take initial guesses for mu1, sigma1^2, mu2, sigma2^2, pi 
	mu = centroids
	sigma = [np.eye(dim)] * K
	# initialize priors
	pi = 1/K
	
	if initializer == 'KMeans':
		model = KMeans(n_clusters=K)
		model.fit(data)	
		labels = model.labels_
		mu[0] = (data[np.where(labels == 0)]).mean(axis=0)
		mu[1] = (data[np.where(labels == 1)]).mean(axis=0)
		sigma[0] = np.cov((data[np.where(labels == 0)]).T)
		sigma[1] = np.cov((data[np.where(labels == 1)]).T)


	iterations = 0
	log_likelihoods = []
	mus = []
	while True:
		iterations += 1
		gamma = expectation_step(sample_size, mu, sigma, pi)
		mu, sigma, pi = maximization_step(sample_size, gamma, mu, sigma)
		log_likelihood = calculate_log_likelihood(sample_size, mu, sigma, pi)
		log_likelihoods.append(log_likelihood)
		mus.append(mu)
		if len(log_likelihoods) > 1:
			log_likelihood_delta = log_likelihoods[-1] - log_likelihoods[-2]
			if log_likelihood_delta < threshold:
				break
	return np.array(mus), iterations
			

# E-step: compute responsibilities
def expectation_step(sample_size, mu, sigma, pi):
	gamma = np.zeros(sample_size)

	for i in range(sample_size):
	    numerator = pi * mn.pdf(data[i], mu[1], sigma[1])
	    denominator = (1-pi) * mn.pdf(data[i], mu[0], sigma[0]) + numerator
	    gamma[i] = numerator/denominator
	    gamma[i] = np.nan_to_num(gamma[i])
	return gamma    


# M-Step: compute weighted means and variances
def maximization_step(sample_size, gamma, mu, sigma):
	mu[0] = np.dot(1-gamma, data)/np.sum(1-gamma)
	mu[1] = np.dot(gamma, data)/ np.sum(gamma)
	mu = np.nan_to_num(mu)
	sigma[0] = np.dot(1-gamma, np.square(data-mu[0]))/np.sum(1-gamma)
	sigma[1] = np.dot(gamma, np.square(data-mu[1]))/np.sum(gamma)
	sigma = np.nan_to_num(sigma)
	pi = np.sum(gamma/sample_size)
	return mu, sigma, pi

# calculate log likelihood
def calculate_log_likelihood(sample_size, mu, sigma, pi):
	for i in range(sample_size):
		log_likelihood = np.sum(np.log((1-pi) * mn.pdf(data[i], mu[0], sigma[0]) + pi * mn.pdf(data[i], mu[1], sigma[1])))
	return log_likelihood


def plot2DTrajectories(initializer):
	plt.figure()
	plt.title('Old Faithful Geyser Eruption and Waiting Times')
	plt.scatter(data[:,0], data[:,1])
	plt.plot(mus[:,0][:,0], mus[:,0][:,1],'rv-',markersize=5, linewidth=5)
	plt.plot(mus[:,1][:,0], mus[:,1][:,1],'mD-',markersize=5, linewidth=5)

	
	plt.xlabel('Eruption Duration (min)')
	plt.ylabel('Waiting Time (min)')
	plt.savefig(initializer + '_Trajectories_Faithful_Eruption_Waiting_Times.png')


#run GMM 50 iterations with different initial parameter guesses
def runGMM50Times(initializer):
	tot_iters = []
	for i in range(50):
		mus, iterations = gmm(2, 0.001, initializer)
		tot_iters.append(iterations)
	return tot_iters	

def plotHistogram(initializer):
	plt.figure()
	plt.hist(np.array(tot_iters))
	plt.xlabel('Iterations')
	plt.ylabel('Count of Iterations')
	plt.title('Distribution of Iterations Until Convergence')
	plt.savefig(initializer + '_Distribution_Iterations_Until_Convergence.png')

data = loadData()
plotData()
mus, iterations = gmm(2, 0.001, 'Random')
plot2DTrajectories('Random')
tot_iters = runGMM50Times('Random')
plotHistogram('Random')

mus, iterations = gmm(2, 0.001, 'KMeans')
plot2DTrajectories('KMeans')
tot_iters = runGMM50Times('KMeans')
plotHistogram('KMeans')

