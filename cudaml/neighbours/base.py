import torch


class KNeighborsClassifier():
	def __init__(self,):
		self.X = None
		self.y = None

	def fit(self,X,y):
		self.X = X.cuda()
		self.y = y.cuda()
	def predict(self,X,top_neighbors=3):
		dist = torch.norm(self.X - X, dim=1, p=None)
		knn = dist.topk(top_neighbors, largest=False)
		print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
