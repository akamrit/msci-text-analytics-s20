# MSCI 641 Assignment


* The accuracies calculated using 3 different activation functions in the feed-forward neural network classifier model is given below-

	* Activation Function      Accuracy
	* ReLu                       59.8%
	* Sigmoid                    54.6%
	* tanh                       54.4%

* We see that, though without adding activation functions to our model will make our model look much simpler but adding activation functions but that model will be less powerful and will not be able to handle large and complex dataset. Therefore we have added activation function to our model to improve complexity. 
* By adding l2-norm regularization reduces the overfitting problem. Therefore we see that the performance of the model increases. And there is a slight improvement in accuracy rate too, within the 3 classifiers.
* By adding a Dropout of 0.02 helps us to do better generalization of data and is less likely to overfit the data. The neurons are dropped-out temporarily for better performance, that can be easily seen in the accuracy rate. Thus we can see an increment of approx 5% overall in the accuracy rate.