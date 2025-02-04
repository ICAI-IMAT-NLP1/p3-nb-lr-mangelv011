import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.shape[1] # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples
        class_priors: Dict[int, torch.Tensor] = {0:(labels==0).sum().float()/len(labels),
                                                 1:(labels==1).sum().float()/len(labels)}
        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # TODO: Estimate conditional probabilities for the words in features and apply smoothing
        conditional_probabilities: Dict[int, torch.Tensor] = {}

      
        for label in torch.unique(labels):
        
            class_features = features[labels == label]
            # Sumar la frecuencia de cada palabra en los ejemplos de la clase (sumamos sobre las filas)
            word_counts = torch.sum(class_features, dim=0)
            
            # Aplicar suavizado de Laplace: sumar delta a cada cuenta.
            smoothed_counts = word_counts + delta
            
            # Calcular el total de palabras (incluyendo el suavizado)
            total_count = torch.sum(smoothed_counts)
        
            probabilities = smoothed_counts / total_count
            
            conditional_probabilities[label.item()] = probabilities # son las probabilities de cada una de las palabras del vocab

        return conditional_probabilities

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # TODO: Calculate posterior based on priors and conditional probabilities of the words
        class_labels = list(self.class_priors.keys())
        log_posteriors: torch.Tensor = torch.empty(size=(len(class_labels),))
        for i in range(len(class_labels)): # para cada clase 
            class_prior = self.class_priors[class_labels[i]] # obtenemos su prior 
            # calculamos la log verosimilitud de la feature multiplicando cada ocurrencia de palabra por el logaritmo de su probabilidad condicional
            log_likelihood = torch.sum(feature * torch.log(self.conditional_probabilities[class_labels[i]])).item()
            log_posteriors[i] = torch.log(class_prior).item() + log_likelihood
            
        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # TODO: Calculate log posteriors and obtain the class of maximum likelihood 
        pred: int = None
        log_posteriors = self.estimate_class_posteriors(feature)
        pred = torch.argmax(log_posteriors).item()
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        probs: torch.Tensor = None
        log_posteriors = self.estimate_class_posteriors(feature)
        probs = torch.softmax(log_posteriors, dim=0)
        return probs
