import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_score



class LalEnv(object):

    def __init__(self, dataset, model, epochs, classifier_batch_size, target_precision):

        # Initialize the environment with attributes: dataset, model, quality function and other attributes.
        self.dataset = dataset        
        self.model = model
        self.epochs = epochs
        self.classifier_batch_size = classifier_batch_size
        self.target_precision = target_precision
        self.total_length_of_train_data = len(self.dataset.train_data)

        self.number_of_classes = np.size(np.unique(self.dataset.train_labels)) # # The number of classes as a number of unique labels in the train dataset.    
        self.episode_qualities = [] # A list where testing quality at each iteration will be written.
        self.episode_losses= [] # A list where loss for testing at each iteration will be written:
        self.episode_training_qualities = [] # A list where training quality at each iteration will be written:
        self.episode_training_losses = [] # A list where training loss at each iteration will be written.
        self.rewards_bank = [] # Rewards bank to store the rewards.
        self.batch_bank = [] # Batch bank to store the batch size.
        self.precision_bank = [] # Precision bank to store the precision per class.
    
    def reset(self, number_of_first_samples=10, isBatchAgent=False, target_budget=1.0):

        # Sample initial data points.
        self.dataset.regenerate()
        self.episode_qualities.append(0)
        self.rewards_bank.append(0)
        self.batch_bank.append(0.1)
        self.precision_bank.append(0)
        self.episode_losses.append(2)
        self.isBatchAgent = isBatchAgent
        self.target_budget = target_budget

        # To train an initial classifier we need at least self.number_of_classes samples.
        if number_of_first_samples < self.number_of_classes:
            print('number_of_first_samples', number_of_first_samples, ' number of points is less than the number of classes', self.number_of_classes, ', so we change it.')
            number_of_first_samples = self.number_of_classes

        # Sample number_of_first_samples data points.
        self.indices_known = []
        self.indices_unknown = []
        for i in np.unique(self.dataset.train_labels):
            # First get 1 point from each class.
            cl = np.nonzero(self.dataset.train_labels==i)[0]
            # Ensure that we select random data points.
            indices = np.random.permutation(cl)
            self.indices_known.append(indices[0])
            self.indices_unknown.extend(indices[1:])
        self.indices_known = np.array(self.indices_known)
        self.indices_unknown = np.array(self.indices_unknown)

        # The self.indices_unknown now contains first all points of class_1, then all points of class_2 etc.
        # So, we permute them.
        self.indices_unknown = np.random.permutation(self.indices_unknown)

        # Then, sample the rest of the data points at random.
        if number_of_first_samples > self.number_of_classes:
            self.indices_known = np.concatenate(([self.indices_known, self.indices_unknown[0:number_of_first_samples-self.number_of_classes]]))
            self.indices_unknown = self.indices_unknown[number_of_first_samples-self.number_of_classes:]
            
        # BUILD AN INITIAL MODEL.

        # Get the data corresponding to the selected indices.
        known_data = self.dataset.train_data[self.indices_known,:]
        known_labels = self.dataset.train_labels[self.indices_known]

        known_labels_one_hot_encoding = keras.utils.to_categorical(known_labels, num_classes = self.dataset.number_of_classes)

        # Train a model using data corresponding to indices_known:
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks = [early_stopping, checkpoint]
        self.model._ckpt_saved_epoch = 0
        history = self.model.fit(known_data, known_labels_one_hot_encoding, batch_size=self.classifier_batch_size, epochs=self.epochs, verbose=0, validation_data=(self.dataset.test_data, self.dataset.test_labels_one_hot_encoding), callbacks=callbacks)

        # Compute banks:

        # Testing accuracy.
        new_score = history.history['val_acc'][-1]
        self.episode_qualities.append(new_score)

        # Compute the precision:
        predictions = self.model.predict(self.dataset.test_data)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.dataset.test_labels_one_hot_encoding, axis=1)
        precision_scores = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        self.precision_bank.append(precision_scores)

        # Batch.
        self.batch_bank.append(number_of_first_samples/len(self.dataset.train_data))

        # Testing loss.
        self.episode_losses.append(history.history['val_loss'][-1])

        # Training accuracy.
        self.episode_training_qualities.append(history.history['acc'][-1])

        # Training loss.
        self.episode_training_losses.append(history.history['loss'][-1])

        # Get the features categorizing the state.
        state, next_action = self._get_state()
        self.n_actions = np.size(self.indices_unknown)

        return state, next_action
        


    def step(self, batch_actions_indices):

        # The batch_actions_indices value indicates the positions
        # of the batch of data points in self.indices_unknown that we want to sample in unknown_data.
        # The index in train_data should be retrieved.
        selection_absolute = self.indices_unknown[batch_actions_indices]

        # Label a datapoint: add its index to known samples and remove from unknown.
        self.indices_known = np.concatenate((self.indices_known, selection_absolute))
        self.indices_unknown = np.delete(self.indices_unknown, batch_actions_indices)

        # Train a model with new labeled data:
        known_data = self.dataset.train_data[self.indices_known,:]
        known_labels = self.dataset.train_labels[self.indices_known]
        known_labels_one_hot_encoding = keras.utils.to_categorical(known_labels, num_classes = self.dataset.number_of_classes)
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks = [early_stopping, checkpoint]
        self.model._ckpt_saved_epoch = 0
        history = self.model.fit(known_data, known_labels_one_hot_encoding, batch_size=self.classifier_batch_size, epochs=self.epochs, verbose=0, validation_data=(self.dataset.test_data, self.dataset.test_labels_one_hot_encoding), callbacks=callbacks)

        # Get a new state.
        state, next_action = self._get_state()

        # Update the number of available actions.
        self.n_actions = np.size(self.indices_unknown)
        # Compute the quality of the current classifier.
        new_score = history.history['val_acc'][-1] # Testing accuracy.
        self.episode_qualities.append(new_score)

        # Compute the precision:
        predictions = self.model.predict(self.dataset.test_data)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.dataset.test_labels_one_hot_encoding, axis=1)
        precision_scores = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        self.precision_bank.append(precision_scores)

        self.batch_bank.append(len(batch_actions_indices)/len(self.dataset.train_data)) # Batch.
        self.episode_losses.append(history.history['val_loss'][-1]) # Testing loss.
        self.episode_training_qualities.append(history.history['acc'][-1]) # Training accuracy.
        self.episode_training_losses.append(history.history['loss'][-1]) # Training loss.

        # Compute the reward.
        reward = self._compute_reward()

        # Check if this episode terminated.
        done = self._compute_is_terminal()

        return state, next_action, self.indices_unknown,  reward, done
      
    def _get_state(self):

        # Compute current state. Use margin score.
        predictions = self.model.predict_proba(self.dataset.state_data)[:,0]
        predictions = np.array(predictions)
        idx = np.argsort(predictions)

        # The state representation is the sorted list of scores.
        state = predictions[idx]
        
        # Compute next_action.
        unknown_data = self.dataset.train_data[self.indices_unknown,:]

        next_action = []
        for i in range(1, len(unknown_data)+1):
            next_action.append(np.array([i]))

        return state, next_action
    
    def _compute_reward(self):
        reward = 0.0
        return reward
    
    def _compute_is_terminal(self):
        if self.n_actions==0:
            done = True
        else:
            done = False
        return done
        
    
    
class LalEnvFirstAccuracy(LalEnv): 

    def __init__(self, dataset, model, epochs, classifier_batch_size, target_precision):

        # Initialize the environment with its normal attributes.
        LalEnv.__init__(self, dataset, model, epochs, classifier_batch_size, target_precision)
    
    def reset(self, number_of_first_samples=10, isBatchAgent=False, target_budget=1.0):

        state, next_action = LalEnv.reset(self, number_of_first_samples=number_of_first_samples, isBatchAgent=isBatchAgent, target_budget=target_budget)
        current_reward = self._compute_reward()

        # Store the current rewatd.
        self.rewards_bank.append(current_reward)
        
        return state, next_action, self.indices_unknown, current_reward
       
    def _compute_reward(self):
        # Calculate the reward as a combination of accuracy and exploration.
        new_score = self.precision_bank[-1] / self.batch_bank[-1]
        previous_score = self.precision_bank[-2] / self.batch_bank[-2]
        reward = new_score - previous_score + 0.01 * (1 - np.random.rand())  # Add a bonus for exploration.
        self.rewards_bank.append(reward)
        return reward

    def _compute_is_terminal(self):
        done = False
        done = LalEnv._compute_is_terminal(self)
        percentage_precision = 85
        percentage_budget = 50
        # Check if the target accuracy has been exceeded.
        if self.isBatchAgent==True:
            #print("BATCH-AGENT TERMINATION CONDINTION.")
            if (((percentage_precision*self.target_precision)/100) <= self.precision_bank[-1]) and (((percentage_budget*self.target_budget)/100) > self.batch_bank[-1]):
                print("-- Exceed target precision with lower budget, so this is the end of the episode.")
                done = True
        else:
            # If the last three rewards are declining, then terminate the episode.
            #print("WARM-START TERMINATION CONDINTION.")
            if len(self.rewards_bank) >= 4:
                if self.rewards_bank[-1] < self.rewards_bank[-2] and self.rewards_bank[-2] < self.rewards_bank[-3] and self.rewards_bank[-3] < self.rewards_bank[-4]:
                    done = True
                    return done
            return done
        return done
    
    def return_episode_qualities(self):
        return self.episode_qualities
    
    def return_episode_precisions(self):
        return self.precision_bank