#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from sklearn import metrics
import pandas as pd

import matplotlib.pyplot as plt
import random
import shutil
import os
from scipy.interpolate import make_interp_spline

# Classifier.
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,Flatten,MaxPooling2D, Activation, GlobalAveragePooling2D
from keras import optimizers
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import BatchNormalization

from dataset import DatasetCIFAR10

# Weights and biases
import wandb


# In[ ]:


# BatchAgent
from batch_envs import LalEnvFirstAccuracy
from batch_helpers import ReplayBuffer
from batch_dqn import DQN


# In[ ]:


# Start time.
import time
start_time = time.time()


# In[ ]:


wandb.login()


# ### Parameters.

# In[ ]:


# Classifier parameters.
CLASSIFIER_NUMBER_OF_CLASSES = 10
CLASSIFIER_NUMBER_OF_EPOCHS = 50
CLASSIFIER_LEARNING_RATE = 0.01
CLASSIFIER_BATCH_SIZE = 64

# Parameters for both agents.

REPLAY_BUFFER_SIZE = 5e4
PRIOROTIZED_REPLAY_EXPONENT = 3

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TARGET_COPY_FACTOR = 0.01
BIAS_INITIALIZATION = 0

NUMBER_OF_STATE_DATA = 1000
TRAIN_DATASET_LENGTH = 5000

# BatchAgent's parameters.

DIRNAME = './batch_agent/' # The resulting batch_agent of this experiment will be written in a file.

WARM_START_EPISODES_BATCH_AGENT = 4500
NN_UPDATES_PER_WARM_START_BATCH_AGENT = 100

TRAINING_EPOCHS_BATCH_AGENT = 1000
TRAINING_EPISODES_PER_EPOCH_BATCH_AGENT = 10
NN_UPDATES_PER_EPOCHS_BATCH_AGENT = 100

# Agent's parameters.

DIRNAME = './agent/' # The resulting batch_agent of this experiment will be written in a file.

WARM_START_EPISODES_AGENT = 1000
NN_UPDATES_PER_WARM_START_AGENT = 100

TRAINING_EPOCHS_AGENT = 1500
TRAINING_EPISODES_PER_EPOCH_AGENT = 10
NN_UPDATES_PER_EPOCHS_AGENT = 100


# In[ ]:


cwd = os.getcwd() # Find the current directory.

# Delete following directories if they exist.
shutil.rmtree(cwd+'/__pycache__', ignore_errors=True)
shutil.rmtree(cwd+'/batch_agent', ignore_errors=True)
shutil.rmtree(cwd+'/agent', ignore_errors=True)
shutil.rmtree(cwd+'/AL_results', ignore_errors=True)
shutil.rmtree(cwd+'/checkpoints', ignore_errors=True)
shutil.rmtree(cwd+'/summaries', ignore_errors=True)


# Initialise the dataset.

# In[ ]:


dataset = DatasetCIFAR10(number_of_state_data=NUMBER_OF_STATE_DATA, train_dataset_length=TRAIN_DATASET_LENGTH)
print("Train data are {}.".format(len(dataset.train_data)))
print("State data are {}.".format(len(dataset.state_data)))
print("Test data are {}.".format(len(dataset.test_data)))


# In[ ]:


classes = [0,1,2,3,4,5,6,7,8,9]
print("Train data.")
for data_class in classes:
    count = 0
    for i in range(len(dataset.train_labels)):
         if int(dataset.train_labels[i])==data_class:
                count+=1
    print("Class {}: {}.".format(data_class, count))
print("\n")
print("Test data.")
for data_class in classes:
    count = 0
    for i in range(len(dataset.test_labels)):
         if int(dataset.test_labels[i])==data_class:
                count+=1
    print("Class {}: {}.".format(data_class, count))
print("\n")
print("State data.")
for data_class in classes:
    count = 0
    for i in range(len(dataset.state_labels)):
         if int(dataset.state_labels[i])==data_class:
                count+=1
    print("Class {}: {}.".format(data_class, count))


# # Classifier.

# # BatchAgent

# In[ ]:


# Parameters.
input_shape = dataset.train_data.shape[1:4]
optimizer = optimizers.Adam(lr=0.0003)

# Create the classifier.
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
classifier.add(Dropout(0.5))
classifier.add(Dense(CLASSIFIER_NUMBER_OF_CLASSES, activation='softmax'))

# Compile classifier.
learning_rate = 0.01
decay = learning_rate/10
sgd = SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=False)
classifier.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# Initialise the BatchAgent environment.

# In[ ]:


batch_env = LalEnvFirstAccuracy(dataset, classifier, epochs=CLASSIFIER_NUMBER_OF_EPOCHS, classifier_batch_size=CLASSIFIER_BATCH_SIZE)


# Initialise the replay buffer.

# In[ ]:


replay_buffer = ReplayBuffer(buffer_size=REPLAY_BUFFER_SIZE, prior_exp=PRIOROTIZED_REPLAY_EXPONENT)


# ### Warm-start Episodes. | BatchAgent.

# In[ ]:


# Weights and Biases.
run = wandb.init(
    # Set the project where this run will be logged.
    project="RAL_Image_Classification_3",
    # Track hyperparameters and run metadata.
    config={
        "Training data": TRAIN_DATASET_LENGTH,
        "Test data": NUMBER_OF_STATE_DATA,
        "State data": NUMBER_OF_STATE_DATA,
        "Learning_rate": CLASSIFIER_LEARNING_RATE,
        "Classifier_epochs": CLASSIFIER_NUMBER_OF_EPOCHS,
        "Number_of_classes": CLASSIFIER_NUMBER_OF_CLASSES,
        "Dataset": "CIFAR10",
        "Classifier": "Simple CNN",
        "Warm-start episodes for the BatchAgent": WARM_START_EPISODES_BATCH_AGENT,
        "Training epochs for BatchAgent": TRAINING_EPOCHS_BATCH_AGENT,
        "Training episodes per epoch for BatchAgent": TRAINING_EPISODES_PER_EPOCH_BATCH_AGENT,
        "Warm-start episodes for the Agent": WARM_START_EPISODES_AGENT,
        "Training epochs for Agent": TRAINING_EPOCHS_AGENT,
        "Training episodes per epoch for Agent": TRAINING_EPISODES_PER_EPOCH_AGENT,
        "Using ε-greedy policy": "No",
        "Same classifier for both agents": "Yes",
        "BatchAgent termination condition": "Terminate a BatchAgent episode if two rewards are declining.",
        "Agent termination condition": "Terminate an Agent episode if we reach the higher average precision per class in comparison to BatchAgent AND budget size is smaller than BatchAgent's.",
        "Special condition": "No"
    }
)


# In[ ]:


# Keep track of episode duration to compute average.
episode_durations = []
episode_scores = []
episode_number = 1
episode_losses = []
episode_precisions = []
batch_bank = [0]

for _ in range(WARM_START_EPISODES_BATCH_AGENT):

    print("Episode {}.".format(episode_number))
    # Reset the environment to start a new episode.
    # The state value contains a vector representation of state of the environment (depends on the classifier).
    # The next_action contains a vector representations of all actions available to be taken at the next step.
    state, next_action, reward = batch_env.reset()
    done = False
    episode_duration = CLASSIFIER_NUMBER_OF_CLASSES

    # Before we reach a terminal state, make steps.
    while not done:

        # Choose a random action. Choose a random batch number (each number should be chosen once, to exmplore the whole range).
        next_action_integers = []
        for next_action_integer in range(len(next_action)):
            next_action_integers.append(next_action_integer+1)
        count = 0
        if len(next_action_integers) > 0:
            for number in batch_bank:
                if number in next_action_integers:
                    count += 1
                    if count < len(next_action_integers):
                        next_action_integers.remove(number)
        batch = random.choice(next_action_integers)
        batch_bank.append(batch)

        # Getting numbers from 0 to n_actions.
        inputNumbers =range(0,batch_env.n_actions)

        # Non-repeating using sample() function.
        batch_actions_indices = np.array(random.sample(inputNumbers, batch))
        action = batch
        next_state, next_action, reward, done = batch_env.step(batch_actions_indices)
        
        if next_action==[]:
            next_action.append(np.array([0]))

        # Store the transition in the replay buffer.
        replay_buffer.store_transition(state, action, reward, next_state, next_action, done)

        # Get ready for the next step.
        state = next_state
        episode_duration += batch
    wandb.log({"BatchAgent |  Warm-start Precision": batch_env.precision_bank[-1], "BatchAgent | Warm-start ACC": batch_env.episode_qualities[-1]*100, "BatchAgent | Warm-start Budget": episode_duration/len(dataset.train_data), "BatchAgent | Warm-start Loss": batch_env.episode_losses[-1]})
    episode_scores.append(batch_env.episode_qualities[-1])
    episode_losses.append(batch_env.episode_losses[-1])
    episode_durations.append(episode_duration)
    episode_precisions.append(batch_env.precision_bank[-1])
    episode_number+=1

# Compute the average episode duration of episodes generated during the warm start procedure.
av_episode_duration = np.mean(episode_durations)
BIAS_INITIALIZATION = -av_episode_duration/2


# Plots for warm-start episodes.

# Initialize the DQN for the BatchAgent.

# In[ ]:


batch_agent = DQN(experiment_dir=DIRNAME,
            observation_length=NUMBER_OF_STATE_DATA,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            target_copy_factor=TARGET_COPY_FACTOR,
            bias_average=BIAS_INITIALIZATION,
           )


# Do updates of the network based on the warm-start episodes.

# In[ ]:


for update in range(NN_UPDATES_PER_WARM_START_BATCH_AGENT):
    minibatch = replay_buffer.sample_minibatch(BATCH_SIZE)
    td_error = batch_agent.train(minibatch)
    replay_buffer.update_td_errors(td_error, minibatch.indices)


# ### Train BatchAgent.

# In[ ]:


max_ACC_per_epoch_training = []
loss_per_epoch_training = []
max_ACC_per_epoch_testing = []
loss_per_epoch_testing = []
budget_per_epoch = []
max_precision_per_epoch = []

for epoch in range(TRAINING_EPOCHS_BATCH_AGENT):

    print("TRAINING EPOCH {}.".format(epoch+1))
    
    # Simulate training episodes.
    episode_scores_training = []
    episode_losses_training = []
    episode_scores_testing = []
    episode_losses_testing = []
    episode_durations = []
    episode_precisions = []

    for training_episode in range(TRAINING_EPISODES_PER_EPOCH_BATCH_AGENT):

        # print("Episode {}.".format(training_episode+1))
        
        # Reset the environment to start a new episode.
        state, next_action, reward = batch_env.reset()
        done = False
        episode_duration = CLASSIFIER_NUMBER_OF_CLASSES
        
        # Run an episode.
        while not done:
            batch = batch_agent.get_action(state, next_action)
            batch = int(next_action[batch])

            inputNumbers =range(0,batch_env.n_actions)
            batch_actions_indices = np.array(random.sample(inputNumbers, batch))

            # Make another step.
            next_state, next_action, reward, done = batch_env.step(batch_actions_indices)

            # Store a step in the replay buffer.
            action = batch
            if next_action==[]:
                next_action.append(np.array([0]))
            replay_buffer.store_transition(state, action, reward, next_state, next_action, done)
            
            # Change the state of the environment.
            state = next_state
            episode_duration += batch

        epoch_episode = epoch.__str__() + "." + training_episode.__str__()
        episode_scores_training.append(batch_env.episode_training_qualities[-1])
        episode_losses_training.append(batch_env.episode_training_losses[-1])
        episode_scores_testing.append(batch_env.episode_qualities[-1])
        episode_losses_testing.append(batch_env.episode_losses[-1])
        episode_durations.append(episode_duration)
        episode_precisions.append(batch_env.precision_bank[-1])
    
    max_ACC_training = []
    max_ACC_testing = []
    ACC_training = 0
    ACC_testing = 0
    training_losses = []
    training_loss = 0
    testing_losses = []
    testing_loss = 0
    precision = []
    budgets = []
    budget = 0
    max_precision = max(episode_precisions)
    for i in range(len(episode_precisions)):
        if episode_precisions[i]==max_precision:
            budgets.append(episode_durations[i])
            max_ACC_training.append(episode_scores_training[i])
            max_ACC_testing.append(episode_scores_testing[i])
            training_losses.append(episode_losses_training[i])
            testing_losses.append(episode_losses_testing[i])
    budget = np.array(budgets).min()
    ACC_training = np.array(max_ACC_training).max()
    ACC_testing = np.array(max_ACC_testing).max()
    training_loss = np.array(training_losses).min()
    testing_loss = np.array(testing_loss).min()

    max_ACC_per_epoch_training.append(ACC_training)
    loss_per_epoch_training.append(training_loss)
    max_ACC_per_epoch_testing.append(ACC_testing)
    loss_per_epoch_testing.append(testing_loss)
    budget_per_epoch.append(budget)
    max_precision_per_epoch.append(max_precision)

    wandb.log({"BatchAgent Training ACC": ACC_training*100, "BatchAgent Testing ACC": ACC_testing*100, "BatchAgent Budget": budget/len(dataset.train_data), "BatchAgent Training Loss": training_loss, "BatchAgent Testing Loss": testing_loss, "BatchAgent Precision": max_precision})

    # NEURAL NETWORK UPDATES.
    for update in range(NN_UPDATES_PER_EPOCHS_BATCH_AGENT):
        # print("Update {}.".format(update+1))
        minibatch = replay_buffer.sample_minibatch(BATCH_SIZE)
        td_error = batch_agent.train(minibatch)
        replay_buffer.update_td_errors(td_error, minibatch.indices)


# Plots for training episodes.

# In[ ]:


# Find the maximum accuracy per validation epoch and their respective budgets.
budgets_for_max_precision = []
budget_for_max_precision = 0
scores_for_max_precision = []
score_for_max_precision = 0
max_precision = max(max_precision_per_epoch)
for i in range(TRAINING_EPOCHS_BATCH_AGENT):
    if max_precision_per_epoch[i]==max_precision:
        budgets_for_max_precision.append(budget_per_epoch[i])
        scores_for_max_precision.append(max_ACC_per_epoch_testing[i])
budget_for_max_precision = np.array(budgets_for_max_precision).min()
score_for_max_precision = np.array(scores_for_max_precision).max()
print("The maximum precision is {}.".format(max_precision))
print("The maximum accuracy is {}.".format(score_for_max_precision))
print("The budget for the maximum accuracy is {}.".format(budget_for_max_precision))


# # Agent

# In[ ]:


# Agent
from agent_envs import LalEnvFirstAccuracy
from agent_helpers import ReplayBuffer
from agent_dqn import DQN


# Initialise the environment.

# In[ ]:


TARGET_ACCURACY = score_for_max_precision
TARGET_BUDGET = budget_for_max_precision
TARGET_PRECISION = max_precision
agent_env = LalEnvFirstAccuracy(dataset, classifier, epochs=CLASSIFIER_NUMBER_OF_EPOCHS, classifier_batch_size=CLASSIFIER_BATCH_SIZE, target_precision=TARGET_PRECISION, target_budget=TARGET_BUDGET)


# Initialise the replay buffer.

# In[ ]:


replay_buffer = ReplayBuffer(buffer_size=REPLAY_BUFFER_SIZE, prior_exp=PRIOROTIZED_REPLAY_EXPONENT)


# ### Warm-start Episodes. | Agent.

# In[ ]:


# Keep track of episode duration to compute average.
episode_durations = []
episode_scores = []
episode_number = 1
episode_losses = []
episode_precisions = []

for _ in range(WARM_START_EPISODES_AGENT):
    
    print("Episode {}.".format(episode_number))
    # Reset the environment to start a new episode.
    # The state value contains a vector representation of state of the environment (depends on the classifier).
    # The next_action contains a vector representations of all actions available to be taken at the next step.
    state, next_action, reward = agent_env.reset()
    done = False
    episode_duration = CLASSIFIER_NUMBER_OF_CLASSES

    # Before we reach a terminal state, make steps.
    while not done:

        # Choose a random action.
        next_action_for_batch_agent = []
        for i in range(1, len(next_action.T)+1):
            next_action_for_batch_agent.append(np.array([i]))
        batch = batch_agent.get_action(state, next_action_for_batch_agent)
        batch = int(next_action_for_batch_agent[batch])

        # Getting numbers from 0 to n_actions.
        inputNumbers =range(0,agent_env.n_actions)

        # Non-repeating using sample() function.
        help_action = np.array(random.sample(inputNumbers, batch))
        action = next_action[:,help_action]
        next_state, next_action, reward, done = agent_env.step(help_action)

        # Store the transition in the replay buffer.
        buffer_action = []
        for _ in range(TRAIN_DATASET_LENGTH):
            buffer_action.append([0,0,0])
        for i in range(len(action.T)):
            buffer_action[i]=action.T[0]
        if next_action.size!=0:
            replay_buffer.store_transition(state, buffer_action, reward, next_state, next_action, done)
        else:
            done=True

        # Get ready for next step.
        state = next_state
        episode_duration += batch

    wandb.log({"Agent | Warm-start Precision": agent_env.precision_bank[-1], "Agent | Warm-start ACC": agent_env.episode_qualities[-1]*100, "Agent | Warm-start Budget": episode_duration/len(dataset.train_data), "Agent | Warm-start Loss": agent_env.episode_losses[-1]})
    episode_scores.append(agent_env.episode_qualities[-1])
    episode_losses.append(agent_env.episode_losses[-1])
    episode_durations.append(episode_duration)
    episode_precisions.append(agent_env.precision_bank[-1])
    episode_number+=1

# Compute the average episode duration of episodes generated during the warm start procedure.
av_episode_duration = np.mean(episode_durations)
BIAS_INITIALIZATION = -av_episode_duration/2


# Plots for warm-start episodes.

# In[ ]:


tf.reset_default_graph()


# Initialize the DQN agent.

# In[ ]:


agent = DQN(experiment_dir=DIRNAME,
            observation_length=NUMBER_OF_STATE_DATA,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            target_copy_factor=TARGET_COPY_FACTOR,
            bias_average=BIAS_INITIALIZATION,
            depth=TRAIN_DATASET_LENGTH,
           )


# Do updates of the network based on warm start episodes.

# In[ ]:


for _ in range(NN_UPDATES_PER_WARM_START_AGENT):
    
    minibatch = replay_buffer.sample_minibatch(BATCH_SIZE)
    td_error = agent.train(minibatch, depth=TRAIN_DATASET_LENGTH)
    replay_buffer.update_td_errors(td_error, minibatch.indices)


# # Train RL

# In[ ]:


max_ACC_per_epoch_training = []
loss_per_epoch_training = []
max_ACC_per_epoch_testing = []
loss_per_epoch_testing = []
budget_per_epoch = []
max_precision_per_epoch = []

for epoch in range(TRAINING_EPOCHS_AGENT):

    print("TRAINING EPOCH {}.".format(epoch+1))
    
    # Simulate training episodes.
    episode_scores_training = []
    episode_losses_training = []
    episode_durations = []
    episode_scores_testing = []
    episode_losses_testing = []
    episode_precisions = []

    for training_episode in range(TRAINING_EPISODES_PER_EPOCH_AGENT):

        print("Episode {}.".format(training_episode+1))
        
        # Reset the environment to start a new episode.
        state, next_action, reward = agent_env.reset()
        done = False
        episode_duration = CLASSIFIER_NUMBER_OF_CLASSES

        # Run an episode.
        while not done:

            # Choose a random action.
            batch_next_action = []
            for i in range(1, len(next_action.T)+1):
                batch_next_action.append(np.array([i]))
            batch = batch_agent.get_action(state, batch_next_action)
            batch = int(batch_next_action[batch])
            action = agent.get_action(state, next_action, batch)

            taken_action = next_action[:,action]

            # Make another step.
            next_state, next_action, reward, done = agent_env.step(action)

            # Store a step in replay buffer.
            buffer_action = []
            for _ in range(TRAIN_DATASET_LENGTH):
                buffer_action.append([0,0,0])
            for i in range(len(taken_action.T)):
                buffer_action[i]=taken_action.T[0]
                
            if len(next_action.T)!=0:
                replay_buffer.store_transition(state, buffer_action, reward, next_state, next_action, done)
            else:
                done=True
           
            # Change a state of environment.
            state = next_state            
            episode_duration += batch

        epoch_episode = epoch.__str__() + "." + training_episode.__str__()
        episode_scores_training.append(agent_env.episode_training_qualities[-1])
        episode_losses_training.append(agent_env.episode_training_losses[-1])
        episode_scores_testing.append(agent_env.episode_qualities[-1])
        episode_losses_testing.append(agent_env.episode_losses[-1])
        episode_durations.append(episode_duration)
        episode_precisions.append(agent_env.precision_bank[-1])

    max_ACC_training = []
    max_ACC_testing = []
    ACC_training = 0
    ACC_testing = 0
    training_losses = []
    training_loss = 0
    testing_losses = []
    testing_loss = 0
    precision = []
    budgets = []
    budget = 0
    max_precision = max(episode_precisions)
    for i in range(len(episode_precisions)):
        if episode_precisions[i]==max_precision:
            budgets.append(episode_durations[i])
            max_ACC_training.append(episode_scores_training[i])
            max_ACC_testing.append(episode_scores_testing[i])
            training_losses.append(episode_losses_training[i])
            testing_losses.append(episode_losses_testing[i])
    budget = np.array(budgets).min()
    ACC_training = np.array(max_ACC_training).max()
    ACC_testing = np.array(max_ACC_testing).max()
    training_loss = np.array(training_losses).min()
    testing_loss = np.array(testing_loss).min()

    max_ACC_per_epoch_training.append(ACC_training)
    loss_per_epoch_training.append(training_loss)
    max_ACC_per_epoch_testing.append(ACC_testing)
    loss_per_epoch_testing.append(testing_loss)
    budget_per_epoch.append(budget)
    max_precision_per_epoch.append(max_precision)

    wandb.log({"Agent Training ACC": ACC_training*100, "Agent Testing ACC": ACC_testing*100, "Agent Budget": budget/len(dataset.train_data), "Agent Training Loss": training_loss, "Agent Testing Loss": testing_loss, "Agent Precision": max_precision})

    # NEURAL NETWORK UPDATES.
    for update in range(NN_UPDATES_PER_EPOCHS_AGENT):
        # print("Update {}.".format(update+1))
        minibatch = replay_buffer.sample_minibatch(BATCH_SIZE)
        td_error = agent.train(minibatch, depth=TRAIN_DATASET_LENGTH)
        replay_buffer.update_td_errors(td_error, minibatch.indices)


# In[ ]:


# Find the maximum accuracy per validation epoch and their respective budgets.
budgets_for_max_precision = []
budget_for_max_precision = 0
scores_for_max_precision = []
score_for_max_precision = 0
max_precision = max(max_precision_per_epoch)
for i in range(TRAINING_EPOCHS_BATCH_AGENT):
    if max_precision_per_epoch[i]==max_precision:
        budgets_for_max_precision.append(budget_per_epoch[i])
        scores_for_max_precision.append(max_ACC_per_epoch_testing[i])
budget_for_max_precision = np.array(budgets_for_max_precision).min()
score_for_max_precision = np.array(scores_for_max_precision).max()
print("The maximum precision is {}.".format(max_precision))
print("The maximum accuracy is {}.".format(score_for_max_precision))
print("The budget for the maximum accuracy is {}.".format(budget_for_max_precision))


# In[ ]:


print("BatchAgent achieved precision {:.1f}% with accuracy {:.1f}% and {} samples.".format(TARGET_PRECISION*100, TARGET_ACCURACY*100, TARGET_BUDGET))
print("Agent achieved precision {:.1f}% with accuracy {:.1f}% and {} samples.".format(max_precision*100, score_for_max_precision*100, budget_for_max_precision))


# In[ ]:


# End time.
seconds = time.time() - start_time
print("Total run time is {}.".format(time.strftime("%H:%M:%S",time.gmtime(seconds))))


# In[ ]:


wandb.finish()

