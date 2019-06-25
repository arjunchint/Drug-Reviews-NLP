import numpy as np
import pandas as pd
import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import sklearn.metrics

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# Function to create ensemble model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
   # create model
   model = Sequential()
   model.add(Dense(12, input_dim=16, kernel_initializer=init, activation='relu'))
   model.add(Dense(8, kernel_initializer=init, activation='relu'))
   model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
   # Compile model
   model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
   return model

def get_train_test(whole, downsample = False, n_is_called = 1, seed = 1234):
    X = whole.drop(columns = ['is_recalled']).values
    y = whole['is_recalled'].values.reshape([-1, 1])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    y_test = y_test.reshape((-1,1))

    if downsample == False:
        y_train = y_train.reshape((-1,1))
        return X_train, y_train, X_test, y_test
    else:
        not_recalled_idx = [x for x in range(y_train.shape[0]) if y_train[x]==0]
        is_recalled_idx = [x for x in range(y_train.shape[0]) if y_train[x]==1]
        assert len(not_recalled_idx) + sum(y_train)[0] == y_train.shape[0]
        chose = np.random.choice(not_recalled_idx, int(n_is_called * sum(y_train)[0]), replace=False)
        X_train_downsample = np.vstack((X_train[chose, :],X_train[is_recalled_idx, :]))
        y_train_downsample = np.vstack((y_train[chose, :], y_train[y_train==1].reshape((-1, 1))))
        y_train_downsample = y_train_downsample.reshape((-1, 1))
        return X_train_downsample, y_train_downsample, X_test, y_test

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return


if __name__ == '__main__':
    print('Starting ML Model Generation')
    whole = pd.read_csv("drug_reviews_features_data.csv", sep=',')
    print("Dataset Shape: ", whole.shape)

    print(whole.head())

    whole = whole.drop(columns = ['drug_name', 'condition', 'review', 'rating', 'review_date', 'cleaned_words','partial_name','combined_sentiment','combined_vader','combined_stanford'])
    assert(whole.isnull().values.any() == False)
    y = whole['is_recalled'].values.reshape([-1, 1])
    X = whole.drop(columns = ['is_recalled']).values

    n_recalled = y.sum()
    n_not_recalled = len(y) - n_recalled
    print('# Recalled: ',n_recalled,'# Not Recalled: ',n_not_recalled,"Ratio: ",n_not_recalled / (len(y)))

    # MLP MODEL

    # inputs
    training_epochs = 1000
    learning_rate = 0.01

    cost_history = np.empty(shape=[1],dtype=float)

    X = tf.placeholder(tf.float32,[None,16])
    Y = tf.placeholder(tf.float32,[None,1])
    is_training=tf.Variable(True,dtype=tf.bool)

    num_inputs = 16    # 16 independent variables
    num_hid1 = 128
    num_hid2 = 64
    num_hid3 = 32
    num_output = 1 # is_recalled/not_recalled

    X_train, y_train, X_test, y_test = get_train_test(whole, downsample = True, n_is_called = 1)
    y_train_ = y_train.reshape((y_train.shape[0],1))
    y_test_ = y_test.reshape((y_test.shape[0],1))

    # models
    initializer = tf.contrib.layers.xavier_initializer()
    h1 = tf.layers.dense(X, num_hid1, activation=tf.nn.elu, kernel_initializer=initializer, name = 'h1')
    h2 = tf.layers.dense(h1, num_hid2, activation=tf.nn.elu, kernel_initializer=initializer, name = 'h2')
    h3 = tf.layers.dense(h2, num_hid3, activation=tf.nn.elu, kernel_initializer=initializer, name = 'h3')
    output = tf.layers.dense(h3, num_output, activation=None, name = 'output')

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    predicted = tf.round(tf.nn.sigmoid(output))
    correct_pred = tf.equal(predicted, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_losses = []
        train_accuracies = []
        for step in range(training_epochs + 1):
            sess.run(optimizer, feed_dict={X: X_train, Y: y_train})
            loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={
                                     X: X_train, Y: y_train})
            cost_history = np.append(cost_history, acc)
            train_losses.append(loss)
            train_accuracies.append(acc)
            if step % 100 == 0:
                print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                    step, loss, acc))
                
        # Test model and check accuracy
        plt.figure(1)
        plt.title('MLP: Loss Curve')
        plt.plot(range(0, len(train_losses)), train_losses,label='Training Loss')
        # plt.ylim(ymax=.8) 
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.figure(2)
        plt.title('MLP: Accuracy Curve')
        plt.plot(range(0, len(train_accuracies)), train_accuracies,label='Training Accuracy')
        # plt.ylim(ymax=.8) 
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.figure(3)
        acc_pred = sess.run(accuracy, feed_dict={X: X_test, Y: y_test})
        pred = sess.run(predicted, feed_dict={X: X_test, Y: y_test})
        cnf_matrix = confusion_matrix(y_test, pred)
        np.set_printoptions(precision=2)
        class_names = ['Recalled', 'Not Recalled']

        # plt.show()   
        plot_confusion_matrix(cnf_matrix,normalize=True, classes=class_names, title='MLP Normalized confusion matrix')

        precision = sklearn.metrics.precision_score(y_test, pred)
        f1 = sklearn.metrics.f1_score(y_test, pred)
        recall = sklearn.metrics.recall_score(y_test, pred)
        roc = sklearn.metrics.roc_auc_score(y_test, pred)

        print('MLP Metrics: ')
        print('MLP Test Accuracy:', acc_pred,'Precision: ', precision, ' F1: ', f1, ' Recall: ', recall, ' ROC AUC: ', roc)

    # Decision Tree
    X_train, y_train, X_test, y_test = get_train_test(whole, downsample = True, n_is_called = 1.8)
    dt_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=10, min_samples_leaf=1)
    dt_gini.fit(X_train, y_train)
    y_pred_gini = dt_gini.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred_gini)
    np.set_printoptions(precision=2)
    class_names = ['Recalled', 'Not Recalled']
    plt.figure(4)
    plot_confusion_matrix(cnf_matrix,normalize=True, classes=class_names, title='DTree Normalized confusion matrix')

    precision = sklearn.metrics.precision_score(y_test, y_pred_gini)
    f1 = sklearn.metrics.f1_score(y_test, y_pred_gini)
    recall = sklearn.metrics.recall_score(y_test, y_pred_gini)
    roc = sklearn.metrics.roc_auc_score(y_test, y_pred_gini)

    print('DTree Metrics: ')
    print('DTree Test Accuracy:', accuracy_score(y_test,y_pred_gini)*100,'Precision: ', precision, ' F1: ', f1, ' Recall: ', recall, ' ROC AUC: ', roc)

    # ENSEMBLE MODEL
    print('\nStarting Ensemble: Will take a while \n')
    X_train, y_train, X_test, y_test = get_train_test(whole)

    model = KerasClassifier(build_fn=create_model, verbose=0)
    # grid search epochs, batch size and optimizer
    optimizers = ['rmsprop', 'adam']
    init = ['glorot_uniform', 'normal', 'uniform']
    # epochs = [50, 100, 150]

    # To reduce time: only using 50 epochs
    epochs = [50]
    batches = [5, 10, 20]
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
       print("%f (%f) with: %r" % (mean, stdev, param))

    y_pred_ensemble = grid_result.predict(X_test)

    cnf_matrix = confusion_matrix(y_test, y_pred_ensemble)
    np.set_printoptions(precision=2)
    class_names = ['Recalled', 'Not Recalled']
    plt.figure(5)
    plot_confusion_matrix(cnf_matrix,normalize=True, classes=class_names, title='Ensemble Normalized confusion matrix')

    precision = sklearn.metrics.precision_score(y_test, y_pred_ensemble)
    f1 = sklearn.metrics.f1_score(y_test, y_pred_ensemble)
    recall = sklearn.metrics.recall_score(y_test, y_pred_ensemble)
    roc = sklearn.metrics.roc_auc_score(y_test, y_pred_ensemble)

    print('Ensemble Metrics: ')
    print('Ensemble Test Accuracy:', accuracy_score(y_test,y_pred_ensemble)*100,'Precision: ', precision, ' F1: ', f1, ' Recall: ', recall, ' ROC AUC: ', roc)

    plt.show()