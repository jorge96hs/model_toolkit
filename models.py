import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler # , MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
import numpy as np



# Useful metrics

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


def get_pipe(**kwargs):
    """
    Get a pipeline with the specified steps.
    :param kwargs: Keyword arguments to build the pipeline
    :return: A pipeline with the specified steps
    :rtype: sklearn.compose.ColumnTransformer

    Example:

        modelpipe = {
            'numeric': {
                'vars': numerical_features,
                'impute': SimpleImputer(strategy = 'mean'),
                'scale': StandardScaler()
            }
            ,'categorical': {
                'vars': categorical_features,
                'impute': SimpleImputer(strategy = 'most_frequent'),
                'encode': OneHotEncoder(handle_unknown = 'ignore', sparse = False, drop = 'if_binary')
            }
        }

        pipeline = get_pipe(**modelpipe)
    """
    transformers = []
    for name, transformer in kwargs.items():
        steps = []
        for step, value in transformer.items():
            if step == 'vars':
                continue
            steps.append((step, value))
        transformers.append((name, Pipeline(steps = steps), transformer['vars']))

    return ColumnTransformer(transformers = transformers)

def make_model_bilstm_mlp(
        shape1,
        shape2,
        dense_dim=64,
        lstm_layers=1,
        lstm_dropout=0.0,
        dense_layers1=1,
        dense_dropout1=0.0,
        dense_layers2=1,
        dense_dropout2=0.0,
        dense_dropout3=0.0,
        learning_rate=0.001,
        output_bias=None,
        metrics=['accuracy']):
    """
    Makes a bidirecional LSTM / MLP model.
    :param int shape1: Shape of the non-sequential data
    :param int shape2: Shape of the sequential data
    :param int dense_dim: Shape of the dense layers
    :param int lstm_layers: Number of LSTM layers
    :param float lstm_dropout: Probability of dropping out a node in the lstm layer during training
    :param int dense_layers1: Number of dense layers for the first set of dense layers
    :param float dense_dropout1: Probability of dropping out a node in the first set of dense layers during training
    :param int dense_layers2: Number of dense layers for the second set of dense layers
    :param float dense_dropout2: Probability of dropping out a node in the second set of dense layers during training
    :param float dense_dropout3: Probability of dropping out a node in the last set of dense layers during training
    :param float learning_rate: Learning rate of the optimizer
    :param float output_bias: Output bias to the final layer
    :param list metrics: List of metrics to evaluate during compilation
    :return: A compiled model
    """
    # Input1
    input1 = tf.keras.Input(shape=(shape1,))

    for i in range(dense_layers1):
        if i == 0:
            x1 = tf.keras.layers.Dense(dense_dim, activation='relu')(input1)
        else:
            x1 = tf.keras.layers.Dense(dense_dim, activation='relu')(x1)
        x1 = tf.keras.layers.Dropout(dense_dropout1)(x1)

    # Input2
    input2 = tf.keras.Input(shape=(shape2, 1))

    for i in range(lstm_layers):
        return_sequences = ((lstm_layers - i) != 1)
        if i == 0:
            x2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                dense_dim,
                dropout=lstm_dropout,
                recurrent_dropout=lstm_dropout,
                return_sequences=return_sequences))(input2)
        else:
            x2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                dense_dim,
                dropout=lstm_dropout,
                recurrent_dropout=lstm_dropout,
                return_sequences=return_sequences))(x2)

    for i in range(dense_layers2):
        x2 = tf.keras.layers.Dense(dense_dim, activation='relu')(x2)
        x2 = tf.keras.layers.Dropout(dense_dropout2)(x2)

    # Cost function
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    else:
        output_bias = 'zeros'

    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(dense_dim, activation='relu')(x)
    x = tf.keras.layers.Dropout(dense_dropout3)(x)

    outputs = tf.keras.layers.Dense(
        1,  # len(y_train.unique()),
        activation='sigmoid',  # 'softmax',
        bias_initializer=output_bias)(x)

    # Model
    model = tf.keras.Model([input1, input2], outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss='binary_crossentropy',  # 'sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=metrics)

    return model


def plot_confusion_matrix(y_true, y_pred, cutoff = 0.5, normalize ='true'):
    """
    Plot a confusion matrix given true and predicted labels.

    :param array-like y_true: Ground truth (correct) target values.
    :param array-like y_pred: Estimated targets as returned by a classifier.
    :param float cutoff: Cutoff threshold for the predicted values, default is 0.5.
    :param str normalize: Specifies which normalization to apply on the confusion matrix.
    One of {'true', 'pred', 'all', None}, default is 'true'.
    :return: A ConfusionMatrixDisplay object that can be plotted.
    :rtype: ConfusionMatrixDisplay
    """
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true = y_true,
        y_pred =(y_pred > cutoff) * 1,
        normalize = normalize
    )

    return disp


def plot_roc(name, labels, predictions, **kwargs):
    """
    Plot the ROC curve for a given model.

    :param str name: Name of the model.
    :param array-like labels: True binary labels in range {0, 1} or {-1, 1}.
    :param array-like predictions: Target scores, can either be probability estimates of the positive class,
    confidence values, or non-thresholded measure of decisions.
    :param kwargs: Optional keyword arguments to pass to the plot.
    """
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 100.5])
    plt.ylim([-0.5, 100.5])
    plt.grid(True)


def get_indices_from_date_table(date_table, indices):
    """
    Get the list of indices from a date table.
    :param pandas.Series date_table: Cumulative sum of the count of dates in the DataFrame.
    :param list indices: List of indices to retrieve from the date_table.
    :returns: A list of indices.
    :rtype: list
    """
    indices_list = []
    for index in indices:
        if index == 0:
            indices_list += [i for i in range(date_table[index])]
        else:
            indices_list += [i for i in range(date_table[index - 1], date_table[index])]

    return indices_list


def get_ts_cv(df, datecol = 'Fecha', n_splits=3):
    """
    Get cross-validation indices for time series data.

    :param pandas.DataFrame df: Input DataFrame.
    :param str datecol: Column name containing the dates, default is 'Fecha'.
    :param int n_splits: Number of splits for cross-validation, default is 3.
    :returns: A list of cross-validation index tuples (train_indices, test_indices).
    :rtype: list
    """
    cv = []
    date_table = df.groupby(datecol)[datecol].count().cumsum()
    ts_split = TimeSeriesSplit(n_splits)

    for train_index, test_index in ts_split.split(date_table.index.to_list()):
        train_indices = get_indices_from_date_table(date_table, train_index)
        test_indices = get_indices_from_date_table(date_table, test_index)

        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        cv.append((train_indices, test_indices))

    return cv


def train_PCA(df, numeric_cols, categ_cols):
    """
    Train PCA for the input DataFrame.

    :param pandas.DataFrame df: Input DataFrame.
    :param list numeric_cols: List of numeric column names.
    :param list categ_cols: List of categorical column names.
    :returns: A tuple containing the trained PCA model and the transformed PCA data.
    :rtype: tuple
    """

    # Crear pipeline
    pcapipe = {
        'numeric': {
            'vars': numeric_cols,
            'impute': SimpleImputer(strategy='mean'),
            'scale': StandardScaler()
        }
        , 'categorical': {
            'vars': categ_cols,
            'impute': SimpleImputer(strategy='most_frequent'),
            'encode': OneHotEncoder(handle_unknown='ignore', sparse=False)
        }
    }
    pca_processor = get_pipe(**pcapipe)

    # Entrenar PCA
    pca = PCA(n_components=2)
    pcadata = pca.fit_transform(pca_processor.fit_transform(df[numeric_cols + categ_cols]))

    return pca, pcadata