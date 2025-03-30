import numpy as np
import pandas as pd
import os
from triaxial.triaxialdata import TriaxialData

class DataLoad:
  TRAIN = "train"
  TEST = "test"
  TEST2 = "test2"

  def __init__(self, data_dir):
    self.data_dir = data_dir

  def load_file(self, filepath):
    """
    Load single file for model fit/eval into 1D array
    """
    try:
      df = pd.read_csv(filepath, delimiter='\\s+', header=None)
    except FileNotFoundError as err:
      print(f"File Not Found: {err}")
    return df.values

  def load_xfiles(self, xfilepath, files):
    """
    Load files (test or train) for model fit/eval into 3D array

    Returns: column matrix: first two axes correspond to rows and columns of original arrays
      Third axis corresponds to the depth of the array, i.e. number of files
       in the group (0-5 for x,y,z (acc) + x,y,z (gyro))
    """
    loaded = list()

    for f in files:
        abs_file = "{}{}".format(xfilepath, f)
        csv_data = self.load_file(abs_file)
        # add np array to list
        loaded.append(csv_data)
    #print(f"Size of {group} loaded files: {len(loaded)}")
    # stack arrays sequence depth-wise
    loaded = np.dstack(loaded)
    return loaded

  def load_dataset_type(self, type):
    """
    Load dataset type: test or training files including the target labels
    
    Returns: 3D array of X, 1D array of y 
    """
    xfilepath = self.data_dir + '/body/'

    # 6x files ttl: 3x body_acc, 3x body_gyro
    xfiles = ['body_acc_x_'+type+'.txt',
             'body_acc_y_'+type+'.txt',
             'body_acc_z_'+type+'.txt',
             'body_gyro_x_'+type+'.txt',
             'body_gyro_y_'+type+'.txt',
             'body_gyro_z_'+type+'.txt']

    X = self.load_xfiles(xfilepath, xfiles)
    # load labels
    y = self.load_file(self.data_dir + '/y_'+type+'.txt')
    return X, y

  def load_all_datasets(self, show_results=False):
    """
    Convenience function returning all data for train and test datasets.

    Returns: TriaxialData for both train and test datasets
      TriaxialData is a custom object encapsulating the datasets and providing data access ease.

    """
    assert os.path.exists(self.data_dir), f"{self.data_dir} must be mounted to proceed with data loading!"
    
    X_train, y_train = self.load_dataset_type(DataLoad.TRAIN)
    X_test, y_test = self.load_dataset_type(DataLoad.TEST)
    X_test2, y_test2 = self.load_dataset_type(DataLoad.TEST2)

    # zero-offset target/class to match label_map
    y_train = y_train - 1
    y_test = y_test - 1
    y_test2 = y_test2 - 1

    # flatten y array to 1D
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    y_test2 = y_test2.flatten()

    if show_results:
      train_obs, test_obs = X_train.shape[0], X_test.shape[0]
      train_percent = round(train_obs/(train_obs+test_obs) * 100, 0)
      print(
        f"""Datasets Loaded => Split: train: {train_percent}, test: {100 - train_percent}\n
        [Training Set]:\n\tX_train {X_train.shape}, y_train {y_train.shape}
        [Test Set]:\n\tX_test {X_test.shape}, y_test {y_test.shape}
        [Test Set2]:\n\tX_test2 {X_test2.shape}, y_test2 {y_test2.shape}"""
      )

    # instantiate TriaxialData for easy data access
    taxial_train = TriaxialData(X_train, y_train)
    taxial_test = TriaxialData(X_test, y_test)
    taxial_test2 = TriaxialData(X_test2, y_test2)

    return taxial_train, taxial_test, taxial_test2
