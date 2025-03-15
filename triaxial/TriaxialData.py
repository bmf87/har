import numpy as np

class TriaxialData:
  ACCELEROMETER = "accelerometer"
  GYROSCOPE = "gyroscope"
  label_map = {
      0: "WALKING", 1: "WALKING_UPSTAIRS",
      2: "WALKING_DOWNSTAIRS", 3: "SITTING",
      4: "STANDING", 5: "LAYING"
  }
  
  def __init__(self, X_dataset, y_dataset):
    self.X_dataset = X_dataset
    self.y_dataset = y_dataset
    self.type = type

  def get_all_signals(self, obs):
    '''
    Returns all channels (x6) for a row/observation from
    the raw inertial signals dataset.

    Returns: column matrix: column matrix: 128 timesteps x 6 channels
        data will contain x,y,z tri-axial signals for body
        acceleration and gyroscope in 3D array.
    '''
    # get 128 timesteps for 2x (x,y,z)
    all_signals_component = self.X_dataset[obs][:, :]
    return all_signals_component.reshape(1, 128, 6)

  def get_signal_component(self, obs, component):
    '''
    Returns a row for a single component (0-5) for a row/observation from
    the raw inertial signal dataset.

     Returns: 1D array: 128 timesteps x 1 channel
      data will contain a single signal component (x, y, or z) for body acceleration
        or gyroscope data in 1D array.
    '''
    # get 128 timesteps 1D array for either x,y,z
    signal_component = self.X_dataset[obs][:, component]
    return signal_component

  def get_body_acc_signals(self, obs):
    '''
    Returns all columns for the accelerometer channels (0,1,2) for a row/observation
    from the raw inertial dataset.

    Returns: column matrix: 128 timesteps x 3 channels
      data will contain x,y,z tri-axial signals for body acceleration in 3D array.
    '''
    # get 128 timesteps for (x,y,z)
    x = self.X_dataset[obs][:, 0]
    y = self.X_dataset[obs][:, 1]
    z = self.X_dataset[obs][:, 2]
    acc_signal_component = np.dstack((x, y, z))
    return acc_signal_component

  def get_gyro_signals(self, obs):
    '''
    Returns all columns for the gyroscope channels (3,4,5) for a row/observation
    from the raw inertial signal dataset.

    Returns: column matrix: 128 timesteps x 3 channels
      data will contain x,y,z tri-axial signals for gyroscope in 3D array.
    '''
    # get 128 timesteps for (x,y,z)
    x = self.X_dataset[obs][:, 3]
    y = self.X_dataset[obs][:, 4]
    z = self.X_dataset[obs][:, 5]
    gyro_signal_component = np.dstack((x, y, z))
    return gyro_signal_component

  def get_sensor_signals(self, obs, sensor):
    '''
    Convenience function that returns all columns for accelerometer or gyroscope 
    channels for a row/observation from the raw inertial signal dataset.

    Returns: column matrix: 128 timesteps x 3 channels
      data will contain x,y,z tri-axial signals for required sensor in 3D array.
    '''
    signal_component = None
    if sensor.lower() == self.ACCELEROMETER:
      signal_component = self.get_body_acc_signals(obs)
    if sensor.lower() == self.GYROSCOPE:
      signal_component = self.get_gyro_signals(obs)
    return signal_component

  def get_signal_label(self, obs):
    return self.label_map[self.y_dataset[obs]]

  def get_signal_label_index(self, obs):
    return self.y_dataset[obs]