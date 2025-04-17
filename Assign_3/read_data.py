import pandas as pd

# Read the train data
train_data = pd.read_csv('Train.dat', sep=r'\s+')
print("\nTrain Data:")
print(train_data)
print("\nTrain Data Shape:", train_data.shape)

# Read the test data
test_data = pd.read_csv('Test.dat', sep=r'\s+')
print("\nTest Data:")
print(test_data)
print("\nTest Data Shape:", test_data.shape) 