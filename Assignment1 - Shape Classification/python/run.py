from logisticRegressionTrain import logisticRegressionTrain
from logisticRegressionTest import logisticRegressionTest

# Dataset directory
train_dir = '../dataset/train'
test_dir = '../dataset/test'

# Train
w, b,min_y, max_y = logisticRegressionTrain(train_dir, 10, False)

# Test
t, test_err  = logisticRegressionTest(test_dir, w,b, min_y, max_y, False)
#print(w, t, test_err)