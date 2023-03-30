from main import resnet34instance
from keras.metrics import CategoricalAccuracy, Precision, Recall
from keras.callbacks import CSVLogger, ModelCheckpoint
import pandas as pd
from dataset import train, val

resnet34instance.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[CategoricalAccuracy(), Precision(), Recall()])
resnet34instance.summary() 
#define callbacks
logdir = 'logs'
csv_logger = CSVLogger('training_history.csv')
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

# #Train the Model
history = resnet34instance.fit(train, epochs=30, validation_data=val, callbacks=[csv_logger, checkpoint])

#save the history in a datframe
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history_df.csv', index=False)