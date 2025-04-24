from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Nadam(1e-3),
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint(
    "deepfreq.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="auto",
    verbose=1
)
earlystop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    verbose=1,
    batch_size= batch_size,
    callbacks=[tensorboard, checkpoint, earlystop]
)

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.savefig("AUC.png")

y_class=[np.argmax(x) for x in Y_pred]
y_class[:5]

plot_sample(X_test,y_test_new,5)

predictions = model.predict(x_test_each_class)
predicted_class = np.argmax(predictions, axis=1)
predicted_class

def plot_actual_predicted(images, pred_classes):
  fig, axes = plt.subplots(1, 4, figsize=(16, 15))
  axes = axes.flatten()

  # plot
  ax = axes[0]
  dummy_array = np.array([[[0, 0, 0, 0]]], dtype='uint8')
  ax.set_title("Base reference")
  ax.set_axis_off()
  ax.imshow(dummy_array, interpolation='nearest')
  # plot image
  for k,v in images.items():
    ax = axes[int(k)]
    ax.imshow(v, cmap=plt.cm.binary)
    ax.set_title(f"True: %s \nPredict: %s" % (classes[k], classes[pred_classes[k]]))
    ax.set_axis_off()
  plt.tight_layout()
  plt.show()
plot_actual_predicted(images_dict, predicted_class)
plt.savefig('predited.png',dpi=300)

print(classification_report(y_test_new,y_class))

cm=tf.math.confusion_matrix(labels=y_test_new,predictions=y_class)
import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,fmt='d',cmap='summer')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig(".png")
