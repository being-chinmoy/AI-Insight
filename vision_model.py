import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'dataset/damage_detection' # Placeholder path
MODEL_SAVE_PATH = 'damage_vision_model.h5'

def build_model():
    """
    Builds a Transfer Learning model using MobileNetV2 (Pre-trained on ImageNet).
    We use MobileNetV2 because it is lightweight and fast for deployment (perfect for our Support Agent).
    """
    print("   > Loading Base Model (MobileNetV2)...")
    # Load base model without the top (classification) layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze the base model to keep pre-trained patterns
    base_model.trainable = False
    
    # Add custom head for our binary classification (Damaged vs Intact)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Prevent overfitting
    predictions = Dense(1, activation='sigmoid')(x) # 1 Output: Probability of Damage
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("   > Model Structure Built.")
    return model

def train_vision_model():
    """
    Main training loop. 
    NOTE: This requires a directory structure:
        dataset/damage_detection/
            train/
                damaged/
                intact/
            validation/
                damaged/
                intact/
    """
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ Data directory '{DATA_DIR}' not found. Please arrange data to train.")
        return

    print("--- Preparing Data Generators ---")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'validation'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    # Build Model
    model = build_model()
    model.summary()
    
    # Train
    print(f"--- Starting Training for {EPOCHS} Epochs ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )
    
    # Save
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

def predict_damage(image_path):
    """
    Inference function for the Chatbot.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        print("⚠️ Model file not found. Please train first.")
        # Fallback simulation for demo purposes
        import random
        return random.random() # Random confidence
        
    print(f"   > Loading model from {MODEL_SAVE_PATH}...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Batch of 1
    img_array /= 255.0 # Normalize
    
    prediction = model.predict(img_array)[0][0]
    return prediction

if __name__ == "__main__":
    # Create a dummy file structure if running directly to show user what to do
    print("To train this model, please place your images in 'dataset/damage_detection/'")
    print("Running in DEMO mode (Building Architecture Only)...")
    model = build_model()
    model.summary()
