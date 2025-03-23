#this is test.py
import string
import numpy as np
from PIL import Image
import os
import logging
import argparse
from pickle import load
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.layers import add

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("GPU Enabled for inference")
    except RuntimeError as e:
        logging.error(f"GPU error: {e}")

# Parse command line arguments
ap = argparse.ArgumentParser(description="Test image captioning model")
ap.add_argument('-i', '--image', help="Path to a single image")
ap.add_argument('-d', '--directory', help="Directory of images to process")
ap.add_argument('-m', '--model', default='models2/model_9.h5', help="Path to the model")
ap.add_argument('-t', '--tokenizer', default='tokenizer.p', help="Path to the tokenizer")
ap.add_argument('-s', '--save', action='store_true', help="Save captioned images")
ap.add_argument('-o', '--output', default='output', help="Output directory for saved images")
args = vars(ap.parse_args())

# Create output directory if saving images
if args['save'] and not os.path.exists(args['output']):
    os.makedirs(args['output'])

def load_tokenizer(tokenizer_path):
    """Load the tokenizer from file"""
    try:
        return load(open(tokenizer_path, "rb"))
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        return None

def extract_features(filename, model):
    """Extract features from image using Xception model"""
    try:
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.array(image)
        
        # Handle images with 4 channels (RGBA)
        if image.shape[2] == 4: 
            image = image[..., :3]
            
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        
        feature = model.predict(image, verbose=0)
        return feature
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

def word_for_id(integer, tokenizer):
    """Convert a word ID to its string representation"""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    """Generate a description for an image"""
    try:
        in_text = 'start'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            pred = model.predict([photo, sequence], verbose=0)
            pred = np.argmax(pred)
            word = word_for_id(pred, tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
        return in_text
    except Exception as e:
        logging.error(f"Error generating description: {e}")
        return "Error generating description"

def define_model(vocab_size, max_length):
    """Define the model architecture"""
    # Features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # LSTM sequence model
    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256, use_cudnn=False)(se2)  # Removed use_cudnn=False

    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

def show_image_with_caption(img_path, description, save_path=None):
    """Display image with caption and optionally save it"""
    try:
        img = Image.open(img_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        
        # Clean up the caption by removing start and end tokens
        clean_caption = ' '.join(description.split()[1:-1]) if 'end' in description else ' '.join(description.split()[1:])
        plt.title(clean_caption, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            logging.info(f"Saved captioned image to {save_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Error displaying image: {e}")

def calculate_bleu(reference_descriptions, generated_description):
    """Calculate BLEU score for generated caption"""
    # This function can be used if you have reference captions
    references = [desc.split()[1:-1] for desc in reference_descriptions]  # Remove start/end
    candidate = generated_description.split()[1:-1]  # Remove start/end
    return corpus_bleu([references], [candidate])

def process_image(img_path, model, tokenizer, max_length, xception_model, save=False, output_dir=None):
    """Process a single image to generate a caption"""
    try:
        # Extract features
        photo = extract_features(img_path, xception_model)
        if photo is None:
            return
        
        # Generate description
        description = generate_desc(model, tokenizer, photo, max_length)
        
        # Display results
        filename = os.path.basename(img_path)
        logging.info(f"Image: {filename}")
        logging.info(f"Caption: {description}")
        
        # Show and optionally save the image with caption
        if save and output_dir:
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(output_dir, f"{base_name}_captioned.jpg")
            show_image_with_caption(img_path, description, save_path)
        else:
            show_image_with_caption(img_path, description)
            
        return description
    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")
        return None

def main():
    # Maximum sequence length
    max_length = 32
    
    # Load tokenizer
    tokenizer = load_tokenizer(args['tokenizer'])
    if tokenizer is None:
        return
    
    # Get vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    logging.info(f"Vocabulary size: {vocab_size}")
    
    # Load the model
    try:
        model = load_model(args['model'])
        logging.info(f"Model loaded successfully from {args['model']}")
    except:
        logging.info("Could not load model directly - defining architecture and loading weights...")
        model = define_model(vocab_size, max_length)
        model.load_weights(args['model'])
        logging.info(f"Model weights loaded from {args['model']}")
    
    # Load Xception model for feature extraction
    xception_model = Xception(include_top=False, pooling="avg")
    logging.info("Xception model loaded for feature extraction")
    
    # Process single image or directory
    if args['image']:
        process_image(
            args['image'], 
            model, 
            tokenizer, 
            max_length, 
            xception_model, 
            args['save'], 
            args['output']
        )
    elif args['directory']:
        img_dir = args['directory']
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        # Get all valid image files
        image_files = [
            os.path.join(img_dir, f) for f in os.listdir(img_dir) 
            if f.lower().endswith(valid_extensions)
        ]
        
        if not image_files:
            logging.warning(f"No image files found in {img_dir}")
            return
            
        logging.info(f"Processing {len(image_files)} images")
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            process_image(
                img_path, 
                model, 
                tokenizer, 
                max_length, 
                xception_model, 
                args['save'], 
                args['output']
            )
    else:
        logging.error("Please provide either an image file (-i) or a directory of images (-d)")

if __name__ == "__main__":
    main()


# python test-better-1C.py --image Flicker8k_Dataset/1859941832_7faf6e5fa9.jpg --save
# correct
