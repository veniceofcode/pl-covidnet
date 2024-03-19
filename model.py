import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import onnx
import onnx_tf
import boto3

def create_model():
    # Define or load the model architecture
    base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))

    # Add custom layers on top of the base model
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)  # Assuming you have 10 classes for your new task

    # Create a new model with the modified architecture
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def load_weights_from_checkpoint(model, checkpoint_path):
    # Load weights from the checkpoint file stored in S3
    s3 = boto3.client('s3',
                      endpoint_url='https://minio-s3-minio-operator.apps.cluster-8hg7m.sandbox258.opentlc.com',
                      aws_access_key_id='ikTv0XdEUD9n6MmgR74f',
                      aws_secret_access_key='TlGkYU8pk12naKGCZU7z46sGOZebzvb9OFMoOy8w',
                      region_name='us-east')
    s3.download_file('covidnet', checkpoint_path, 'checkpoint_file.ckpt')

    # Load weights from the downloaded checkpoint file
    checkpoint = tf.train.load_checkpoint('checkpoint_file.ckpt')

    # Iterate through the layers of the model and assign the corresponding weights from the checkpoint
    for var_name in checkpoint.get_variable_to_dtype_map():
        try:
            model.get_layer(var_name).set_weights([checkpoint.get_tensor(var_name)])
        except ValueError:
            print(f"Skipping variable {var_name} as it doesn't exist in the model.")

def main():
    # Create model
    model = create_model()

    # Load weights from the desired checkpoint file in S3
    checkpoint_path = "path/to/your/checkpoint_file.ckpt"  # Update this path to point to your checkpoint file in S3
    load_weights_from_checkpoint(model, checkpoint_path)

    # Optionally, freeze some layers if you want to perform transfer learning with only the new layers being trained
    for layer in model.layers:
        layer.trainable = False

    # Convert the model to ONNX format
    onnx_model = onnx_tf.keras_to_onnx(model, 'pl-covidnet')

    # Save the ONNX model
    onnx.save_model(onnx_model, 'pl-covidnet.onnx')

if __name__ == "__main__":
    main()
