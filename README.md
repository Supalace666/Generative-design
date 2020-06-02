# Autoencoder for 3D car shapes
> Au autoencoder network trained by 3D car shapes with three manipulation functions built in it. This piece of work is based on the framework shared by Ivan Xu [[Link]](https://github.com/xuyanwen2012/interactive_generative_3d_shapes)

## Install

To run the code, npm needs to be installed. It is a package manager for the JavaScript programming language. Run the code below to install:

    npm install
    
## Usage

### Training

To train the autoencoder (this will automatically create a new model, or train from an existing model):

	./autoencoder.py train <num-epochs>

Where `<num-epochs>` is the number of epochs that the model will train for. ie if the model has been trained for 20000 epochs, training it for 50 will train it from 20000 -> 20050 epochs.


### Generation of results

#### Repredict (ie. use the trained autoencoder to reproduce N test + N train obj meshes)

	./autoencoder.py repredict <output-directory> <num-output-train-and-test-meshes> [--model <path-to-model-snapshot>]
	
#### Gen-random (ie. use the trained decoder to build N random models by sampling randomly from the encoder's latent space)

	./autoencoder.py gen-random <output-directory> <num-outputs> [--model <path-to-model-snapshot>]
	
#### Gen-input (ie. use the trained decoder to build a model by inputing latent space vectors)

	./autoencoder.py gen-input <output-directory>
	
The default value of the vectors are 0 and can be changed in the python script autoencoder.py
	
### Manipulations

#### Linearlly merge two shapes

	./autoencoder.py interpolate <key1 (from)> <key2 (to)> <interp>
	
Keys correspond to model names in the source dataset. You can list all of them with

	./autoencoder.py list-keys
	
`<interp>` specifies the step size, ie. 0.1 will step from 0.0, 0.1, 0.2, ..., 0.9, 1.0, and generate obj meshes for each of those.
	
#### Adding/Removing features of a car shape to/from another car shape

	./autoencoder.py add-features <key1 (add features to)> <key2 (add features from)>
	
Keys correspond to model names in the source dataset. You can list all of them with

	./autoencoder.py list-keys


## References 

<b>Exploring Generative 3D shapes using Autoencoder Networks</b> [[Link]](https://github.com/xuyanwen2012/interactive_generative_3d_shapes)


<b>Exploring Generative 3D Shapes Using Autoencoder Networks (Autodesk 2017)</b> [[Paper]](https://www.autodeskresearch.com/publications/exploring_generative_3d_shapes)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Exploring%20Generative%203D%20Shapes%20Using%20Autoencoder%20Networks.jpeg" /></p>
