# READ ME

conda create --name <env> --file requirements.txt

This project uses some code from the following github repo

https://github.com/daveredrum/Pointnet2.ScanNet

That code includes C++ extensions to PyTorch which are installed at runtime in the docker file (https://pytorch.org/tutorials/advanced/cpp_extension.html)

So therefore it is necessary to have cuda installed

The project also uses various other libraries so I have bundled the code into a Docker image for ease, this can be found at https://hub.docker.com/r/antmillar/semanticpoint

Docker GPU support is currently only available in Linux Images in Docker, and these cannot be run in windows yet

I have included some scenes in the app, if you'd like more there's some in the extra scenes folder you can load them in the app 


------------------------------------------------------------------------------------------------------------

# Instructions for Installation

------------------------------------------------------------------------------------------------------------


The easiest way to run the project is via a google compute engine instance

## INSTRUCTIONS FOR GOOGLE COMPUTE ENGINE

You'll need a google cloud account, this provides up to $300 free credit, of which this app should use a miniscule amount

Go to https://console.cloud.google.com/compute/

Create a new instance and Select instance from the Marketplace on the the left hand side 

Launch "Deep Learning VM" Image

See readmeImage1 for initial settings. It's important the graphics card is the same (Tesla T4). Deploy the Image

Navigate to https://console.cloud.google.com/compute/instances

Edit the Image, and enable HTTP so the app can be accessed from outside the image.

Connect to the instance with SSH button

Pull the docker image typing this line in sh: "sudo docker pull antmillar/pointsemantics"

run the docker image typing this line in sh: "sudo docker run --gpus all -p 5000:5000 antmillar/pointsemantics"

you should see the following 


	 * Serving Flask app "app" (lazy loading)
	 * Environment: production
	   WARNING: This is a development server. Do not use it in a production deployment.
	   Use a production WSGI server instead.
	 * Debug mode: off
	 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)


find the external IP for the app at https://console.cloud.google.com/compute/instances

navigate to [external_ip:5000] in your browser, the app should load

to see instances running type "docker ps -a"
to stop an instance type "docker stop [instance_id]"

Once you have finished with the application make sure to stop the instance and delete it




------------------------------------------------------------------------------------------------------------

If you have Linux installed you should be able to run the project locally if you have a cuda compatible GPU

## INSTRUCTIONS FOR LINUX:


Install Docker-CE for Linux: https://docs.docker.com/install/linux/docker-ce/ubuntu/

Install Nvidia-Docker : https://github.com/NVIDIA/nvidia-docker (this allows usage of GPUs in linux docker containers)

Install nvidia-container-runtime : https://github.com/NVIDIA/nvidia-container-runtime (this allows containers that build cuda libraries at runtime)


Pull image to local disk from Docker Hub (approx 10GB)

	sudo docker pull antmillar/semanticpoint

Run Image using GPUs (https://docs.docker.com/engine/reference/run/)

	sudo docker run --gpus all -p 5000:5000 antmillar/semanticpoint


The app should then be available on localhost:5000 or http://127.0.0.1:5000/


(You can also run the app without docker using the instructions in the windows section below + dependencies installed.)



----------------------------------------------------------------------------------------------------------------

I struggled to get CUDA working in Windows hence using Linux/VMs instead. However in theory it's possible to run the application in Windows if you can install the right dependencies. Which are:

CUDA version 10.1 (may need to manually configure ENV variables)
Python 3.6+
Anaconda/Miniconda
PyTorch 1.4.0

## INSTRUCTIONS FOR WINDOWS:

Then from the root folder of the app in command line run:

(probably worth creating a new conda Env or Venv first)


	pip install -r requirements.txt
	pip install flask
	pip install plyfile
	pip install scipy

	cd pointnet2 
	python setup.py install

	cd ..

	python app.py

The last line will launch the app at localhost:5000 or http://127.0.0.1:5000;


If you get an error saying No Cuda Kernel is Available, you may need to reinstall the cuda extensions to target your GFX card. You can do this as follows:

In the pointnet2 folder delete all the FOLDERS except src, then run:

	TORCH_CUDA_ARCH_LIST="7.5" python app.py

where 7.5 is the Compute Capability of your Nvidia Gfx card


The app was built on a laptop with GeForce GTX 960M (cc 5.0) and rarely ran out of Cuda Memory when running the model.










 
