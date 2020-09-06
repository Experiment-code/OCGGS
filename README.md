# OCGGS
This is the experiment code for the OCGGS problem.

To run the code, you can build your own docker image in the following way:

```
cd the_path_you_store_OCGGS/OCGGS/docker
docker build -t="your docker image name" .
cd ..
docker/run_docker.sh your_docker_image_name
```
After you run the docker container, you can call our methods in the python environment, or run example python files in the "examples" folder.
