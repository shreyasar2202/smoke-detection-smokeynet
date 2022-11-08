FROM public.ecr.aws/lambda/python:3.9

# YOU NEED THIS YUM COMMAND IF IN REQUIREMENTS.txt
# we are trying to install opencv-python instead of opencv-python-headless
# HOWEVER, INCLUDE IT HERE ANYWAYS
# more on this here https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN yum install mesa-libGL -y

# Copy function code
COPY data_preprocess.py ./

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements.txt ./
RUN  pip3 install -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD ["data_preprocess.lambda_handler"]
#hmm