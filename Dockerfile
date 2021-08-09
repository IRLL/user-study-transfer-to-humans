FROM python:3.9
WORKDIR optionmetrics

RUN apt-get update && apt-get install -y \
	xvfb \
	python-opengl

ADD ./ ./

ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install ./crafting
RUN pip install -r hippo_gym/requirements.txt

# RUN pip3 install --upgrade pip
# RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD hippo_gym/xvfb.sh
