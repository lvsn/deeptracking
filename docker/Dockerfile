FROM nvidia/cuda:8.0-cudnn5-devel
MAINTAINER Mathieu Garon <mathieu.garon.2@ulaval.ca>

ENV PATH /usr/local/bin:$PATH
ENV LANG C.UTF-8


RUN apt-get update && \
    apt-get install -y \
        curl wget vim \
        software-properties-common ipython3 \
        openexr libopenexr-dev zlib1g-dev \
        libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev \
        libffi-dev \
        libhdf5-dev \
        pkg-config \
        tmux htop strace ltrace \
        gdb \
        bzip2 git-core gfortran sudo && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*


RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

# Setup Scipy & cie
RUN cd / && pip install tqdm

# Configuration
ADD .bashrc /root/.bashrc
ADD .vimrc /root/.vimrc


RUN chsh -s /bin/bash

WORKDIR /root

CMD [ "/bin/bash" ]

RUN git clone https://github.com/torch/distro.git /root/torch --recursive && cd /root/torch && \
    sed -i "s/sudo -E //g" install-deps && sed -i "s/sudo //g" install-deps && ./install-deps && \
    ./install.sh

RUN /bin/bash -c 'cd ~ && git clone https://github.com/hughperkins/pytorch.git && cd pytorch && source ~/torch/install/bin/torch-activate && ./build.sh'
RUN /root/torch/install/bin/luarocks install rnn

RUN apt-get update && \
    apt-get install -y \
        libprotobuf-dev protobuf-compiler && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN /root/torch/install/bin/luarocks install loadcaffe
