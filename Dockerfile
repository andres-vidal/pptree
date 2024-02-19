FROM arm64v8/ubuntu

WORKDIR /pptree

ENV ASDF_DIR /root/.asdf

COPY .tool-versions Makefile conanfile.txt ./
ADD .git ./.git/

RUN mkdir core
RUN apt-get update
RUN apt-get -y install curl git cmake python3.10-venv gnupg build-essential gdb
RUN git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.14.0
RUN echo '\n. $HOME/.asdf/asdf.sh' >> ~/.bashrc
RUN echo '\n. $HOME/.asdf/completions/asdf.bash' >> ~/.bashrc
RUN echo '\n alias conan_env="source _conan/conanbuild.sh"' >> ~/.bashrc
RUN . $HOME/.asdf/asdf.sh && asdf plugin add make
RUN . $HOME/.asdf/asdf.sh && asdf plugin add conan https://github.com/amrox/asdf-pyapp.git
RUN . $HOME/.asdf/asdf.sh && asdf plugin add pre-commit
RUN . $HOME/.asdf/asdf.sh && asdf plugin add python
RUN . $HOME/.asdf/asdf.sh && asdf install
RUN . $HOME/.asdf/asdf.sh && conan profile detect
RUN . $HOME/.asdf/asdf.sh && make install
RUN . $HOME/.asdf/asdf.sh && make install-debug

RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"
