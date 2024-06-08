FROM amd64/ubuntu

WORKDIR /pptree

ADD .git ./.git/

RUN mkdir core
RUN apt-get update
RUN apt-get -y install curl git cmake gnupg build-essential gdb make python3 r-base
RUN Rscript -e "install.packages('Rcpp')"
RUN Rscript -e "install.packages('RcppEigen')"
RUN Rscript -e "install.packages('devtools')"

RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"
