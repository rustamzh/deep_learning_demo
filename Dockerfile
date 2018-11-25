FROM continuumio/miniconda:4.5.11
RUN conda config --add channels conda-forge
RUN conda config --add channels auto
COPY req.txt /app/req.txt
RUN conda create -n demo --file /app/req.txt
RUN apt-get install -y libsndfile1
RUN ln -s /opt/conda/envs/demo/lib/libhdf5.so /opt/conda/envs/demo/lib/libhdf5.so.10
RUN ln -s /opt/conda/envs/demo/lib/libhdf5_hl.so /opt/conda/envs/demo/lib/libhdf5_hl.so.10
COPY . /app
CMD ["/app/change_env.sh"]
