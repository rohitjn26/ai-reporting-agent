FROM cubejs/cube:v1.6.40

WORKDIR /

COPY package.json ./
RUN npm install

COPY cube-conf/ .
COPY data-models/ data-models/

EXPOSE 4000

CMD ["cubejs", "server"]
