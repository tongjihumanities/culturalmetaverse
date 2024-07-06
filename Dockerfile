FROM python:3.10.12

RUN pip install poetry==1.8.3

RUN poetry config virtualenvs.create false

COPY . .

RUN poetry install --no-interaction --no-ansi --no-root

EXPOSE 8080

RUN useradd -m manager

RUN echo 'manager:tongjihumanities' | chpasswd

RUN usermod -aG sudo manager

RUN useradd -m customer

RUN deluser customer sudo || true

USER customer

CMD exec uvicorn app.server:app --host 127.0.0.1 --port 8080

