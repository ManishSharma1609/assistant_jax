# JAX Documentation Helper


This is a simple web application made with RAG for building an JAX assistant using FAISS as a vectorstore. It answers questions about JAX based on JAX's official documentation.



https://github.com/user-attachments/assets/ac2ce9ed-eb37-4dfb-ba35-2dc14c476c14




## Tech Stack
Client: Streamlit

Vectorstore: FAISS 

## Environment Variables

To run this project, you will need to add the `OPENAI_API_KEY` environment variables to your .env file


## Run Locally

Clone the project

```bash
  git clone https://github.com/ManishSharma1609/assistant_jax.git
```

Go to the project directory

```bash
  cd assistant_jax
```


Install dependencies

```bash
  pip install -r requirements.txt
```

Start the flask server

```bash
  streamlit run main.py
```
