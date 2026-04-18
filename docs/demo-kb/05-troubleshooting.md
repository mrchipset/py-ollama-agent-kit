# Troubleshooting

## Ollama connection fails

If the agent cannot connect to Ollama, check the host value first. In the demo environment it points to a LAN address rather than localhost. Make sure the machine running the server is reachable and the model is available.

## Retrieved references look wrong

If search results are weak or off-topic, confirm that the Markdown file was added to the index and that the query uses the same terminology as the source text. Short demo corpora often depend heavily on headings and repeated keywords.

## RAG answers feel too broad

If the answer sounds generic, inspect the retrieved citations first. If the wrong sections are being pulled, try smaller chunks, better headings, or a more specific query.

## A file was updated but search still shows old text

This demo uses explicit file add. If a Markdown file changes, add it again so the index is refreshed. The same source path will replace the previous chunks for that file.

## Auto RAG is getting in the way

Use `--no-rag` for pure chat sessions. That is useful when you want to isolate the raw model behavior without retrieval context.