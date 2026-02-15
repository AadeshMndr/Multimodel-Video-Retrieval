# Run the backend server

```bash
python main.py
```


# Upload the file:

```bash
curl -X POST "http://127.0.0.1:5050/upload" -F "file=@people.mp4"
```


# Give Prompt


```bash
curl -X POST "http://127.0.0.1:5050/prompt" \
-H 'Content-Type: application/json' \
-d '{ "prompt": "Find me clips with exactly two people", "filename": "people.mp4" }' \
--no-buffer
```


# Download Video Clip

```bash
curl -o myclip.mp4 "http://127.0.0.1:5050/download/people.mp4"
```

