# Run the backend server

```bash
python main.py
```


# Upload the file:

```bash
curl -X POST "http://127.0.0.1:5050/upload" -F "file=@people.mp4"
```
```bash
curl -X POST "http://127.0.0.1:5050/upload" -F "file=@C:\Users\Legion\Desktop\Video clip experiments\Spiderman.mp4"
```


# Give Prompt


```bash
curl -X POST "http://127.0.0.1:5050/prompt" \
-H 'Content-Type: application/json' \
-d '{ "prompt": "Find me clips with exactly two people", "filename": "people.mp4" }' \
--no-buffer
```
```bash
curl -N -X POST "http://localhost:5050/prompt" -H "Content-Type: application/json" -d "{\"prompt\":\"Find where the speaker tells differences between backpropagation and brain learning\",\"filename\":\"yt_video.mp4\"}"
```
```bash
curl -N -X POST "http://localhost:5050/prompt" -H "Content-Type: application/json" -d "{\"prompt\":\"Man in orange shirt\",\"filename\":\"Spiderman.mp4\"}"
```


# Download Video Clip

```bash
curl -o myclip.mp4 "http://127.0.0.1:5050/download/people.mp4"
```

