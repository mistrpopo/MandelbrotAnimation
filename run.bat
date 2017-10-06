cargo run --release
ffmpeg -r 50 -i image_0.%%03d.png  -c:v libx264 out.mp4
DEL *.png