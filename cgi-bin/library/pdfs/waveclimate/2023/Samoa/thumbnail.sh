rm *.png
for f in *.pdf; do magick convert -thumbnail x400 -background white -alpha remove "$f"[0] "${f%.pdf}.png"; done
echo Thumbnails Generated.

