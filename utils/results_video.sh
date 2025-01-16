#!/bin/zsh

#
# Make a QuickTime compatible video of files named [number].png in the present folder.
# Writes output to the folder one above.
# 
# Usage:
# sh make_video.sh [FPS (optional)]
#

CRF=10      # quality; lower the better. ~10 suitable for small videos, ~25 for large

ORIGINAL_FPS=120  # original framerate
if [[ "$1" != "" ]]
then
    ORIGINAL_FPS="$1"
fi
SLOW_FACTOR=10     # Slowdown factor (3x duration in this case)
NEW_FPS=$((ORIGINAL_FPS / SLOW_FACTOR)) # Adjusted framerate
echo "Original FPS: $ORIGINAL_FPS"
echo "New FPS: $NEW_FPS"

folder=${PWD##*/}
files=(`ls -Art | grep -E '^[0-9]+.png' | sort -n`)
CONCAT_FILE="concat_tmp_asdasd.txt"
rm -f $CONCAT_FILE

# Scale duration to milliseconds (1000 ms = 1 second)
DURATION_MS=$((1000 / NEW_FPS))

for f in "${files[@]}"
do
    echo "file '$f'"  >> $CONCAT_FILE
    echo "duration $DURATION_MS"  >> $CONCAT_FILE
done

ffmpeg -r $NEW_FPS -f concat -safe 0 -i $CONCAT_FILE -f mp4 -vcodec h264 -pix_fmt yuv420p -crf $CRF -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,fps=$NEW_FPS" ../$folder.mp4

rm -f $CONCAT_FILE
