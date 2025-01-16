#/bin/zsh

#
# Make a QuickTime compatible video of files named [number].png in the present folder.
# Writes output to the folder one above.
# 
# Usage:
# sh make_video.sh [FPS (optional)]
#

CRF=25      # quality; lower the better. ~10 suitable for small videos, ~25 for large

FPS=120     # desired framerate
if [[ "$1" != "" ]]
then
    FPS="$1"
fi
echo $FPS

folder=${PWD##*/}
files=(`ls -Art | grep -E '^[0-9]+.png' | sort -n`)
CONCAT_FILE="concat_tmp_asdasd.txt"
rm -f $CONCAT_FILE

for f in "${files[@]}"
do
    echo "file '$f'"  >> $CONCAT_FILE
    echo "duration $((1.0 / $FPS))"  >> $CONCAT_FILE
done

ffmpeg -f concat -i $CONCAT_FILE -f mp4 -vcodec h264 -pix_fmt yuv420p -crf $CRF -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2",fps=$FPS ../$folder.mp4

rm -f $CONCAT_FILE
