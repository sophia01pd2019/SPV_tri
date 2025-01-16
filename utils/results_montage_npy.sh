#!/bin/zsh

#
# Creates a montage from npy data files of results at given time points (see below).
# If inkscape available in PATH, converts the montage svg to png.
#
# Depends:
# - npy_to_png.py
# - make_montage.py
# - inkscape (optional, to convert svg to png)
#

# Relative time points to include in the montage. You may modify this as desired.
timePoints=(0 1/32 1/16 1/8 1/4 1/2 1)

rm -rf __pycache__

folders=(`ls -d */ | sort -V`)    # folders where to look for images
cnt=0                   # folder counter 
tmpFiles=()             # temporary files for creating a montage
basedir=$(dirname "$0") # location of the script

# If no info.txt given, will generate one from the folder name tokens separated
# by underscores, minus the last item (timestamp/ID)
noinfo=false
if ! [ -e info.txt ]
then
    noinfo=true
fi

for folder in "${folders[@]}"
do
    # Generate the info.txt now if needed.
    if $noinfo
    then
        info=(`echo $folder | sed 's/_/\n/g'`)
        info[-1]=()     # remove the last item (timestamp/ID)
        for token in "${info[@]}"
        do
            echo -n $token >> info.txt
            echo -n " " >> info.txt
        done
        echo "" >> info.txt
    fi

    cd $folder
    files=(`ls -Art | grep -E '^[0-9]+.npy' | sort -n`)
    
    # Make temporary copies of the requested time points.
    for t in "${timePoints[@]}"
    do
        idx=$(((${#files[*]}-1) * $t))
        file=${files[@]:$idx:1}
        cp $file ../$cnt"_"$file
        tmpFiles+=($cnt"_"$file)
    done
    
    cd ..
    cnt=$((cnt+1))
done

# Convert npys to pngs:
python $basedir/npy_to_png.py

# Create image montage with number of columns == number of time points.
# Font size 9. Row info expected to be given in 'info.txt' in the present folder.
outName=${PWD##*/}
python $basedir/make_montage.py . $outName".svg" ${#timePoints[*]} 9 info.txt

# Clean-up
for file in "${tmpFiles[@]}"
do
    file="${file%.*}"
    rm $file".png"
    rm $file".npy"
done

# Make a PNG copy of the SVG for easier viewing. ImageMagic doesn't work for
# unknown reason.
inkscape $outName".svg" -d 300 -o $outName".png"
