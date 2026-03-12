#!/bin/bash
# This script unzips all of the archive files stored in directories named <??>

for folder in ??; do
    for ziparch in "$folder"/*.zip; do
        if [ -f $ziparch ]; then
            unzip $ziparch -d ${ziparch%.zip}
            rm $ziparch
        fi
    done

    for redundfold in $folder/??-????/??; do
        if [ -d "$redundfold" ]; then
            stateyear=${redundfold%/*}
            mv $redundfold/* $stateyear
            rmdir $redundfold
        fi
    done

    for redundfold in $folder/??-????/??-????/??; do
        if [ -d "$redundfold" ]; then
            stateyear=${redundfold%/??-????/??}
            mv $redundfold/* $stateyear
            rmdir $redundfold
        fi
    done
    
done
